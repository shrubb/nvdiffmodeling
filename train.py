# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import time
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import nvdiffrast.torch as dr

import src.renderutils as ru
from src import obj
from src import util
from src import mesh
from src import texture
from src import render
from src import regularizer
from src import data
from src.mesh import Mesh

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Utility mesh loader
###############################################################################

def load_mesh(filename, mtl_override=None):
    name, ext = os.path.splitext(filename)
    if ext == ".obj":
        return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)
    assert False, "Invalid mesh file extension"

###############################################################################
# Loss setup
###############################################################################

def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relativel2":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

def optimize_mesh(
    FLAGS,
    out_dir,
    log_interval=10,
    mesh_scale=2.0
    ):

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mesh"), exist_ok=True)
    tensorboard_writer = SummaryWriter(out_dir)

    # ==============================================================================================
    #  Custom dataset (camera parameters, possibly random, possibly with images)
    # ==============================================================================================
    dataloader = data.get_dataloader(FLAGS.batch, FLAGS.custom_dataset, FLAGS.train_res)
    dataloader_iterator = iter(dataloader)
    dataset_has_images = dataloader.dataset.kind == data.DatasetKind.FIXED_CAMERAS_AND_IMAGES

    # Projection matrix; for now, only constant matrix is supported
    proj_mtx = dataloader.dataset.get_projection_matrix()

    SAVE_CAMERAS_AND_IMAGES = False
    if SAVE_CAMERAS_AND_IMAGES:
        cameras = []

    # Reference mesh
    ref_mesh = load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override)
    print("Ref mesh has %d triangles and %d vertices." % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0]))

    # Check if the training texture resolution is acceptable
    ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
    if 'normal' in ref_mesh.material:
        ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())
    if FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
        print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))

    # Base mesh
    base_mesh = load_mesh(FLAGS.base_mesh)
    print("Base mesh has %d triangles and %d vertices." % (base_mesh.t_pos_idx.shape[0], base_mesh.v_pos.shape[0]))
    print("Avg edge length: %f" % regularizer.avg_edge_length(base_mesh))

    # Create normalized size versions of the base and reference meshes. Normalized base_mesh is important as it makes it easier to configure learning rate.
    if dataset_has_images:
        normalized_base_mesh = base_mesh
    else:
        normalized_base_mesh = mesh.unit_size(base_mesh)
    normalized_ref_mesh = mesh.unit_size(ref_mesh)

    assert not FLAGS.random_train_res or FLAGS.custom_mip, "Random training resolution requires custom mip."

    # ==============================================================================================
    #  Initialize weights / variables for trainable mesh
    # ==============================================================================================
    trainable_list = []

    v_pos_opt = normalized_base_mesh.v_pos.clone().detach().requires_grad_(True)

    # Trainable normal map
    if FLAGS.textures_init == 'random':
        # initialize to (0,0,1) & make sure normals are always in positive hemisphere
        normal_map_init = np.array([0, 0, 1])
    elif FLAGS.textures_init == 'ref':
        normal_map_init = ref_mesh.material['normal']
    elif FLAGS.textures_init == 'base':
        normal_map_init = base_mesh.material['normal']
    else:
        raise ValueError(f"Unknown textures_init mode '{FLAGS.textures_init}'")
    normal_map_opt = texture.create_trainable(normal_map_init, FLAGS.texture_res, not FLAGS.custom_mip)

    # Setup Kd, Ks albedo and specular textures
    if FLAGS.textures_init == 'random':
        if FLAGS.layers > 1:
            kd_map_opt = texture.create_trainable(np.random.uniform(size=FLAGS.texture_res + [4], low=0.0, high=1.0), FLAGS.texture_res, not FLAGS.custom_mip)
        else:
            kd_map_opt = texture.create_trainable(np.random.uniform(size=FLAGS.texture_res + [3], low=0.0, high=1.0), FLAGS.texture_res, not FLAGS.custom_mip)

        ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
        ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=FLAGS.min_roughness, high=1.0)
        ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=1.0)
        ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip)
    elif FLAGS.textures_init == 'ref':
        kd_map_opt = texture.create_trainable(ref_mesh.material['kd'], FLAGS.texture_res, not FLAGS.custom_mip)
        ks_map_opt = texture.create_trainable(ref_mesh.material['ks'], FLAGS.texture_res, not FLAGS.custom_mip)
    elif FLAGS.textures_init == 'base':
        kd_map_opt = texture.create_trainable(base_mesh.material['kd'], FLAGS.texture_res, not FLAGS.custom_mip)
        ks_map_opt = texture.create_trainable(base_mesh.material['ks'], FLAGS.texture_res, not FLAGS.custom_mip)
    else:
        raise ValueError(f"Unknown textures_init mode '{FLAGS.textures_init}'")

    # Trainable displacement map
    displacement_map_var = None
    if FLAGS.subdivision > 0:
        displacement_map_var = torch.tensor(np.zeros(FLAGS.texture_res + [1], dtype=np.float32), dtype=torch.float32, device='cuda', requires_grad=True)

    # Add trainable arguments according to config
    if not 'position' in FLAGS.skip_train:
        trainable_list += [v_pos_opt]
    if not 'normal' in FLAGS.skip_train:
        trainable_list += normal_map_opt.getMips()
    if not 'kd' in FLAGS.skip_train:
        trainable_list += kd_map_opt.getMips()
    if not 'ks' in FLAGS.skip_train:
        trainable_list += ks_map_opt.getMips()
    if not 'displacement' in FLAGS.skip_train and displacement_map_var is not None:
        trainable_list += [displacement_map_var]

    # ==============================================================================================
    #  Setup material for optimized mesh
    # ==============================================================================================

    opt_material = {
        'bsdf'   : ref_mesh.material['bsdf'],
        'kd'     : kd_map_opt,
        'ks'     : ks_map_opt,
        'normal' : normal_map_opt
    }

    # ==============================================================================================
    #  Setup reference mesh. Compute tangentspace and animate with skinning
    # ==============================================================================================

    render_ref_mesh = mesh.compute_tangents(ref_mesh)

    # Compute AABB of reference mesh. Used for centering during rendering TODO: Use pre frame AABB?
    ref_mesh_aabb = mesh.aabb(render_ref_mesh.eval())

    # ==============================================================================================
    #  Setup base mesh operation graph, precomputes topology etc.
    # ==============================================================================================

    # Create optimized mesh with trainable positions
    opt_base_mesh = Mesh(v_pos_opt, normalized_base_mesh.t_pos_idx, material=opt_material, base=normalized_base_mesh)

    # Scale from [-1, 1] local coordinate space to match extents of the reference mesh
    if not dataset_has_images:
        opt_base_mesh = mesh.align_with_reference(opt_base_mesh, ref_mesh)
    # Compute smooth vertex normals

    opt_base_mesh = mesh.auto_normals(opt_base_mesh)

    # Set up tangent space
    opt_base_mesh = mesh.compute_tangents(opt_base_mesh)

    # Subdivide if we're doing displacement mapping
    if FLAGS.subdivision > 0:
        # Subdivide & displace optimized mesh
        subdiv_opt_mesh = mesh.subdivide(opt_base_mesh, steps=FLAGS.subdivision)
        opt_detail_mesh = mesh.displace(subdiv_opt_mesh, displacement_map_var, FLAGS.displacement, keep_connectivity=True)
    else:
        opt_detail_mesh = opt_base_mesh

    # Laplace regularizer
    if FLAGS.relative_laplacian:
        with torch.no_grad():
            orig_opt_base_mesh = opt_base_mesh.eval().clone()
        lap_loss_fn = regularizer.laplace_regularizer_const(opt_detail_mesh, orig_opt_base_mesh)
    else:
        lap_loss_fn = regularizer.laplace_regularizer_const(opt_detail_mesh)

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    optimizer  = torch.optim.Adam(trainable_list, lr=FLAGS.learning_rate)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002)))

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    # Background color
    if FLAGS.background == 'checker':
        background = torch.tensor(util.checkerboard(FLAGS.display_res, 8), dtype=torch.float32, device='cuda')
    elif FLAGS.background == 'white':
        background = torch.ones((1, FLAGS.display_res, FLAGS.display_res, 3), dtype=torch.float32, device='cuda')
    else:
        background = None

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    ang = 0.0
    img_loss_vec = []
    lap_loss_vec = []
    iter_dur_vec = []
    glctx = dr.RasterizeGLContext()
    for it in range(FLAGS.iter):
        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
        save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
        if display_image or save_image:
            eye = np.array(FLAGS.camera_eye)
            up  = np.array(FLAGS.camera_up)
            at  = np.array([0,0,0])
            a_mv =  util.lookAt(eye, at, up)
            a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]
            a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
            a_campos = np.linalg.inv(a_mv)[None, :3, 3]

            params = {'mvp' : a_mvp, 'lightpos' : a_lightpos, 'campos' : a_campos, 'resolution' : [FLAGS.display_res, FLAGS.display_res],
            'time' : 0}

            # Render images, don't need to track any gradients
            with torch.no_grad():
                # Center meshes
                _opt_ref    = mesh.center_by_reference(render_ref_mesh.eval(params), ref_mesh_aabb, mesh_scale)
                if dataset_has_images:
                    _opt_detail = opt_detail_mesh.eval(params)
                else:
                    _opt_detail = mesh.center_by_reference(opt_detail_mesh.eval(params), ref_mesh_aabb, mesh_scale)

                # Render
                if FLAGS.subdivision > 0:
                    if dataset_has_images:
                        _opt_base = opt_base_mesh.eval(params)
                    else:
                        _opt_base   = mesh.center_by_reference(opt_base_mesh.eval(params), ref_mesh_aabb, mesh_scale)
                    img_base = render.render_mesh(glctx, _opt_base, a_mvp, a_campos, a_lightpos, FLAGS.light_power, FLAGS.display_res,
                        num_layers=FLAGS.layers, background=background, min_roughness=FLAGS.min_roughness)
                    img_base = util.scale_img_nhwc(img_base, [FLAGS.display_res, FLAGS.display_res])

                img_opt = render.render_mesh(glctx, _opt_detail, a_mvp, a_campos, a_lightpos, FLAGS.light_power, FLAGS.display_res,
                    num_layers=FLAGS.layers, background=background, min_roughness=FLAGS.min_roughness, ambient_only=FLAGS.ambient_only)
                img_ref = render.render_mesh(glctx, _opt_ref, a_mvp, a_campos, a_lightpos, FLAGS.light_power, FLAGS.display_res,
                    num_layers=1, spp=FLAGS.spp, background=background, min_roughness=FLAGS.min_roughness)

                val_img_loss = image_loss_fn(img_opt, img_ref).item()
                tensorboard_writer.add_scalar(f"{FLAGS.loss}, validation", val_img_loss, it)

                # Rescale
                img_opt  = util.scale_img_nhwc(img_opt,  [FLAGS.display_res, FLAGS.display_res])
                img_ref  = util.scale_img_nhwc(img_ref,  [FLAGS.display_res, FLAGS.display_res])

                result_image = [img_opt]
                if FLAGS.subdivision > 0:
                    result_image.insert(0, img_base)
                if not dataset_has_images:
                    result_image.append(img_ref)
                result_image = torch.cat(result_image, axis=2)

            result_image[0] = util.tonemap_srgb(result_image[0])
            np_result_image = result_image[0].cpu().numpy()
            if display_image:
                util.display_image(np_result_image, size=FLAGS.display_res, title='%d / %d' % (it, FLAGS.iter))
            if save_image:
                util.save_image(out_dir + '/' + ('img_%06d.png' % img_cnt), np_result_image)
                img_cnt = img_cnt+1

        # ==============================================================================================
        #  Initialize training
        # ==============================================================================================
        iter_start_time = time.time()
        img_loss = torch.zeros([1], dtype=torch.float32, device='cuda')
        lap_loss = torch.zeros([1], dtype=torch.float32, device='cuda')

        iter_res = FLAGS.train_res
        iter_spp = FLAGS.spp
        if FLAGS.random_train_res:
            # Random resolution, 16x16 -> train_res. Scale up sample count so we always land close to train_res*samples_per_pixel samples
            iter_res = np.random.randint(16, FLAGS.train_res+1)
            iter_spp = FLAGS.spp * (FLAGS.train_res // iter_res)

        # ==============================================================================================
        #  Build transform stack for minibatching
        # ==============================================================================================
        batch = next(dataloader_iterator)

        if dataset_has_images:
            r_mv, color_ref, foreground_masks = batch
            color_ref = color_ref.cuda()
            foreground_masks = foreground_masks.cuda()

            if FLAGS.random_train_res:
                raise NotImplementedError(
                    "Datasets with custom images not supported with --random_train_res")
        else:
            r_mv, = batch
            color_ref = None
            foreground_masks = None

        mvp = torch.as_tensor(proj_mtx)[None] @ r_mv
        campos = r_mv.inverse()[:, :3, 3]

        if FLAGS.constant_training_light is not None:
            lightpos = torch.as_tensor(FLAGS.constant_training_light).expand(FLAGS.batch, 3)
        else:
            lightpos = torch.zeros((FLAGS.batch, 3),   dtype=torch.float32)
            for lightpos_sample, campos_sample in zip(lightpos, campos):
                lightpos_sample[:] = torch.as_tensor(
                    util.cosine_sample(campos_sample.numpy()) * data.RADIUS)

        params = {'mvp' : mvp, 'lightpos' : lightpos, 'campos' : campos, 'resolution' : [iter_res, iter_res], 'time' : 0}

        # Random bg color
        randomBgMean = torch.rand(FLAGS.batch, 1, 1, 3, device='cuda')
        randomBgNoise = torch.rand(FLAGS.batch, iter_res, iter_res, 3, device='cuda')
        randomBackground = (randomBgMean + randomBgNoise * 0.1).clamp(0, 1)

        # ==============================================================================================
        #  Evaluate all mesh ops (may change when positions are modified etc) and center/align meshes
        # ==============================================================================================
        _opt_ref  = mesh.center_by_reference(render_ref_mesh.eval(params), ref_mesh_aabb, mesh_scale)
        if dataset_has_images:
            _opt_detail = opt_detail_mesh.eval()
        else:
            _opt_detail = mesh.center_by_reference(opt_detail_mesh.eval(params), ref_mesh_aabb, mesh_scale)

        # ==============================================================================================
        #  Render reference mesh (if not available from dataloader)
        # ==============================================================================================
        if color_ref is None:
            with torch.no_grad():
                color_ref = render.render_mesh(glctx, _opt_ref, mvp, campos, lightpos, FLAGS.light_power, iter_res,
                    spp=iter_spp, num_layers=1, background=randomBackground, min_roughness=FLAGS.min_roughness,
                    ambient_only=FLAGS.ambient_only_reference)
        else:
            color_ref = color_ref + (1.0 - foreground_masks) * randomBackground

        if SAVE_CAMERAS_AND_IMAGES:
            import cv2
            img = (color_ref[0].cpu().numpy() * 255.0).round().astype(np.uint8)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
            cv2.imwrite("data/spot-40FixedViews/%03d.png" % it, img)

            cameras.append(r_mv)

        # ==============================================================================================
        #  Render the trainable mesh
        # ==============================================================================================
        color_opt = render.render_mesh(glctx, _opt_detail, mvp, campos, lightpos, FLAGS.light_power, iter_res,
            spp=iter_spp, num_layers=FLAGS.layers, msaa=True , background=randomBackground,
            min_roughness=FLAGS.min_roughness, ambient_only=FLAGS.ambient_only)

        # Debugging output
        if log_interval and it % log_interval == 0:
            with torch.no_grad():
                image_for_tensorboard = torch.cat((color_opt[:2], color_ref[:2]), dim=2)
                image_for_tensorboard = image_for_tensorboard.view(-1, *image_for_tensorboard.shape[2:])
                image_for_tensorboard.clamp_(0, 1)
                image_for_tensorboard = image_for_tensorboard.cpu()

            tensorboard_writer.add_image("Render + GT (online)", image_for_tensorboard, it, dataformats='HWC')

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        # Image-space loss
        img_loss = image_loss_fn(color_opt, color_ref)

        # Compute laplace loss
        lap_loss = lap_loss_fn.eval(params)

        # Log losses
        img_loss_vec.append(img_loss.item())
        lap_loss_vec.append(lap_loss.item())
        tensorboard_writer.add_scalar(f"{FLAGS.loss}, training (online)", img_loss_vec[-1], it)
        tensorboard_writer.add_scalar(f"Laplacian loss, training (online)", lap_loss_vec[-1], it)

        # Schedule for laplacian loss weight
        if it == 0:
            if FLAGS.laplacian_factor is not None:
                lap_fac = FLAGS.laplacian_factor
            else:
                ratio = 0.1 / lap_loss.item() # Hack that assumes RMSE ~= 0.1
                lap_fac = ratio * 0.25
            min_lap_fac = lap_fac * 0.02
        else:
            lap_fac = (lap_fac - min_lap_fac) * 10**(-it*0.000001) + min_lap_fac

        # Compute total aggregate loss
        total_loss = img_loss + lap_loss * lap_fac

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================

        normal_map_opt.clamp_(min=-1, max=1)
        kd_map_opt.clamp_(min=0, max=1)
        ks_map_opt.clamp_rgb_(minR=0, maxR=1, minG=FLAGS.min_roughness, maxG=1.0, minB=0.0, maxB=1.0)

        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Log & save outputs
        # ==============================================================================================

        # Print/save log.
        if log_interval and (it % log_interval == 0):
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            lap_loss_avg = np.mean(np.asarray(lap_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))

            remaining_time = (FLAGS.iter-it)*iter_dur_avg
            print("iter=%5d, img_loss=%.6f, lap_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" %
                (it, img_loss_avg, lap_loss_avg*lap_fac, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))

        if log_interval and it % (log_interval * 10) == 0 or it == FLAGS.iter - 1:
            # Save final mesh to file
            obj.write_obj(os.path.join(out_dir, "mesh/"), opt_detail_mesh.eval())

    if SAVE_CAMERAS_AND_IMAGES:
        cameras = torch.cat(cameras).numpy()
        from generate_camera_matrices import write_cameras_to_file
        write_cameras_to_file(cameras, "data/spot-40FixedViews/cameras.txt", ["%03d.png" % i for i in range(len(cameras))])

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

def main():
    import random
    random.seed(777)
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    np.random.seed(777)

    parser = argparse.ArgumentParser(description='diffmodeling')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train_res', type=int, default=512)
    parser.add_argument('-rtr', '--random_train_res', action='store_true', default=False)
    parser.add_argument('-dr', '--display_res', type=int, default=None)
    parser.add_argument('-tr', '--texture_res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display_interval', type=int, default=0)
    parser.add_argument('-si', '--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=40)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-lp', '--light_power', type=float, default=5.0)
    parser.add_argument('-mr', '--min_roughness', type=float, default=0.08)
    parser.add_argument('-sd', '--subdivision', type=int, default=0)
    parser.add_argument('-mip', '--custom_mip', action='store_true', default=False)
    parser.add_argument('-rt', '--textures_init', type=str,
        choices=['random', 'ref', 'base'], default='ref',
        help="How to initialize ks/kd/normals: randomly, from reference mesh, or from base mesh")
    parser.add_argument('-lf', '--laplacian_factor', type=float, default=None)
    parser.add_argument('-rl', '--relative_laplacian', type=bool, default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relativel2'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base_mesh', type=str)
    parser.add_argument('--custom_dataset', type=str, default=None,
        help="Where to look for custom camera parameters (path to a text file) and possibly "
        "images (path to a directory with cameras.txt). "
        "File format is explained in src/data.py:MultiViewDataset.")
    parser.add_argument('--constant_training_light', type=list, default=None,
        help="During training, place light source at this fixed position instead of always "
        "sampling that position randomly (example: [0.82547714, -2.4586678, 2.35022099])")
    parser.add_argument('--ambient_only', action='store_true', default=False)
    parser.add_argument('--ambient_only_reference', action='store_true', default=False)
    parser.add_argument('--camera_eye', type=list, default=[0.0, 0.0, data.RADIUS])
    parser.add_argument('--camera_up', type=list, default=[0.0, 1.0, 0.0])
    parser.add_argument('--skip_train', type=list, default=[],
        help="Any of: 'position', 'normal', 'kd', 'ks', 'displacement'")
    parser.add_argument('--displacement', type=float, default=0.15)
    parser.add_argument('--mtl_override', type=str, default=None,
        help="Path to custom .mtl for reference mesh")

    FLAGS = parser.parse_args()

    if FLAGS.config is not None:
        with open(FLAGS.config) as f:
            new_data = json.load(f)
            for key in new_data:
                print(key, new_data[key])
                if key not in FLAGS:
                    raise KeyError(f"Unknown keyword '{key}' in config file")
                FLAGS.__dict__[key] = new_data[key]

    # Dynamic defaults
    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res

    if FLAGS.out_dir is None:
        FLAGS.out_dir = 'debug'
    out_dir = 'out/' + FLAGS.out_dir

    optimize_mesh(FLAGS, out_dir, log_interval=FLAGS.log_interval)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
