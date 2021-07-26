from . import util

import torch
import numpy as np
import cv2

import json
import math
import enum
from pathlib import Path
import random

RADIUS = 3.5

class DatasetKind(enum.Enum):
    RANDOM_CAMERAS = 1
    FIXED_CAMERAS = 2
    FIXED_CAMERAS_AND_IMAGES = 3

class MultiViewDataset(torch.utils.data.IterableDataset):
    """
    A dataset for views of a 3D object.
    One sample represents one view, and is a tuple of length 1 or 3:
    `(<camera matrix Rt>,)`
    or
    `(<camera matrix Rt>, <image>, <foreground mask>)`.
    """
    def __init__(self, path=None, resolution=None):
        """
        path:
            `None` or convertible to `pathlib.Path`
            - If `None`, sample cameras randomly, don't yield images.
            - If path to a file, yield cameras from that file, don't yield images.
            - If path to a directory, yield cameras from "`path`/cameras.json" and yield
              images from that directory.

            In two last cases, the file format is as https://github.com/shrubb/nerf-pytorch
            (see load_blender.py). Example:
            {
                "image_height": 378,
                "image_width": 504,
                "camera_angle_x": 1.088803798681739,
                "camera_angle_y": 0.8524697440569041,
                "frames": [
                    {
                        "file_path": "./000.png",
                        "rotation": 0.012566370614359171,
                        "transform_matrix": [
                            [
                                0.9987568259239197,
                                0.011033710092306137,
                                0.04861102253198624,
                                0.18809327483177185
                            ],
                            ...

        resolution:
            `int`
            Not yet supported.
            (When `path` is set up to load images, ...)
        """
        if path is None:
            self.kind = DatasetKind.RANDOM_CAMERAS
            self.camera_matrices = None
            self.projection_matrix = util.projection(r=0.4, f=1000.0)

            assert resolution is not None, "Training resolution not specified in config"
            self.resolution = resolution
        else:
            path = Path(path)

            if path.is_file():
                self.kind = DatasetKind.FIXED_CAMERAS
                cameras_path = path
                self.images_root = None
            elif (path / "cameras.json").is_file():
                self.kind = DatasetKind.FIXED_CAMERAS_AND_IMAGES
                cameras_path = path / "cameras.json"
                self.images_root = path
            else:
                raise FileNotFoundError(f"No '{path}' and '{path / 'cameras.json'}'")

            def load_blender_poses(path):
                with open(path, 'r') as f:
                    metadata = json.load(f)

                W = metadata.get('image_width', 800)
                H = metadata.get('image_height', W)
                camera_angle_x = metadata['camera_angle_x']
                camera_angle_y = metadata.get('camera_angle_y', camera_angle_x)

                camera_poses = [
                    np.linalg.inv(frame['transform_matrix']) for frame in metadata['frames']]
                camera_poses = np.float32(camera_poses)

                image_names = [frame['file_path'] for frame in metadata['frames']]

                return image_names, camera_poses, camera_angle_x, camera_angle_y, H, W

            self.image_paths, self.camera_matrices, fov_x, fov_y, H, W = \
                load_blender_poses(cameras_path)

            if self.kind == DatasetKind.FIXED_CAMERAS_AND_IMAGES and resolution is not None:
                raise NotImplementedError("Custom training images resolution specified in config")
            self.resolution = (H, W)

            self.projection_matrix = util.projection(
                r=math.tan(fov_x / 2), t=-math.tan(fov_y / 2), f=1000.0)

    def get_optimal_num_workers(self):
        if self.camera_matrices is None:
            return 2
        else:
            return min(4, len(self.camera_matrices))

    def get_projection_matrix(self):
        return self.projection_matrix

    def get_resolution(self):
        return self.resolution

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        if self.camera_matrices is None:
            # Yield random cameras for reference mesh
            while True:
                r_rot = util.random_rotation_translation(0.25)
                r_mv = util.translate(0, 0, -RADIUS) @ r_rot
                yield r_mv.astype(np.float32),
        else:
            # Yield fixed cameras...
            _piece_size = (len(self.camera_matrices) + num_workers - 1) // num_workers
            start_idx = worker_id * _piece_size
            end_idx = min((worker_id + 1) * _piece_size, len(self.camera_matrices))

            while True:
                indices = list(range(start_idx, end_idx))
                random.shuffle(indices)

                for i in indices:
                    camera_matrix = self.camera_matrices[i]
                    if self.images_root is None:
                        # ...either for reference mesh...
                        yield camera_matrix,
                    else:
                        # ...or along with images.
                        image = cv2.imread(
                            str(self.images_root / self.image_paths[i]), cv2.IMREAD_UNCHANGED)

                        if image.shape[:2] != self.resolution:
                            image = cv2.resize(
                                image, self.resolution, interpolation=cv2.INTER_CUBIC)

                        if image.shape[2] == 4:
                            image, foreground_mask = image[..., :3], image[..., 3:]
                        else:
                            foreground_mask = np.zeros_like(image[..., :1])

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = image.astype(np.float32)
                        image /= 255.0
                        foreground_mask = foreground_mask.astype(np.float32)
                        foreground_mask /= 255.0

                        yield camera_matrix, image, foreground_mask


def get_dataloader(batch_size, path, resolution):
    def worker_init_fn(worker_id):
        import random
        import numpy as np
        import torch

        random.seed(worker_id)
        np.random.seed(worker_id)
        torch.manual_seed(worker_id)

    dataset = MultiViewDataset(path, resolution)

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=dataset.get_optimal_num_workers(), worker_init_fn=worker_init_fn)
