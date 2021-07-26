import numpy as np
np.random.seed(123)

from src import util

import argparse
import json
from pathlib import Path

def write_cameras_to_txt(cameras, path, names=None):
    """
    Write camera transformations to disk in custom format for 'nvdiffmodeling'.
    TODO: migrate nvdiffmodeling to json.
    """
    if names is None:
        names = [f"./{idx:03}.png" for idx in range(len(cameras))]

    with open(path, 'w') as f:
        for camera, name in zip(cameras, names):
            f.write(name + "\n")
            for row in camera:
                f.write(" ".join(map(str, row)) + "\n")

def write_cameras_to_json(cameras, path, names=None):
    """
    Write camera transformations to disk in 'nerf-pytorch''s "blender" json format.
    """
    output_json = {
        "image_width": 800,
        "camera_angle_x": 0.6911112070083618, # field of view, a hardcoded default
        "frames": [
            {
                "file_path": names[idx] if names is not None else f"./{idx:03}.png",
                "rotation": 0.012566370614359171, # don't know what this is, took it from lego
                "transform_matrix": np.linalg.inv(camera).tolist()
            } for idx, camera in enumerate(cameras)
        ]
    }

    with open(path, 'w') as f:
        json.dump(output_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', type=Path,
        help="Ouptut file")
    parser.add_argument('kind', type=str, choices=['random', 'uniform'], default='random',
        help="Generation algorithm")
    parser.add_argument('n', type=int, default=30,
        help="How many cameras to generate")
    parser.add_argument('--radius', type=float, default=3.5,
        help="How far (on average, if kind='random') should all cameras be from the origin")
    parser.add_argument('--radius_range', type=float, default=0.25,
        help="Max radius difference from average")
    parser.add_argument('--elevation_upper_limit', type=float, default=0.0,
        help="Highest elevation angle, degrees from 0 (straight above) to 180 (straight below) "
        "(default: not limited)")
    parser.add_argument('--elevation_lower_limit', type=float, default=180.0,
        help="Lowest elevation angle, degrees from 0 (straight above) to 180 (straight below) "
        "(default: not limited)")
    parser.add_argument('--up_axis', type=str, choices=['x', 'y', 'z', 'x-', 'y-', 'z-'], default='y',
        help="Which axis to treat as 'up'. Usually it will be 'y' for standard nvdiffmodeling "
        "datasets, 'z' for NeRF datasets.")

    args = parser.parse_args()

    up_axis = ['x', 'y', 'z'].index(args.up_axis.strip('-'))
    up_axis_inverted = -1.0 if '-' in args.up_axis else 1.0

    cameras = np.empty((args.n, 4, 4), dtype=np.float32)

    if args.kind == 'random':
        for i in range(args.n):
            # Sample a point until the elevation is suitable
            # (yes, this is inefficient)
            while True:
                eye = np.random.normal(0, 1, 3)
                radius = np.linalg.norm(eye)
                elevation = np.arccos(up_axis_inverted * eye[up_axis] / radius)
                if args.elevation_upper_limit <= np.rad2deg(elevation) <= args.elevation_lower_limit:
                    break

            # Bring the point to the desired radius
            eye *= np.random.uniform(
                args.radius - args.radius_range, args.radius + args.radius_range) / np.linalg.norm(eye)

            # Construct the camera matrix
            up = np.array([0, 0, 0]); up[up_axis] = up_axis_inverted
            at = np.array([0, 0, 0])
            camera_matrix = util.lookAt(eye, at, up)

            cameras[i] = camera_matrix
    else:
        raise NotImplementedError(f"{args.kind}")

    if args.output_file.suffix == '.txt':
        write_cameras_to_txt(cameras, args.output_file)
    elif args.output_file.suffix == '.json':
        write_cameras_to_json(cameras, args.output_file)
    else:
        raise NotImplementedError(f"Saving to {args.output_file.suffix} not implemented")
