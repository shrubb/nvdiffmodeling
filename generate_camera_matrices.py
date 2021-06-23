import numpy as np

from src import util

import argparse
from pathlib import Path

def write_cameras_to_file(cameras, path):
    with open(path, 'w') as f:
        for camera in cameras:
            f.write("\n")
            for row in camera:
                f.write(" ".join(map(str, row)) + "\n")


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
    parser.add_argument('--elevation_levels', type=int, default=3,
        help="Over how many elevation levels to spread cameras")

    args = parser.parse_args()

    cameras = np.empty((args.n, 4, 4), dtype=np.float32)

    if args.kind == 'random':
        for i in range(args.n):
            r_rot = util.random_rotation_translation(args.radius_range)
            r_mv = util.translate(0, 0, -args.radius) @ r_rot
            cameras[i] = r_mv
    else:
        raise NotImplementedError(f"{args.kind}")

    write_cameras_to_file(cameras, args.output_file)
