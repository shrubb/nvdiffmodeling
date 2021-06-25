from generate_camera_matrices import write_cameras_to_file

import numpy as np

import json
import sys
from pathlib import Path

source_file, destination_file = sys.argv[1:]
destination_file = Path(destination_file)

if destination_file.exists():
	raise FileExistsError(destination_file)

with open(source_file, 'r') as f:
	cameras_json = json.load(f)

cameras = []
names = []

for frame in cameras_json['frames']:
	names.append(frame['file_path'] + ".png")
	cameras.append(np.linalg.inv(frame['transform_matrix']))

write_cameras_to_file(cameras, destination_file, names)
