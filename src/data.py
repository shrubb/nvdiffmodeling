from . import util

import torch
import numpy as np
import cv2

from pathlib import Path

RADIUS = 3.5

class MultiViewDataset(torch.utils.data.IterableDataset):
    """
    A dataset for views of a 3D object.
    One sample represents one view, and is a tuple of length 1 or 2:
    `(<camera matrix Rt>, <image [optional]>)`.
    """
    def __init__(self, path=None, resolution=512):
        """
        path:
            `None` or convertible to `pathlib.Path`
            - If `None`, sample cameras randomly, don't yield images.
            - If path to a file, yield cameras from that file, don't yield images.
            - If path to a directory, yield cameras from "`path`/cameras.txt" and yield
              images from that directory. Images have to be square (for now).

            In two last cases, the file format is:
                front.jpg
                2.5 0.0 0.0 0.0
                0.0 1.0 0.0 0.0
                0.0 0.0 1.0 -3.0
                0.0 0.0 0.0 1.0
                side_1.png
                1.0 0.0 0.0 0.0
                0.0 -1.0 0.0 0.0
                0.0 0.0 1.0 2.0
                0.0 0.0 0.0 1.0
                <etc.>
            When images aren't to be loaded, put empty lines in place of file names.

        resolution:
            `int`
            When `path` is set up to load images, sets size (square side) of these images.
            No effect otherwise.
        """
        self.resolution = int(resolution)

        if path is None:
            self.camera_matrices = None
        else:
            path = Path(path)

            if path.is_file():
                cameras_path = path
                self.images_root = None
            elif (path / "cameras.txt").is_file():
                cameras_path = path / "cameras.txt"
                self.images_root = path
            else:
                raise FileNotFoundError(f"No '{path}' and '{path / 'cameras.txt'}'")

            def camera_matrices_file_iterator(cameras_path):
                with open(cameras_path, 'r') as cameras_file:
                    while True:
                        image_file_name = cameras_file.readline()
                        if not image_file_name: # EOF
                            return
                        image_file_name = image_file_name.strip()

                        camera_matrix = np.fromstring(
                            ''.join(cameras_file.readline() for _ in range(4)),
                            sep=' ', dtype=np.float32)
                        assert camera_matrix.size == 4 * 4, f"File format error: {path}"
                        camera_matrix = camera_matrix.reshape(4, 4)

                        yield camera_matrix, image_file_name

            self.camera_matrices, self.image_paths = \
                zip(*camera_matrices_file_iterator(cameras_path))

            self.camera_matrices = np.stack(self.camera_matrices)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise NotImplementedError("Multiple workers aren't supported yet")

        if self.camera_matrices is None:
            while True:
                r_rot = util.random_rotation_translation(0.25)
                r_mv = util.translate(0, 0, -RADIUS) @ r_rot
                yield r_mv.astype(np.float32),
        else:
            while True:
                indices = torch.randperm(len(self.camera_matrices))

                for i in indices:
                    camera_matrix = self.camera_matrices[i]
                    if self.images_root is None:
                        yield camera_matrix,
                    else:
                        image = cv2.imread(str(self.images_root / self.image_paths[i]))
                        if image.shape[0] != image.shape[1]:
                            raise NotImplementedError(
                                f"Non-square images not supported yet: {self.image_paths[i]}")

                        if image.shape[0] != self.resolution:
                            image = cv2.resize(
                                image,
                                (self.resolution, self.resolution),
                                interpolation=cv2.INTER_CUBIC)

                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
                        image = image.astype(np.float32)
                        image /= 255.0

                        yield camera_matrix, image


def get_dataloader(batch_size, path, resolution):
    return torch.utils.data.DataLoader(
        MultiViewDataset(path, resolution), batch_size=batch_size, num_workers=1)
