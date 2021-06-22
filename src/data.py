from . import util

import torch

from pathlib import Path

RADIUS = 3.5

class MultiViewDataset(torch.utils.data.IterableDataset):
    """
    A dataset for views of a 3D object.
    One sample represents one view, and is a tuple
    `(<camera matrix Rt>, <image [or None, i.e. optional]>)`.
    """
    def __init__(self, path=None):
        """
        path:
            `None` or convertible to `pathlib.Path`

            - If `None`, sample cameras randomly, don't yield images.

            - If path to a file, yield cameras from that file, don't yield images. File format:
                [[2.5, 0.0, 0.0, 0.0], [...], [...], [...]]
                [[1.0, 0.0, 0.0, 0.0], [...], [...], [...]]
                <similar lines with 4x4 matrices, one for each camera...>

            - If path to a directory, yield cameras from "`path`/cameras.txt" and yield
              images from that directory. File format:
                front.jpg
                [[2.5, 0.0, 0.0, 0.0], [...], [...], [...]]
                side.png
                [[1.0, 0.0, 0.0, 0.0], [...], [...], [...]]
                <similar lines, two for each camera...>
        """
        if path is None:
            self.data = None
        else:
            path = Path(path)

            raise NotImplementedError()

    def __iter__(self):
        if self.data is None:
            while True:
                r_rot = util.random_rotation_translation(0.25)
                r_mv = util.translate(0, 0, -RADIUS) @ r_rot
                yield r_mv, None
        else:
            raise NotImplementedError()
