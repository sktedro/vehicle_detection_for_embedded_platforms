from typing import Tuple, Union
import numpy as np

from mmcv.transforms import BaseTransform
from mmdet.structures.bbox import autocast_box_type
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CustomCutOut(BaseTransform):
    """Custom CutOut operation.

    Randomly drop some regions of annotated parts of an image. For each bbox,
    with probability `prob`, it randomly selects a region within the bbox and a
    random cutout area based on `cutout_area` as a fraction of the bbox area.
    Then, draws a square with a random color or random pixels, based on
    `random_pixels`. If a cutout area would reach beyond the bbox, it is simply
    cut to fit in.

    Required Keys:

    - img
    - gt_bboxes

    Modified Keys:

    - img

    Args:
        prob (float): Probability of an object getting cut out of in <0; 1>
        cutout_area (float or tuple[float, float]): Area of the object as
            percentage to be cut out in <0; 1>. If tuple is given, the number
            will be selected randomly and uniformly from the interval
        random_pixels (bool, default True): If True, colors of pixels in the
        cutout area will be completely random. Otherwise, all pixels will have
        the same random color

    """

    def __init__(
        self,
        prob: float,
        cutout_area: Union[float, Tuple[float, float]],
        random_pixels: bool = True
    ) -> None:

        assert 0 <= prob <= 1
        assert ((isinstance(cutout_area, float) and 0 <= cutout_area <= 1)
            or (isinstance(cutout_area, tuple) 
                and 0 <= cutout_area[0] <= 1
                and 0 <= cutout_area[1] <= 1))

        self.prob = prob
        self.cutout_area = cutout_area
        self.random_pixels = random_pixels

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Call function to drop some regions annotations."""

        h, w, _ = results["img"].shape
        bboxes = results["gt_bboxes"]

        for bbox in bboxes:

            if np.random.rand() > self.prob:
                continue

            x1, y1, x2, y2 = [int(i) for i in bbox.tensor.tolist()[0]]
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            area = (x2 - x1) * (y2 - y1) * np.random.uniform(*self.cutout_area)
            wh = min(int(np.sqrt(area)), y2 - y1 - 1, x2 - x1 - 1)

            cutout_x1 = np.random.randint(x1, x2 - wh)
            cutout_y1 = np.random.randint(y1, y2 - wh)

            # Select a random color for each pixel in the cutout area
            if self.random_pixels:
                patch = np.random.randint(0, 256, size=(wh, wh, 3), dtype=np.uint8)
                results['img'][cutout_y1:cutout_y1 + wh,
                            cutout_x1:cutout_x1 + wh,
                            :] = patch

            # Select a random color to be same for all pixels in the cutout area
            else:
                color = np.random.randint(0, 256, size=3, dtype=np.uint8)
                patch = np.tile(color, (wh, wh, 1))
                results['img'][cutout_y1:cutout_y1 + wh,
                            cutout_x1:cutout_x1 + wh,
                            :] = patch

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'cutout_area={self.cutout_area})'
        return repr_str
