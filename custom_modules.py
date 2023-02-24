from typing import Tuple, Union
import numpy as np

from mmcv.transforms import BaseTransform
from mmdet.structures.bbox import autocast_box_type
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CustomCutOut(BaseTransform):
    """Custom CutOut operation.

    Randomly drop some regions of annotated parts of an image. For each bbox,
    with probability `prob`, it randomly selects a cutout center within the bbox
    and a random cutout area based on `cutout_area` as a fraction of the bbox
    area. Then, draws a square with `fill_in` color. The center can be chosen in
    a way that a part of the square sits beyond the bbox if `max_bbox_overflow`
    is bigger than 0. Anyways, the cutout can overflow the bbox if the bbox
    is a long (or high) rectangle, since this is only drawing squares.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): Probability of an object getting cut out of in <0; 1>
        cutout_area (float or tuple[float, float]): Area of the object as
            percentage to be cut out in <0; 1>. If tuple is given, the number
            will be selected randomly and uniformly from the interval
        max_bbox_overflow (float): Maximum percentage of the cutout area that
            is allowed to overflow beyond the bbox. Default 0
        fill_in (tuple[float, float, float] or tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Defaults to random
    """

    def __init__(
        self,
        prob: float,
        cutout_area: Union[float, Tuple[float, float]],
        max_bbox_overflow: float = 0,
        fill_in: Union[Tuple[float, float, float], Tuple[int, int,
                                                         int]] = (-1, -1, -1)
    ) -> None:

        assert 0 <= prob <= 1
        assert ((isinstance(cutout_area, float) and 0 <= cutout_area <= 1)
            or (isinstance(cutout_area, tuple) 
                and 0 <= cutout_area[0] <= 1
                and 0 <= cutout_area[1] <= 1))
        assert 0 <= max_bbox_overflow <= 1

        self.prob = prob
        self.cutout_area = cutout_area
        self.max_bbox_overflow = max_bbox_overflow
        self.fill_in = fill_in

        # Optimization:
        self.min_in_bbox = 1 - self.max_bbox_overflow
        self.min_in_bbox_inv = 1 / self.min_in_bbox


    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Call function to drop some regions annotations."""
        fill_in = self.fill_in

        bboxes = results["gt_bboxes"]

        for bbox in bboxes:

            if np.random.rand() > self.prob:
                continue

            x1, y1, x2, y2 = bbox.tensor.tolist()[0]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            real_area = (x2 - x1) * (y2 - y1) * np.random.uniform(self.cutout_area[0], self.cutout_area[1])

            # Calculate cutout w and h with smaller area (considering
            # max_bbox_overflow) to then overflow the bbox
            # in_bbox_area = real_area * self.min_in_bbox
            # in_bbox_wh = np.sqrt(in_bbox_area)
            # Optimized:
            in_bbox_wh_div2 = np.sqrt(real_area * self.min_in_bbox) / 2

            # Calculate the cutout center
            # cutout_center = [
            #     np.random.uniform(x1 + in_bbox_wh / 2, x2 - in_bbox_wh / 2),
            #     np.random.uniform(y1 + in_bbox_wh / 2, y2 - in_bbox_wh / 2)]

            # Optimized:
            cutout_center_x = np.random.uniform(x1 + in_bbox_wh_div2, x2 - in_bbox_wh_div2)
            cutout_center_y = np.random.uniform(y1 + in_bbox_wh_div2, y2 - in_bbox_wh_div2)

            # Real cutout coordinates
            # real_wh = np.sqrt(real_area)
            # cutout_coords_real = {
            #     "x1": np.round(cutout_center[0] - (real_wh / 2)),
            #     "y1": np.round(cutout_center[1] - (real_wh / 2)),
            #     "x2": np.round(cutout_center[0] + (real_wh / 2)),
            #     "y2": np.round(cutout_center[1] + (real_wh / 2)),
            # }

            # results['img'][cutout_coords_real["y1"]:cutout_coords_real["y2"],
            #                cutout_coords_real["x1"]:cutout_coords_real["x2"], 
            #                :] = self.fill_in

            if self.fill_in == (-1, -1, -1):
                fill_in = tuple(np.random.randint(0, 256) for _ in range(3))

            # Optimized
            real_wh_div2 = int(np.sqrt(real_area) / 2)
            cutout_center_x = int(cutout_center_x)
            cutout_center_y = int(cutout_center_y)
            results['img'][cutout_center_y - real_wh_div2
                            :cutout_center_y + real_wh_div2,
                           cutout_center_x - real_wh_div2
                            :cutout_center_x + real_wh_div2,
                           :] = fill_in

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'cutout_area={self.cutout_area}, '
        repr_str += f'fill_in={self.fill_in})'
        return repr_str
