from dataclasses import dataclass, field
from logging import getLogger
from typing import List, Tuple

import cv2
import numpy as np

logger = getLogger(__name__)


def _clamp(val: float) -> float:
    """clamp value into 0~255"""
    return min(255, max(val, 0))


@dataclass(frozen=True)
class ConvertParams:
    num_colors: int = 0
    rgb_list: List[Tuple[float, float, float]] = field(default_factory=list)
    bgr_list: List[Tuple[float, float, float]] = field(default_factory=list)
    is_random: bool = True

    def __post_init__(self):

        DEFAULT_COLORS = [(0, 255, 255), (10, 50, 80), (170, 80, 10), (50, 0, 5), (0, 10, 40)]
        num_colors = self.num_colors
        color_list: List[Tuple(float, float, float)] = []
        use_bgr_input: bool = False  # use either bgr_list or rgb_list

        if len(self.bgr_list) > 0:
            color_list = self.bgr_list
            use_bgr_input = True
        elif len(self.rgb_list) > 0:
            color_list = self.rgb_list
            use_bgr_input = False
        else:
            if num_colors < 1:  # num_colors shoud be positive number
                num_colors = 5
            color_list = DEFAULT_COLORS
            use_bgr_input = False

        if self.num_colors != num_colors:
            object.__setattr__(self, "num_colors", num_colors)

        if self.num_colors < len(color_list):
            logger.warning(
                f"Since num_colors is smaller than len(color_list), The remaining elements of color_list will be ignored.\
                    （num_colors: {self.num_colors}, len(color list): {len(color_list)}）"
            )
            color_list = color_list[: self.num_colors]

        if self.num_colors > len(color_list):
            logger.warning(
                f"Since num_colors is bigger than len(color_list).CYAN（R:0, G:255, B:255）will be added to color_list for the missing amount.\
                    （num_colors: {self.num_colors}, len(color list): {len(color_list)}）"
            )
            if use_bgr_input:
                CYAN = (255, 255, 0)
            else:
                CYAN = (0, 255, 255)
            color_list += [CYAN for _ in range(self.num_colors - len(color_list))]

        rgb_list: List[Tuple(float, float, float)] = []
        bgr_list: List[Tuple(float, float, float)] = []
        for i, color in enumerate(color_list):
            if use_bgr_input:
                b, g, r = color
            else:
                r, g, b = color
            rgb_list.append((_clamp(int(r)), _clamp(int(g)), _clamp(int(b))))
            bgr_list.append((_clamp(int(b)), _clamp(int(g)), _clamp(int(r))))

        object.__setattr__(self, "rgb_list", rgb_list)
        object.__setattr__(self, "bgr_list", bgr_list)


def clustering(input_img: np.ndarray, n_cluster: int, is_random: bool = True) -> np.ndarray:
    """clustering color image pixels using k-means.

    Args:
        input_img (np.ndarray): input color image
        n_cluster (int): number of cluster
        is_random (bool): use random

    Raises:
        ValueError: [description]

    Returns:
        np.ndarray: cluster label
    """
    if n_cluster < 1:
        logger.error("n_cluster should be positive number")
        raise ValueError

    samples = np.float32(input_img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    if is_random:
        kmeans_flag = cv2.KMEANS_RANDOM_CENTERS
    else:
        kmeans_flag = cv2.KMEANS_PP_CENTERS
    ret, label, center = cv2.kmeans(samples, n_cluster, None, criteria, 10, kmeans_flag)

    return label


def coloring(label: np.ndarray, img_shape: Tuple[int, ...], convert_params: ConvertParams = ConvertParams()) -> np.ndarray:
    """Coloring k-means output

    Args:
        label (np.ndarray): cluster label
        img_shape (Tuple[int, int, int]): image shape
        convert_params (ConvertParams, optional): Convert parameters. Defaults to ConvertParams().

    Returns:
        np.ndarray: [description]
    """
    color_look_up_table = np.array(convert_params.bgr_list, np.uint8)
    product = color_look_up_table[label.flatten()]
    colored_img = product.reshape((img_shape))
    return colored_img


def convert(input_img: np.ndarray, convert_params: ConvertParams = ConvertParams()) -> np.ndarray:
    """Converts the image based on the parameters. Internally, the k-means method and other methods are used.

    Args:
        input_img (np.ndarray): input color image.
        convert_params (ConvertParams): Convert parameters. Defaults to ConvertParams().

    Returns:
        np.ndarray: converted color image.
    """
    label = clustering(input_img, convert_params.num_colors, convert_params.is_random)
    output_img = coloring(label, input_img.shape, convert_params)
    # TODO: 元画像の輝度を出力に反映させる

    return output_img
