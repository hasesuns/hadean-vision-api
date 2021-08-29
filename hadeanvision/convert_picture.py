from dataclasses import dataclass, field
from logging import getLogger
from typing import List, Tuple

import cv2
import numpy as np

logger = getLogger(__name__)


def _clamp(val: float) -> float:
    """val 0~255の範囲に収まる数値に丸める"""
    return min(255, max(val, 0))


@dataclass(frozen=True)
class ConvertParams:
    num_colors: int = 0
    rgb_list: List[Tuple[float, float, float]] = field(default_factory=list)
    bgr_list: List[Tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self):

        DEFAULT_COLORS = [(0, 255, 255), (10, 50, 80), (170, 80, 10), (50, 0, 5), (0, 10, 40)]
        num_colors = self.num_colors
        color_list: List[Tuple(float, float, float)] = []
        use_bgr_input: bool = False

        if len(self.bgr_list) > 0:
            color_list = self.bgr_list
            use_bgr_input = True  # bgr_listの入力を用いるかrgb_listの入力を用いるかを保持するflag
        elif len(self.rgb_list) > 0:
            color_list = self.rgb_list
            use_bgr_input = False
        else:
            if num_colors < 1:  # num_colorsが非正数をとるのはおかしいのでdefault値として5を代入する
                num_colors = 5
            color_list = DEFAULT_COLORS
            use_bgr_input = False

        if self.num_colors != num_colors:
            object.__setattr__(self, "num_colors", num_colors)

        if self.num_colors < len(color_list):
            logger.warning(
                f"num_colorsよりもcolor listの要素数が多いため、後方のcolor listの値は無視されます。\
                    （num_colors: {self.num_colors}, len(color list): {len(color_list)}）"
            )
            color_list = color_list[: self.num_colors]

        if self.num_colors > len(color_list):
            logger.warning(
                f"num_colorsよりもcolor listの要素数が少ないため、color listにcyan（R:0, G:255, B:255）を不足分だけ追加します。\
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


def convert(input_img: np.ndarray, convert_params: ConvertParams = ConvertParams()) -> np.ndarray:
    """input_imgをconvert_paramsに基づいてkmeansなどによりスタイル変換する

    Args:
        input_img (np.ndarray): [description]
        convert_params (ConvertParams): [description]

    Returns:
        np.ndarray: [description]
    """

    samples = np.float32(input_img.reshape((-1, 3)))
    n_cluster = convert_params.num_colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(samples, n_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # TODO: 元画像の輝度を出力に反映させる

    color_look_up_table = np.array(convert_params.bgr_list, np.uint8)
    product = color_look_up_table[label.flatten()]
    output_img = product.reshape((input_img.shape))

    return output_img
