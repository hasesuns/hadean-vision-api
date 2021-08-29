import os
from dataclasses import dataclass, field
from logging import getLogger
from typing import List, Tuple

from dotenv import load_dotenv

from hadeanvision.convert_picture import ConvertParams, convert

logger = getLogger(__name__)
import cv2
import numpy as np
import pytest


@pytest.mark.github_actions
@dataclass(frozen=True)
class ConvertParamsTestCase:
    desc: str
    num_colors: int
    rgb_list: List[Tuple[float, float, float]] = field(default_factory=list)
    bgr_list: List[Tuple[float, float, float]] = field(default_factory=list)
    corrected_params: ConvertParams = ConvertParams()


param_queries = (
    ConvertParamsTestCase(
        desc="num_colorsとlen(rgb_list)が一致している想定通りのケース",
        num_colors=3,
        rgb_list=[(255, 0, 0), (100, 255, 200), (1, 2, 3)],
        corrected_params=ConvertParams(num_colors=3, bgr_list=[(0, 0, 255), (200, 255, 100), (3, 2, 1)]),
    ),
    ConvertParamsTestCase(
        desc="num_colors > len(rgb_list) となるケース",
        num_colors=4,
        rgb_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        corrected_params=ConvertParams(num_colors=4, bgr_list=[(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]),
    ),
    ConvertParamsTestCase(
        desc="num_colors < len(rgb_list) となるケース",
        num_colors=2,
        rgb_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        corrected_params=ConvertParams(num_colors=2, bgr_list=[(0, 0, 255), (0, 255, 0)]),
    ),
    ConvertParamsTestCase(
        desc="rgbの値が0~255の範囲から外れているケース",
        num_colors=3,
        rgb_list=[(500, -1, -10), (-1, 500, -10), (-1, -10, 256)],
        corrected_params=ConvertParams(num_colors=3, bgr_list=[(0, 0, 255), (0, 255, 0), (255, 0, 0)]),
    ),
    ConvertParamsTestCase(
        desc="num_colorsの値が0になっていたケース",
        num_colors=0,
        rgb_list=[],
        corrected_params=ConvertParams(
            num_colors=5, bgr_list=[(255, 255, 0), (80, 50, 10), (10, 80, 170), (5, 0, 50), (40, 10, 0)]
        ),
    ),
    ConvertParamsTestCase(
        desc="ConvertParamsでパラメータを指定しなかったケース",
        num_colors=ConvertParams().num_colors,
        rgb_list=ConvertParams().rgb_list,
        corrected_params=ConvertParams(
            num_colors=5, bgr_list=[(255, 255, 0), (80, 50, 10), (10, 80, 170), (5, 0, 50), (40, 10, 0)]
        ),
    ),
)

id_param_queries = [query.desc for query in param_queries]


@pytest.mark.github_actions
@pytest.mark.parametrize("query", param_queries, ids=id_param_queries)
def test_init_convert_params(query, caplog):
    """convert_params should be generated."""
    convert_params = ConvertParams(num_colors=query.num_colors, rgb_list=query.rgb_list)
    assert convert_params == query.corrected_params


@pytest.mark.github_actions
@pytest.mark.parametrize("query", param_queries, ids=id_param_queries)
def test_convert(query, caplog):
    """convert() output should be hadean colored image."""

    input_img = cv2.imread("data/tests/input/beach.jpg")
    output_img = convert(input_img=input_img, convert_params=query.corrected_params)

    load_dotenv()
    if os.environ.get("ENVIRONMENT") == "local":
        output_img_path = "data/tests/output/beach_hadean.png"
        cv2.imwrite(output_img_path, output_img)
        output_img = cv2.imread(output_img_path)

    output_color_list = np.unique(output_img.reshape((-1, 3)), axis=0)
    assert len(output_color_list) == query.corrected_params.num_colors  # 元の写真を構成する色数がquery.num_colorsより少ないと成立しないので注意

    output_color_set = set([tuple(bgr) for bgr in output_color_list])
    assert output_color_set == set(query.corrected_params.bgr_list)
