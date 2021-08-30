import os
from dataclasses import dataclass, field
from logging import getLogger
from typing import List, Tuple

import cv2
import numpy as np
import pytest
from dotenv import load_dotenv

from hadeanvision.convert_picture import ConvertParams, convert

logger = getLogger(__name__)


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
        desc="Case: num_colors == len(rgb_list)",
        num_colors=3,
        rgb_list=[(255, 0, 0), (100, 255, 200), (1, 2, 3)],
        corrected_params=ConvertParams(num_colors=3, bgr_list=[(0, 0, 255), (200, 255, 100), (3, 2, 1)]),
    ),
    ConvertParamsTestCase(
        desc="Case: num_colors > len(rgb_list)",
        num_colors=4,
        rgb_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        corrected_params=ConvertParams(num_colors=4, bgr_list=[(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]),
    ),
    ConvertParamsTestCase(
        desc="Case: num_colors < len(rgb_list)",
        num_colors=2,
        rgb_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        corrected_params=ConvertParams(num_colors=2, bgr_list=[(0, 0, 255), (0, 255, 0)]),
    ),
    ConvertParamsTestCase(
        desc="Case: RGB values is out of 0~255",
        num_colors=3,
        rgb_list=[(500, -1, -10), (-1, 500, -10), (-1, -10, 256)],
        corrected_params=ConvertParams(num_colors=3, bgr_list=[(0, 0, 255), (0, 255, 0), (255, 0, 0)]),
    ),
    ConvertParamsTestCase(
        desc="Case: num_colors == 0",
        num_colors=0,
        rgb_list=[],
        corrected_params=ConvertParams(
            num_colors=5, bgr_list=[(255, 255, 0), (80, 50, 10), (10, 80, 170), (5, 0, 50), (40, 10, 0)]
        ),
    ),
    ConvertParamsTestCase(
        desc="Case: ConvertParams has no input values",
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
    # Note: This will not work if the number of colors in the original photo is less than query.num_colors.
    assert len(output_color_list) == query.corrected_params.num_colors  

    output_color_set = set([tuple(bgr) for bgr in output_color_list])
    assert output_color_set == set(query.corrected_params.bgr_list)
