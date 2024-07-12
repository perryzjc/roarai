from typing import List

import pytest

from rag.file_conversion_router.conversion.pdf_converter import PdfConverter
from tests.test_rag.conftest import helper_unit_test_on_converter, load_test_cases_config


@pytest.mark.parametrize(
    "input_path, expected_output_paths",
    load_test_cases_config("unit_tests", "pdf_converter", "example_1"),
)
def test_pdf_conversion(input_path: str, expected_output_paths: List[str], tmp_path, pdf_converter, mock_config):
    helper_unit_test_on_converter(input_path, expected_output_paths, tmp_path, PdfConverter())
