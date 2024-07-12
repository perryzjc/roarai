from pathlib import Path
from typing import List

from unittest.mock import patch
import pytest
import yaml

from rag.file_conversion_router.conversion.md_converter import MarkdownConverter
from rag.file_conversion_router.conversion.pdf_converter import PdfConverter
from tests.utils import compare_files
from rag.file_conversion_router.config import CONFIG

# Define common base paths
BASE_PATH = Path(__file__).parent
DATA_FOLDER = BASE_PATH / "data"
CONFIG_FILE_PATH = DATA_FOLDER / "test_cases_config.yaml"
# Global variable to cache the test cases configuration
TEST_CASES_CONFIG = None


def load_yaml_config(config_path):
    """Load YAML configuration from a file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def resolve_paths(config, base_path):
    """Recursively resolve paths in the configuration dictionary.

    Args:
        config (dict): The configuration dictionary.
        base_path (Path): The base path to resolve relative paths against.

    Modifies:
        config: Paths ending with '_path' or '_folder' are resolved to absolute paths.

    >>> test_config = {'input_path': 'input/example_input.pdf'}
    >>> resolve_paths(test_config, Path('/absolute/path/to/data'))
    >>> print(test_config)
    {'input_path': '/absolute/path/to/data/input/example_input.pdf'}
    """
    if isinstance(config, dict):
        for key, value in config.items():
            if key.endswith("_path") or key.endswith("_folder"):
                config[key] = str((base_path / value).resolve())
            elif key.endswith("_paths"):
                config[key] = [str((base_path / path).resolve()) for path in value]
            else:
                resolve_paths(value, base_path)
    elif isinstance(config, list):
        for item in config:
            resolve_paths(item, base_path)


def load_test_cases_config(*keys):
    """Load and return configurations for specified test cases from a YAML file.

    This function loads and processes the configuration once and uses caching to store
    resolved paths. Subsequent calls use the cached data.

    Args:
        *keys (str): Variable length argument list of keys to access nested dictionary values.

    Returns:
        list: A list of tuples containing the values from the specified configuration entries.

    Raises:
        ValueError: If no keys are provided.
    """
    global TEST_CASES_CONFIG
    if not keys:
        raise ValueError("At least one key must be provided")

    if TEST_CASES_CONFIG is None:
        raw_config = load_yaml_config(CONFIG_FILE_PATH)
        resolve_paths(raw_config, DATA_FOLDER)
        TEST_CASES_CONFIG = raw_config

    config = TEST_CASES_CONFIG
    for key in keys:
        config = config[key]

    return [tuple(test_case.values()) for test_case in config]


@pytest.fixture(scope="function")
def pdf_converter():
    return PdfConverter()


@pytest.fixture(scope="function")
def md_converter():
    return MarkdownConverter()


@pytest.fixture
def mock_config():
    """Fixture to mock the configuration values for the tests.
    Below values might be different from what is defined by default in `config.ini`.
    """
    config_values = {
        "pdf_converter.nougat": {
            "model_tag": "0.1.0-small",
            "batch_size": 8,
            "no_skipping": False,
            "recompute": False,
            "use_nougat_server": False,
            "server_port": 8503
        }
    }

    def side_effect(section, option, method="get"):
        # Helper function to return values based on the method
        value = config_values.get(section, {}).get(option, '')
        if method == 'getint' or method == 'getfloat':
            return int(value) if value != '' else 0
        elif method == 'getboolean':
            return str(value).lower() in ['true', '1', 'yes']
        return value

    with patch('rag.file_conversion_router.config.CONFIG.get') as mock_get, \
         patch('rag.file_conversion_router.config.CONFIG.getint') as mock_getint, \
         patch('rag.file_conversion_router.config.CONFIG.getboolean') as mock_getboolean:
        mock_get.side_effect = lambda s, o: side_effect(s, o, 'get')
        mock_getint.side_effect = lambda s, o: side_effect(s, o, 'getint')
        mock_getboolean.side_effect = lambda s, o: side_effect(s, o, 'getboolean')
        yield


def helper_unit_test_on_converter(input_path: str, expected_output_paths: List[str], tmp_path, converter):
    input_path = Path(input_path)
    expected_paths = [Path(expected_output_path) for expected_output_path in expected_output_paths]
    output_folder = tmp_path / input_path.stem
    converter.convert(input_path, output_folder)

    for idx, suffix in enumerate([".md", ".md.pkl", ".md.tree.txt"]):
        output_file_path = output_folder / f"{input_path.stem}{suffix}"
        assert output_file_path.exists(), f"File {output_file_path} does not exist."
        assert compare_files(
            expected_paths[idx], output_file_path
        ), f"File conversion for {input_path} did not meet expectations."


def test_load_test_cases_config():
    result = load_test_cases_config("unit_tests", "example_format")
    assert type(result) == list
    assert all(isinstance(item, tuple) for item in result)
    assert result[0][0].endswith("tests/test_rag/data/unit_tests/example_format/input/example_input.md")
    assert result[0][1][0].endswith("tests/test_rag/data/unit_tests/example_format/expected_output/example_input.md")
    assert result[0][1][1].endswith("tests/test_rag/data/unit_tests/example_format/expected_output/example_input.md.pkl")
    assert result[0][1][2].endswith("tests/test_rag/data/unit_tests/example_format/expected_output/example_input.tree.txt")


def test_mock_config(mock_config):
    assert CONFIG.get("pdf_converter.nougat", "model_tag") == "0.1.0-small"
    assert CONFIG.getint("pdf_converter.nougat", "batch_size") == 8
    assert CONFIG.getboolean("pdf_converter.nougat", "no_skipping") is False
    assert CONFIG.getboolean("pdf_converter.nougat", "recompute") is False
    assert CONFIG.getboolean("pdf_converter.nougat", "use_nougat_server") is False
    assert CONFIG.getint("pdf_converter.nougat", "server_port") == 8503
