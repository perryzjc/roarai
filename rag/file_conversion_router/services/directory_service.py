"""Internal service to process a directory of files and schedule conversion tasks for each file."""
import logging
from concurrent.futures import as_completed
from pathlib import Path
from typing import Dict, Iterable, Type, Union

from rag.file_conversion_router.conversion.base_converter import BaseConverter, ConversionCache
from rag.file_conversion_router.conversion.md_converter import MarkdownConverter
from rag.file_conversion_router.conversion.pdf_converter import PdfConverter
from rag.file_conversion_router.services.task_manager import schedule_conversion

ConverterMapping = Dict[str, Type[BaseConverter]]

# Mapping from file extensions to their corresponding conversion classes
_CONVERTER_MAPPING: ConverterMapping = {
    ".pdf": PdfConverter,
    ".md": MarkdownConverter,
    # TODO: Add more file types and converters here
}


def _validate_directory(directory: Path, directory_type: str) -> None:
    if not directory.is_dir():
        raise ValueError(f"Provided {directory_type} path {directory} is not a directory.")


def _ensure_output_directory(output_dir: Path) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.is_dir():
        raise ValueError(f"Provided output path {output_dir} is not a directory.")


def _get_valid_files(input_dir: Path) -> Iterable[Path]:
    valid_extensions = tuple(_CONVERTER_MAPPING.keys())
    for input_file_path in input_dir.rglob("*"):
        if input_file_path.suffix in valid_extensions and input_file_path.is_file():
            yield input_file_path


def _get_output_file_path(input_file_path: Path, input_dir: Path, output_dir: Path) -> Path:
    output_subdir = output_dir / input_file_path.relative_to(input_dir).parent
    output_subdir.mkdir(parents=True, exist_ok=True)
    return output_subdir / input_file_path.stem


def _get_converter(file_extension: str) -> Type[BaseConverter]:
    return _CONVERTER_MAPPING.get(file_extension)


def _log_conversion_result(future):
    try:
        result = future.result()
        logging.info(f"Conversion result: {result}")
        logging.info("Task completed successfully.")
    except Exception as e:
        logging.error(f"Conversion failed: {e}")


def process_folder(input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """Walk through the input directory and schedule conversion tasks for specified file types."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    _validate_directory(input_dir, "input")
    _ensure_output_directory(output_dir)

    futures = []
    for input_file_path in _get_valid_files(input_dir):
        output_file_path = _get_output_file_path(input_file_path, input_dir, output_dir)
        converter_class = _get_converter(input_file_path.suffix)

        if converter_class:
            converter = converter_class()
            future = schedule_conversion(converter.convert, input_file_path, output_file_path)
            futures.append(future)
            logging.info(f"Scheduled conversion for {input_file_path} to {output_file_path}")
        else:
            logging.warning(f"No converter available for file type {input_file_path.suffix}")

    for future in as_completed(futures):
        _log_conversion_result(future)

    logging.info(f"Completed processing for directory: {input_dir}")
    logging.info(f"Saved conversion time [{ConversionCache.calc_total_savings()} seconds] by using cached results.")
