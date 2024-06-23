"""Internal service to process a directory of files and schedule conversion tasks for each file."""
import logging
from concurrent.futures import as_completed, Future
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Type, Union
from itertools import islice

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
_BATCH_SIZE = 128


def process_folder(input_dir: Union[str, Path], output_dir: Union[str, Path], batch_size: int = _BATCH_SIZE) -> None:
    """Walk through the input directory and schedule conversion tasks for specified file types."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    _validate_directory(input_dir, "input")
    _ensure_output_directory(output_dir)

    for batch_number, batch in enumerate(_batch_process_files(input_dir, batch_size), start=1):
        futures = list(filter(None, (_schedule_conversion_task(file_path, input_dir, output_dir) for file_path in batch)))
        for future in as_completed(futures):
            _handle_conversion_result(future)
        logging.debug(f"Completed batch {batch_number} ({len(batch)} files)")

    logging.info(f"Completed processing for directory: {input_dir}")
    logging.info(f"Saved conversion time [{ConversionCache.calc_total_savings()} seconds] by using cached results.")


def _batch_process_files(input_dir: Path, batch_size: int) -> Iterable[List[Path]]:
    """Yield batches of valid files for processing."""
    valid_files = _get_valid_files(input_dir)
    while True:
        batch = list(islice(valid_files, batch_size))
        if not batch:
            break
        yield batch


def _schedule_conversion_task(input_file_path: Path, input_dir: Path, output_dir: Path) -> Optional[Future]:
    """Schedule a conversion task for a single file."""
    output_file_path = _get_output_file_path(input_file_path, input_dir, output_dir)
    converter_class = _CONVERTER_MAPPING.get(input_file_path.suffix)
    if converter_class:
        converter = converter_class()
        conversion_future = schedule_conversion(converter.convert, input_file_path, output_file_path)
        logging.info(f"Scheduled conversion for {input_file_path} to {output_file_path}")
        return conversion_future
    logging.warning(f"No converter available for file type {input_file_path.suffix}")
    return None


def _validate_directory(directory: Path, directory_type: Literal["input", "output"]) -> None:
    """Validate that the given path is a directory."""
    if not directory.is_dir():
        raise ValueError(f"Provided {directory_type} path {directory} is not a directory.")


def _ensure_output_directory(output_dir: Path) -> None:
    """Ensure that the output directory exists, creating it if necessary."""
    output_dir.mkdir(parents=True, exist_ok=True)


def _get_valid_files(input_dir: Path) -> Iterable[Path]:
    """Yield valid files for processing."""
    valid_extensions = tuple(_CONVERTER_MAPPING.keys())
    return (
        file_path
        for file_path in input_dir.rglob("*")
        if file_path.suffix in valid_extensions and file_path.is_file()
    )


def _get_output_file_path(input_file_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Generate the output file path based on the input file path."""
    relative_path = input_file_path.relative_to(input_dir)
    output_subdir = output_dir / relative_path.parent
    output_subdir.mkdir(parents=True, exist_ok=True)
    return output_subdir / input_file_path.stem


def _handle_conversion_result(future: Future) -> None:
    """Handle the result of a conversion task."""
    try:
        result = future.result()
        logging.info(f"Conversion result: {result}")
        logging.info("Task completed successfully.")
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
