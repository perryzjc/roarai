"""Internal service to process a directory of files and schedule conversion tasks for each file."""
from concurrent.futures import as_completed, Future
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Type, Union
from itertools import islice, tee
from tqdm import tqdm

from rag.file_conversion_router.conversion.base_converter import BaseConverter, ConversionCache
from rag.file_conversion_router.conversion.md_converter import MarkdownConverter
from rag.file_conversion_router.conversion.pdf_converter import PdfConverter
from rag.file_conversion_router.services.task_manager import schedule_conversion
from ..utils.logger import get_logger
from ..config import CONFIG


ConverterMapping = Dict[str, Type[BaseConverter]]


def get_converter_mapping(config=CONFIG) -> ConverterMapping:
    """Generate a mapping from file extensions to their corresponding conversion classes,
    controlling for any converters that are disabled in the configuration.
    """
    converters = {
        '.pdf': PdfConverter,
        '.md': MarkdownConverter
    }
    # Generate dictionary only for enabled converters
    return {extension: converter for extension, converter in converters.items() if
            config.getboolean('converters', f'{extension.split(".")[-1]}_converter', fallback=False)}


_CONVERTER_MAPPING: ConverterMapping = get_converter_mapping(CONFIG)
# Default batch size for processing files
_BATCH_SIZE = 128
logger = get_logger(__name__)


def process_folder(input_dir: Union[str, Path], output_dir: Union[str, Path], batch_size: int = _BATCH_SIZE) -> None:
    """Walk through the input directory and schedule conversion tasks for specified file types."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    _validate_directory(input_dir, "input")
    _ensure_output_directory(output_dir)

    valid_files, valid_files_count = tee(_get_valid_files(input_dir))
    total_files = sum(1 for _ in valid_files_count)

    with tqdm(total=total_files, desc="Processing File Progress", unit="file") as pbar:
        for count, batch in enumerate(_batch_process_files(valid_files, batch_size), start=1):
            logger.debug(f"Processing batch {count} of {len(batch)} files.")
            futures = list(filter(None, (_schedule_conversion_task(file_path, input_dir, output_dir)
                                         for file_path in batch)))
            for future in as_completed(futures):
                _handle_conversion_result(future, pbar)
            logger.debug(f"Completed batch {count} of {len(batch)} files.")

    logger.info(f"Completed processing for directory: {input_dir}")
    logger.info(f"Saved conversion time [{ConversionCache.calc_total_savings()} seconds] by using cached results.")


def _batch_process_files(valid_files: Iterable[Path], batch_size: int) -> Iterable[List[Path]]:
    """Yield batches of valid files for processing."""
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
        logger.info(f"Scheduled conversion for {input_file_path} to {output_file_path}")
        return conversion_future
    logger.warning(f"No converter available for file type {input_file_path.suffix}")
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


def _handle_conversion_result(future: Future, pbar: tqdm) -> None:
    """Handle the result of a conversion task and update the progress bar."""
    try:
        result = future.result()
        logger.info(f"Conversion result: {result}")
        logger.info("Task completed successfully.")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
    finally:
        pbar.update(1)
