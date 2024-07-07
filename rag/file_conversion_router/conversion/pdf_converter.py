"""Convert PDF files to Markdown using the Nougat API."""
from pathlib import Path
import subprocess

from ..utils.logger import get_logger
from ..utils.nougat_api import NougatAPIClient, NougatServer
from rag.file_conversion_router.conversion.base_converter import BaseConverter
from ..config import CONFIG

pdf_converter_logger = get_logger(__name__)


class PdfConverter(BaseConverter):
    """A converter for converting PDF files to Markdown using the Nougat API."""

    USE_NOUGAT_SERVER = CONFIG.getboolean("pdf_converter.nougat", "use_nougat_server")
    if USE_NOUGAT_SERVER:
        nougat_server = NougatServer(
            model_tag=CONFIG.get("pdf_converter.nougat", "model_tag"),
            batch_size=CONFIG.getint("pdf_converter.nougat", "batch_size"),
            port=CONFIG.getint("pdf_converter.nougat", "server_port"),
            no_skipping=CONFIG.get("pdf_converter.nougat", "no_skipping"),
        )
        nougat_api_client = NougatAPIClient(
            f'{CONFIG.getint("pdf_converter.nougat", "server_port")}'
        )
        nougat_server.start_server()
        nougat_api_client.wait_for_server_ready()
    else:
        model_tag = CONFIG.get("pdf_converter.nougat", "model_tag")
        batch_size = CONFIG.getint("pdf_converter.nougat", "batch_size")

    def _to_markdown(self, input_path: Path, output_path: Path) -> None:
        if self.USE_NOUGAT_SERVER:
            self._convert_using_nougat_server(input_path, output_path)
        else:
            self._convert_using_nougat_cli(input_path, output_path)

    @classmethod
    def _convert_using_nougat_server(cls, input_path: Path, output_path: Path) -> None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_text = cls.nougat_api_client.convert_pdf(input_path)
            markdown_text = markdown_text.strip('"').replace('\\n', '\n')
            with open(output_path, "w") as file:
                file.write(markdown_text)
        except Exception as e:
            pdf_converter_logger.error(f"An error occurred: {str(e)}")
            raise

    @classmethod
    def _convert_using_nougat_cli(cls, input_path: Path, output_path: Path) -> None:
        command = [
            "nougat",
            str(input_path),
            "-o",
            str(output_path.parent),
            "--no-skipping",
            # "--recompute",
            "--model",
            cls.model_tag,
            "--batchsize",
            str(cls.batch_size),
        ]
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            pdf_converter_logger.info(f"Output: {result.stdout}")
            pdf_converter_logger.info(f"Errors: {result.stderr}")
            if result.returncode != 0:
                pdf_converter_logger.error(f"Command exited with a non-zero status: {result.returncode}")
            output_mmd_path = output_path.parent / f"{input_path.stem}.mmd"
            output_mmd_path.rename(output_path)
        except Exception as e:
            pdf_converter_logger.error(f"An error occurred {str(e)})")
            raise
