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

    def __init__(self):
        super().__init__()
        self.use_nougat_server = CONFIG.getboolean("pdf_converter.nougat", "use_nougat_server")
        self.no_skip = CONFIG.getboolean("pdf_converter.nougat", "no_skipping")
        self.batch_size = CONFIG.getint("pdf_converter.nougat", "batch_size")
        self.model_tag = CONFIG.get("pdf_converter.nougat", "model_tag")
        self.recompute = CONFIG.getboolean("pdf_converter.nougat", "recompute")

        if self.use_nougat_server:
            self.nougat_server = NougatServer(
                model_tag=self.model_tag,
                batch_size=self.batch_size,
                port=CONFIG.getint("pdf_converter.nougat", "server_port"),
                no_skipping=self.no_skip,
            )
            self.nougat_api_client = NougatAPIClient(
                f'{CONFIG.getint("pdf_converter.nougat", "server_port")}'
            )
            self.nougat_server.start_server()
            self.nougat_api_client.wait_for_server_ready()

    def _to_markdown(self, input_path: Path, output_path: Path) -> None:
        if self.use_nougat_server:
            self._convert_using_nougat_server(input_path, output_path)
        else:
            self._convert_using_nougat_cli(input_path, output_path)

    def _convert_using_nougat_server(self, input_path: Path, output_path: Path) -> None:
        pdf_converter_logger.info(f"Converting PDF file using Nougat Server.")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_text = self.nougat_api_client.convert_pdf(input_path)
            markdown_text = markdown_text.strip('"').replace('\\n', '\n')
            with open(output_path, "w") as file:
                file.write(markdown_text)
        except Exception as e:
            pdf_converter_logger.error(f"An error occurred: {str(e)}")
            raise

    def _convert_using_nougat_cli(self, input_path: Path, output_path: Path) -> None:
        command = [
            "nougat",
            str(input_path),
            "-o",
            str(output_path.parent),
            "--no-skipping" if self.no_skip else "",
            "--recompute" if self.recompute else "",
            "--model",
            self.model_tag,
            "--batchsize",
            str(self.batch_size),
        ]
        command = [arg for arg in command if arg]
        pdf_converter_logger.info(f"Converting PDF file using Nougat CLI. Command used: {command}")
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
