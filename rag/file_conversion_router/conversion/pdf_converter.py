"""Convert PDF files to Markdown using the Nougat API."""
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.nougat_api import NougatAPIClient, NougatServer
from rag.file_conversion_router.conversion.base_converter import BaseConverter
from ..config import CONFIG

pdf_converter_logger = get_logger(__name__)


class PdfConverter(BaseConverter):
    """A converter for converting PDF files to Markdown using the Nougat API."""

    nougat_server = NougatServer(
        model_tag=CONFIG.get("pdf_converter.nougat", "model_tag"),
        batch_size=CONFIG.getint("pdf_converter.nougat", "batch_size"),
        port=CONFIG.getint("pdf_converter.nougat", "server_port"),
        no_skipping=CONFIG.get("pdf_converter.nougat", "no_skipping"),
    )
    nougat_api_client = NougatAPIClient(
        f'{CONFIG.getint("pdf_converter.nougat", "server_port")}'
    )

    def __init__(self):
        super().__init__()
        self.nougat_server.start_server()
        self.nougat_api_client.wait_for_server_ready()

    def _to_markdown(self, input_path: Path, output_path: Path) -> None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_text = self.nougat_api_client.convert_pdf(input_path)
            markdown_text = markdown_text.strip('"').replace('\\n', '\n')
            with open(output_path, "w") as file:
                file.write(markdown_text)
        except Exception as e:
            pdf_converter_logger.error(f"An error occurred: {str(e)}")
            raise
