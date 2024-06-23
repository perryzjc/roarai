import requests
import subprocess
import atexit
import socket
import time
from pathlib import Path
from typing import Optional

from rag.file_conversion_router.conversion.base_converter import BaseConverter
from rag.file_conversion_router.utils.hardware_detection import detect_gpu_setup
from ..utils.logger import get_logger


pdf_converter_logger = get_logger(__name__)


class NougatAPIClient:
    def __init__(self, api_url: str = "http://127.0.0.1:8503"):
        self.api_url = api_url

    def convert_pdf(self, pdf_file: Path, start: Optional[int] = None, stop: Optional[int] = None) -> str:
        url = f"{self.api_url}/predict/"
        params = {}
        if start is not None:
            params["start"] = start
        if stop is not None:
            params["stop"] = stop

        with pdf_file.open(mode="rb") as file:
            files = {"file": (pdf_file.name, file, "application/pdf")}
            headers = {"accept": "application/json"}

            pdf_converter_logger.debug(f"Sending PDF to Nougat API: {pdf_file}")
            try:
                response = requests.post(url, params=params, files=files, headers=headers)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                raise Exception(f"Error converting PDF: {e}") from e


class NougatServer:
    def __init__(self, model_tag: str = "0.1.0-small", batch_size: int = 4, port: int = 8503):
        self.model_tag = model_tag
        self.batch_size = batch_size
        self.port = port
        self.device_type, _ = detect_gpu_setup()
        self._validate_parameters()
        pdf_converter_logger.info(f"Using {self.device_type} on Torch")
        self.process = None
        atexit.register(self.stop_server)

    def _validate_parameters(self):
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError("Batch size must be a non-negative integer.")
        if self.device_type == "cpu":
            self.batch_size = 0
            pdf_converter_logger.info("Forcing batch size to 0 for running on CPU")
        acceptable_models = ["0.1.0-small", "0.1.0-base"]
        if self.model_tag not in acceptable_models:
            raise ValueError(f"Model tag must be one of {acceptable_models}")

    def is_port_in_use(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', self.port)) == 0

    def start_server(self):
        if self.process is None:
            if self.is_port_in_use():
                pdf_converter_logger.info(f"Nougat server already running on port {self.port}")
            else:
                pdf_converter_logger.info(f"Starting Nougat server on port {self.port}")
                command = [
                    "nougat_api",
                    "--no-skipping",
                    "--model",
                    self.model_tag,
                    "--batchsize",
                    str(self.batch_size),
                    "--port",
                    str(self.port),
                ]
                self.process = subprocess.Popen(command)
                pdf_converter_logger.info(f"Nougat server started with PID: {self.process.pid}")

    def stop_server(self):
        print("Stopping Nougat server")
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            print("Nougat server stopped")


class PdfConverter(BaseConverter):
    nougat_server = NougatServer()
    nougat_api_client = NougatAPIClient()

    def __init__(self):
        super().__init__()
        self.nougat_server.start_server()
        self._wait_for_server_ready()

    def _wait_for_server_ready(self, timeout: int = 60, retry_interval: int = 1):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Send a request to the API endpoint
                response = requests.get(f"{self.nougat_api_client.api_url}/predict/")
                if response.status_code == 405:  # Method Not Allowed
                    pdf_converter_logger.info("Nougat server is ready")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(retry_interval)
        raise TimeoutError("Timed out waiting for Nougat server to be ready")

    def _to_markdown(self, input_path: Path, output_path: Path) -> None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_text = self.nougat_api_client.convert_pdf(input_path)
            # Remove the outer quotes and replace escaped newline characters with actual newlines
            markdown_text = markdown_text.strip('"').replace('\\n', '\n')
            with open(output_path, "w") as file:
                file.write(markdown_text)
        except Exception as e:
            pdf_converter_logger.error(f"An error occurred: {str(e)}")
            raise
