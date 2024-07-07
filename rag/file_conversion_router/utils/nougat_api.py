"""Set up a client for interacting with the Nougat API and a server for running the Nougat API.
"""
import time
import atexit
import socket
import subprocess
from http import HTTPStatus
from pathlib import Path
from typing import Literal, Optional

import requests
from requests.exceptions import RequestException

from ..utils.hardware_detection import detect_gpu_setup
from ..utils.logger import get_logger


nougat_logger = get_logger(__name__)

ModelTag = Literal["0.1.0-small", "0.1.0-base"]
# Reference: https://github.com/facebookresearch/nougat?tab=readme-ov-file#api
DEFAULT_SERVER_URL = "http://127.0.0.1"


class NougatAPIClient:
    """A client for interacting with the Nougat API.
    """
    def __init__(self, server_port: str = "8503"):
        self.api_url = f'{DEFAULT_SERVER_URL}:{server_port}'

    def convert_pdf(self, pdf_file: Path, start: Optional[int] = None, stop: Optional[int] = None) -> str:
        """Convert a PDF file to Markdown using the Nougat API format.
        References: https://github.com/facebookresearch/nougat?tab=readme-ov-file#api
        """
        url = f"{self.api_url}/predict/"
        params = {"start": start, "stop": stop}
        params = {k: v for k, v in params.items() if v is not None}

        with pdf_file.open(mode="rb") as file:
            files = {"file": (pdf_file.name, file, "application/pdf")}
            headers = {"accept": "application/json"}

            nougat_logger.debug(f"Sending PDF to Nougat API: {pdf_file}")
            try:
                response = requests.post(url, params=params, files=files, headers=headers)
                return response.text
            except RequestException as e:
                raise Exception(f"Error converting PDF: {e}") from e

    def wait_for_server_ready(self, timeout: int = 60, retry_interval: int = 1):
        """Wait for the Nougat server to be ready.

        Args:
            timeout: The maximum time (in seconds) to wait for the server to be ready.
            retry_interval: The interval (in seconds) between each retry attempt.

        Raises:
            TimeoutError: If the server is not ready within the specified timeout.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # health check
                response = requests.get(f"{self.api_url}/")
                if response.status_code == HTTPStatus.OK:
                    data = response.json()
                    if data.get("status-code") == HTTPStatus.OK:
                        nougat_logger.info("Nougat server is ready")
                        return
            except RequestException:
                pass
            time.sleep(retry_interval)
        raise TimeoutError("Timed out waiting for Nougat server to be ready")


class NougatServer:
    """A class for managing the Nougat server.
    """
    def __init__(self,
                 model_tag: ModelTag = "0.1.0-small",
                 batch_size: int = 4,
                 port: int = 8503,
                 no_skipping: bool = True,
                 recompute: bool = False,
                 ):
        """
        Args:
            model_tag: The tag of the model to use for conversion.
            batch_size: The batch size for processing. Set to 0 when running on CPU.
            port: The port number to use for the server.
            no_skipping: Whether allow nougat to skip pages. Attention, this no skip config does not always work.
        """
        self.model_tag = model_tag
        self.batch_size = batch_size
        self.port = port
        self.device_type, _ = detect_gpu_setup()
        self.no_skipping = no_skipping
        self.recompute = recompute
        self._validate_parameters()

        setup_info = {
            "model": self.model_tag,
            "batch_size": self.batch_size,
            "port": self.port,
            "device_type": self.device_type,
            "no_skipping": self.no_skipping,
            "recompute": self.recompute,
        }
        nougat_logger.info(f"Nougat server is running with the following setup: {setup_info}")

        self.process = None
        atexit.register(self.stop_server)

    def _validate_parameters(self):
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError("Batch size must be a non-negative integer.")
        if self.device_type == "cpu":
            self.batch_size = 0
            nougat_logger.info("Forcing batch size to 0 for running on CPU")

    def is_port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', self.port)) == 0

    def start_server(self):
        """Start the Nougat server if it is not already running.
        """
        if self.process is None:
            if self.is_port_in_use():
                nougat_logger.info(f"Nougat server already running on port {self.port}")
            else:
                nougat_logger.info(f"Starting Nougat server on port {self.port}")
                command = [
                    "nougat_api",
                    "--no-skipping" if self.no_skipping else "",
                    "--recompute" if self.recompute else "",
                    "--model", self.model_tag,
                    "--batchsize", str(self.batch_size),
                    "--port", str(self.port),
                ]
                self.process = subprocess.Popen(command)
                nougat_logger.info(f"Nougat server started with PID: {self.process.pid}")

    def stop_server(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            nougat_logger.info("Nougat server stopped")
