import re
import time
import requests
from tqdm import tqdm
from .BaseStorage import BaseStorage
import urllib.parse


def get_now():
    """Returns the current time in milliseconds."""
    return round(time.time() * 1000)


class HTTPStorage(BaseStorage):
    @staticmethod
    def test(url):
        """Checks if the URL is an HTTP or HTTPS URL."""
        return re.match(r"^https?://", url) is not None

    def __init__(self, url, **kwargs):
        super().__init__(url, **kwargs)
        parts = self.url.split("#", 1)
        self.url = parts[0]
        if len(parts) > 1:
            self.query = urllib.parse.parse_qs(parts[1])

    def __init__(self, url, **kwargs):
        """Initializes HTTPStorage with a given URL and optional extra parameters."""
        super().__init__(url, **kwargs)
        self.url, sep, query_string = self.url.partition('#')
        self.query = urllib.parse.parse_qs(query_string) if query_string else {}

    def upload_file(self, source, dest):
        """Placeholder for HTTP upload functionality, currently not implemented."""
        raise NotImplementedError("HTTP PUT not implemented yet")

    def download_file(self, filename):
        """Downloads a file from the HTTP URL to a specified local filename."""
        print(f"Downloading {self.url} to {filename}...")
        with requests.get(self.url, stream=True) as resp:
            resp.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
            total = int(resp.headers.get("content-length", 0))

            with open(filename, "wb") as file, tqdm(  # Can also replace 'file' with a io.BytesIO object
                desc="Downloading", total=total, unit='iB', unit_scale=True, unit_divisor=1024
            ) as bar:
                total_written = 0
                for data in resp.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
                    total_written += size
                    self.update_status("download", total_written / total)
