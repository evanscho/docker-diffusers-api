import boto3
import botocore
import re
import os
import time
from tqdm import tqdm
from botocore.client import Config
from .BaseStorage import BaseStorage
from lib.vars import AWS_S3_ENDPOINT_URL, AWS_S3_DEFAULT_BUCKET


def get_now():
    return round(time.time() * 1000)


class S3Storage(BaseStorage):
    @staticmethod
    def test(url):
        """Check if the URL is an S3 URL."""
        return re.search(r"^(https?\+)?s3://", url)

    def __init__(self, url, **kwargs):
        super().__init__(url, **kwargs)
        self.setup_s3_details(url, **kwargs)

    def setup_s3_details(self, url, **kwargs):
        """Parses the URL and sets up S3 connection details."""
        # Normalize the URL to remove s3 custom protocols
        url = re.sub(r"^(https?\+)?s3://", lambda m: "http://" if m.group(1) == "http+" else "https://", url)
        self.endpoint_url, self.bucket_name, self.path = self.parse_s3_url(url, **kwargs)

        self._s3resource = boto3.resource(
            "s3",
            endpoint_url=self.endpoint_url,
            config=Config(signature_version="s3v4"),
        )
        self._s3client = None
        self._bucket = None
        print("self.endpoint_url", self.endpoint_url)

    def parse_s3_url(self, url, **kwargs):
        """Extract bucket and path from URL."""
        s3_dest = re.match(r"^(?P<endpoint>https?://[^/]*)(/(?P<bucket>[^/]+))?(/(?P<path>.*))?$", url)
        if not s3_dest:
            raise ValueError("Invalid S3 URL")

        s3_dest = s3_dest.groupdict()
        endpoint = s3_dest["endpoint"]
        bucket = s3_dest["bucket"]
        path = s3_dest["path"]

        print("endpoint", endpoint)

        return (endpoint or AWS_S3_ENDPOINT_URL, bucket or AWS_S3_DEFAULT_BUCKET, path or kwargs.get("default_path", ""))

    def upload_file(self, source, dest=None):
        """Uploads a file to S3."""
        if not dest:
            # ESS: was "dest = self.path", but then path needs to be set first
            dest = os.path.basename(source)

        upload_start = get_now()
        file_size = os.stat(source).st_size

        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading") as bar:
            self.bucket().upload_file(
                Filename=source, Key=dest,
                Callback=lambda bytes_transferred: self.progress_callback(
                    bytes_transferred, file_size, bar, "upload"
                )
            )
        upload_total = get_now() - upload_start
        print(f"Upload completed in {upload_total} milliseconds.")
        return {"$time": upload_total}

    def download_file(self, dest):
        """Downloads a file from S3."""
        if not dest:
            dest = os.path.basename(self.path)
        print(f"Downloading from {self.path} to {dest}...")

        download_start = get_now()
        object = self.s3resource().Object(self.bucket_name, self.path)
        object.load()

        with tqdm(
            total=object.content_length, unit="B", unit_scale=True, desc="Downloading"
        ) as bar:
            object.download_file(
                Filename=dest,
                Callback=lambda bytes_transferred: self.progress_callback(
                    bytes_transferred, object.content_length, bar, "download"
                )
            )
        download_total = get_now() - download_start
        print(f"Download completed in {download_total} milliseconds.")
        return {"$time": download_total}

    def progress_callback(self, bytes_transferred, total_size, bar, operation_type):
        """Generic progress callback to update the tqdm progress bar and status."""
        # bar.n is the current position of the progress bar in tqdm
        increment = bytes_transferred - bar.n  # Calculate the increment since the last update/chunk
        bar.update(increment)  # Update the progress bar by the increment
        self.updateStatus(operation_type, bar.n / total_size)  # Update the status with the new progress

    @property
    def bucket(self):
        """Returns the S3 bucket resource."""
        return self._s3resource.Bucket(self.bucket_name)

    def file_exists(self):
        """Check if the file exists in S3."""
        try:
            self._s3resource.Object(self.bucket_name, self.path).load()
        except botocore.exceptions.ClientError as error:
            if error.response['Error']['Code'] == "404":
                return False
            else:
                raise
        return True
