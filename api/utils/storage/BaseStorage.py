import os
import re
import subprocess
from abc import ABC, abstractmethod
import xtarfile as tarfile


class BaseArchive(ABC):
    def __init__(self, path: str, status=None):
        self.path = path
        self.status = status

    def update_status(self, type: str, progress: float):
        if self.status:
            self.status.update(type, progress)

    def extract(self, dir: str, dry_run: bool = False):
        """Extract the archive to a directory."""
        pass  # DDA: TODO

    def splitext(self):
        base, ext = os.path.splitext(self.path)
        base, subext = os.path.splitext(base)
        return base, ext, subext


class TarArchive(BaseArchive):
    @staticmethod
    def test(path: str):
        return re.search(r"\.tar(\.gz|\.bz2)?$", path)

    def extract(self, dir: str = None, dry_run: bool = False):
        self.update_status("extract", 0)
        if not dir:
            parent_dir = os.path.dirname(self.path)
            base_filename = self.splitext()[0]
            dir = os.path.join(parent_dir, base_filename)

        if not dry_run:
            os.makedirs(dir, exist_ok=True)
            with tarfile.open(self.path, "r") as tar:
                members = tar.getmembers()
                total = len(members)
                for i, member in enumerate(members, 1):
                    tar.extract(member, path=dir)
                    self.update_status("extract", i / total)

            subprocess.run(["ls", "-l", dir])
            os.remove(self.path)

        self.update_status("extract", 1)
        return dir


archive_classes = [TarArchive]


def get_archive_class(path: str, **kwargs):
    for ArchiveClass in archive_classes:
        if ArchiveClass.test(path):
            return ArchiveClass(path, **kwargs)
    return None


class BaseStorage(ABC):
    @staticmethod
    @abstractmethod
    def test(url: str):
        """Check if the URL is valid for this storage type."""
        pass

    def __init__(self, url: str, **kwargs):
        self.url = url
        self.status = kwargs.get("status", None)
        self.query = {}

    def update_status(self, type: str, progress: float):
        if self.status:
            self.status.update(type, progress)

    def splitext(self):
        base, ext = os.path.splitext(self.url)
        base, subext = os.path.splitext(base)
        return base, ext, subext

    def get_filename(self):
        return os.path.basename(self.url)

    @abstractmethod
    def download_file(self, dest: str):
        """Download the file to `dest`"""
        pass

    def download_and_extract(self, filename: str = None, dir: str = None, dry_run: bool = False):
        """
        Downloads the file, and if it's an archive, extract it too.  Returns
        the filename if not, or directory name (filename without extension) if
        it was.
        """
        filename = filename or self.get_filename()
        archive_class = get_archive_class(filename)
        if archive_class:
            archive = archive_class(filename, status=self.status)

            # TODO, streaming pipeline
            self.download_file(filename)
            return archive.extract(dir, dry_run)
        else:
            self.download_file(filename)
            return filename
