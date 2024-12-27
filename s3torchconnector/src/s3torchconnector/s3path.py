#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import errno
import io
import logging
import os
import posixpath
import stat
import time
from types import SimpleNamespace
from typing import Optional

from pathlib import PurePosixPath
from pathlib_abc import ParserBase, PathBase, UnsupportedOperation
from urllib.parse import urlparse

from s3torchconnectorclient._mountpoint_s3_client import S3Exception
from ._s3client import S3Client, S3ClientConfig

logger = logging.getLogger(__name__)

ENV_S3_TORCH_CONNECTOR_REGION = "S3_TORCH_CONNECTOR_REGION"
ENV_S3_TORCH_CONNECTOR_THROUGHPUT_TARGET_GPBS = (
    "S3_TORCH_CONNECTOR_THROUGHPUT_TARGET_GPBS"
)
ENV_S3_TORCH_CONNECTOR_PART_SIZE_MB = "S3_TORCH_CONNECTOR_PART_SIZE_MB"
DRIVE = "s3://"


def _get_default_bucket_region():
    for var in [
        ENV_S3_TORCH_CONNECTOR_REGION,
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
        "REGION",
    ]:
        if var in os.environ:
            return os.environ[var]


def _get_default_throughput_target_gbps():
    if ENV_S3_TORCH_CONNECTOR_THROUGHPUT_TARGET_GPBS in os.environ:
        return float(os.environ[ENV_S3_TORCH_CONNECTOR_THROUGHPUT_TARGET_GPBS])


def _get_default_part_size():
    if ENV_S3_TORCH_CONNECTOR_PART_SIZE_MB in os.environ:
        return int(os.environ[ENV_S3_TORCH_CONNECTOR_PART_SIZE_MB]) * 1024 * 1024


class S3Parser(ParserBase):
    @classmethod
    def _unsupported_msg(cls, attribute):
        return f"{cls.__name__}.{attribute} is unsupported"

    @property
    def sep(self):
        return "/"

    def join(self, path, *paths):
        return posixpath.join(path, *paths)

    def split(self, path):
        scheme, bucket, prefix, _, _, _ = urlparse(path)
        parent, _, name = prefix.lstrip("/").rpartition("/")
        if not bucket:
            return bucket, name
        return (scheme + "://" + bucket + "/" + parent, name)

    def splitdrive(self, path):
        scheme, bucket, prefix, _, _, _ = urlparse(path)
        drive = f"{scheme}://{bucket}"
        return drive, prefix.lstrip("/")

    def splitext(self, path):
        return posixpath.splitext(path)

    def normcase(self, path):
        return posixpath.normcase(path)

    def isabs(self, path):
        s = os.fspath(path)
        scheme_tail = s.split("://", 1)
        return len(scheme_tail) == 2


class S3Path(PathBase):
    __slots__ = ("_region", "_s3_client_config", "_client", "_raw_path")
    parser = S3Parser()
    _stat_cache_ttl_seconds = 1
    _stat_cache_size = 1024
    _stat_cache = {}

    def __init__(
        self,
        *pathsegments,
        client: Optional[S3Client] = None,
        region=None,
        s3_client_config=None,
    ):
        super().__init__(*pathsegments)
        if not self.drive.startswith(DRIVE):
            raise ValueError("Should pass in S3 uri")
        self._region = region or _get_default_bucket_region()
        self._s3_client_config = s3_client_config or S3ClientConfig(
            throughput_target_gbps=_get_default_throughput_target_gbps(),
            part_size=_get_default_part_size(),
        )
        self._client = client or S3Client(
            region=self._region,
            s3client_config=self._s3_client_config,
        )

    def __repr__(self):
        return f"{type(self).__name__}({str(self)!r})"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, S3Path):
            return NotImplemented
        return str(self) == str(other)

    def with_segments(self, *pathsegments):
        path = str("/".join(pathsegments)).lstrip("/")
        if not path.startswith(self.anchor):
            path = f"{self.anchor}{path}"
        return type(self)(
            path,
            client=self._client,
            region=self._region,
            s3_client_config=self._s3_client_config,
        )

    @property
    def bucket(self):
        if self.is_absolute() and self.drive.startswith(DRIVE):
            return self.drive[5:]
        return ""

    @property
    def key(self):
        if self.is_absolute() and len(self.parts) > 1:
            return self.parser.sep.join(self.parts[1:])
        return ""

    def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if buffering != -1:
            raise ValueError("Only default buffering (-1) is supported.")
        if not self.is_absolute():
            raise ValueError("S3Path must be absolute.")
        action = "".join(c for c in mode if c not in "btU")
        if action == "r":
            try:
                fileobj = self._client.get_object(self.bucket, self.key)
            except S3Exception:
                raise FileNotFoundError(errno.ENOENT, "Not found", str(self)) from None
            except:
                raise
        elif action == "w":
            try:
                fileobj = self._client.put_object(self.bucket, self.key)
            except S3Exception:
                raise
            except:
                raise
        else:
            raise UnsupportedOperation()
        if "b" not in mode:
            fileobj = io.TextIOWrapper(fileobj, encoding, errors, newline)
        return fileobj

    def stat(self, *, follow_symlinks=True):
        cache_key = (self.bucket, self.key.rstrip("/"))
        cached_result = self._stat_cache.get(cache_key)
        if cached_result:
            result, timestamp = cached_result
            if time.time() - timestamp < self._stat_cache_ttl_seconds:
                return result
            del self._stat_cache[cache_key]
        try:
            info = self._client.head_object(self.bucket, self.key.rstrip("/"))
            mode = stat.S_IFREG
        except S3Exception as e:
            listobj = next(self._list_objects(max_keys=2))

            if len(listobj.object_info) > 0 or len(listobj.common_prefixes) > 0:
                info = SimpleNamespace(size=0, last_modified=None)
                mode = stat.S_IFDIR
            else:
                error_msg = f"No stats available for {self}; it may not exist."
                raise FileNotFoundError(error_msg) from e

        result = os.stat_result(
            (
                mode,  # mode
                None,  # ino
                DRIVE,  # dev
                None,  # nlink
                None,  # uid
                None,  # gid
                info.size,  # size
                None,  # atime
                info.last_modified or 0,  # mtime
                None,  # ctime
            )
        )
        if len(self._stat_cache) >= self._stat_cache_size:
            self._stat_cache.pop(next(iter(self._stat_cache)))

        self._stat_cache[cache_key] = (result, time.time())
        return result

    def iterdir(self):
        if not self.is_dir():
            raise NotADirectoryError("not a s3 folder")
        key = "" if not self.key else self.key.rstrip("/") + "/"
        for page in self._list_objects():
            for prefix in page.common_prefixes:
                # yield directories first
                yield self.with_segments(prefix.rstrip("/"))
            for info in page.object_info:
                if info.key != key:
                    yield self.with_segments(info.key)

    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if self.is_dir():
            if exist_ok:
                return
            raise FileExistsError(f"S3 folder {self} already exists.")
        with self._client.put_object(self.bucket, self.key.rstrip("/") + "/"):
            pass

    def unlink(self, missing_ok=False):
        if self.is_dir():
            if missing_ok:
                return
            raise IsADirectoryError(
                f"Path {self} is a directory; call rmdir instead of unlink."
            )
        self._client.delete_object(self.bucket, self.key)

    def rmdir(self):
        if not self.is_dir():
            raise NotADirectoryError(f"{self} is not an s3 folder")
        listobj = next(self._list_objects(max_keys=2))
        if len(listobj.object_info) > 1:
            raise Exception(f"{self} is not empty")
        self._client.delete_object(self.bucket, self.key.rstrip("/") + "/")

    def glob(self, pattern, *, case_sensitive=None, recurse_symlinks=True):
        if ".." in pattern:
            raise NotImplementedError(
                "Relative paths with '..' not supported in glob patterns"
            )
        if pattern.startswith(self.anchor) or pattern.startswith("/"):
            raise NotImplementedError("Non-relative patterns are unsupported")

        parts = list(PurePosixPath(pattern).parts)
        select = self._glob_selector(parts, case_sensitive, recurse_symlinks)
        return select(self)

    def with_name(self, name):
        """Return a new path with the file name changed."""
        split = self.parser.split
        if split(name)[0]:
            # Ensure that the provided name does not contain any path separators
            raise ValueError(f"Invalid name {name!r}")
        return self.with_segments(str(self.parent), name)

    def _list_objects(self, max_keys: int = 1000):
        try:
            key = "" if not self.key else self.key.rstrip("/") + "/"
            pages = iter(
                self._client.list_objects(
                    self.bucket, key, delimiter="/", max_keys=max_keys
                )
            )
            for page in pages:
                yield page
        except S3Exception as e:
            raise RuntimeError(f"Failed to list contents of {self}") from e

    def __getstate__(self):
        state = {
            slot: getattr(self, slot, None)
            for cls in self.__class__.__mro__
            for slot in getattr(cls, "__slots__", [])
            if slot
            not in [
                "_client",
            ]
        }
        return (None, state)

    def __setstate__(self, state):
        _, state_dict = state
        for slot, value in state_dict.items():
            if slot not in ["_client"]:
                setattr(self, slot, value)
        self._client = S3Client(
            region=self._region,
            s3client_config=self._s3_client_config,
        )
