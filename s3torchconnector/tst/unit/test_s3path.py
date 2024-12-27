import collections
import io
import time
import pytest

from pathlib_abc import PathBase

from s3torchconnector import S3Path
from s3torchconnector._s3client._s3client import S3Client
from s3torchconnector._s3client._mock_s3client import MockS3Client


def s3_uri(bucket, key=None):
    return f"s3://{bucket}" if key is None else f"s3://{bucket}/{key}"


TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
TEST_REGION = "us-east-1"
TEST_S3_URI = s3_uri(TEST_BUCKET, TEST_KEY)
MISSING_S3_URI = s3_uri(TEST_BUCKET, "foo")


@pytest.fixture
def s3_client() -> S3Client:
    client = MockS3Client(TEST_REGION, TEST_BUCKET)
    return client


@pytest.fixture
def s3_bucket_path(s3_client) -> S3Path:
    s3_bucket_path = S3Path(s3_uri(TEST_BUCKET))
    s3_bucket_path._client = s3_client
    return s3_bucket_path


@pytest.fixture
def s3_path(s3_bucket_path) -> S3Path:
    s3_path = s3_bucket_path / TEST_KEY
    s3_path._client.add_object(TEST_KEY, b"this is an s3 file\n")
    return s3_path


@pytest.fixture
def missing_s3_path(s3_bucket_path) -> S3Path:
    return s3_bucket_path / MISSING_S3_URI


def test_s3path_subclass_path(s3_path: S3Path):
    assert issubclass(S3Path, PathBase)
    assert isinstance(s3_path, PathBase)


def test_s3path_creation(s3_path: S3Path):
    assert s3_path
    assert s3_path.bucket == TEST_BUCKET
    assert s3_path.key == TEST_KEY


@pytest.mark.parametrize(
    "path",
    [(""), (TEST_KEY)],
)
def test_s3path_invalid_creation(path):
    with pytest.raises(ValueError, match="Should pass in S3 uri"):
        S3Path(path)


def test_s3path_samefile(s3_path, missing_s3_path):
    assert s3_path.samefile(TEST_S3_URI)

    with pytest.raises(FileNotFoundError):
        s3_path.samefile(MISSING_S3_URI)
    with pytest.raises(FileNotFoundError):
        s3_path.samefile(missing_s3_path)
    with pytest.raises(FileNotFoundError):
        missing_s3_path.samefile(TEST_S3_URI)
    with pytest.raises(FileNotFoundError):
        missing_s3_path.samefile(s3_path)


def test_s3path_exists(s3_path, s3_bucket_path, missing_s3_path):
    assert s3_path.exists() is True
    assert s3_bucket_path.exists() is True
    assert missing_s3_path.exists() is False


def test_s3path_open(s3_path):
    with s3_path.open("r") as reader:
        assert isinstance(reader, io.TextIOBase)
        assert reader.read() == "this is an s3 file\n"
    with s3_path.open("rb") as reader:
        assert isinstance(reader, io.BufferedIOBase)
        assert reader.read().strip() == b"this is an s3 file"


def test_s3path_read_write_bytes(s3_path):
    (s3_path / "fileA").write_bytes(b"abcd")
    assert (s3_path / "fileA").read_bytes() == b"abcd"

    with pytest.raises(TypeError):
        (s3_path / "fileA").write_bytes("somestr")
    assert (s3_path / "fileA").read_bytes() == b"abcd"


def test_s3path_read_write_text(s3_path):
    (s3_path / "fileA").write_text("äbcd", encoding="latin-1")
    assert (s3_path / "fileA").read_text(encoding="utf-8", errors="ignore") == "bcd"

    with pytest.raises(TypeError):
        (s3_path / "fileA").write_text(b"somebytes")
    assert (s3_path / "fileA").read_text(encoding="latin-1") == "äbcd"


def test_s3path_iterdir(s3_path, s3_bucket_path):
    s3_bucket_path._client.add_object("file1.txt", b"file 1 content")
    s3_bucket_path._client.add_object("file2.txt", b"file 2 content")
    s3_bucket_path._client.add_object("dir1/file3.txt", b"nested file")
    (s3_bucket_path / "dir1" / "nested_dir").mkdir()
    (s3_bucket_path / "dir2").mkdir()

    bucket_contents = list(s3_bucket_path.iterdir())
    assert bucket_contents == [
        s3_bucket_path / "dir1",
        s3_bucket_path / "dir2",
        s3_bucket_path / "file1.txt",
        s3_bucket_path / "file2.txt",
        s3_path,
    ]

    dir1_path = s3_bucket_path / "dir1"
    dir1_contents = list(dir1_path.iterdir())
    assert dir1_contents == [dir1_path / "nested_dir", dir1_path / "file3.txt"]


def test_s3path_nondir(s3_path):
    with pytest.raises(NotADirectoryError):
        # does not follow pathlib in python 3.13+, which raises immediately before iterating
        next(s3_path.iterdir())


def test_s3path_glob(s3_path, s3_bucket_path):
    it = s3_bucket_path.glob(TEST_KEY)
    assert isinstance(it, collections.abc.Iterator)
    assert set(it) == {s3_path}


def test_s3path_glob_empty_pattern(s3_path):
    # no relative paths in s3
    assert list(s3_path.glob("")) == [s3_path]
    assert list(s3_path.glob(".")) == [s3_path]
    assert list(s3_path.glob("./")) == [s3_path]


def test_s3path_stat(s3_path, s3_bucket_path):
    stat_file = s3_path.stat()
    stat_folder = s3_bucket_path.stat()

    # no concept of directories in s3, existing folders count as "directories"
    assert isinstance(stat_file.st_mode, int)
    assert stat_file.st_mode != stat_folder.st_mode

    assert stat_file.st_dev == "s3://"
    assert stat_file.st_dev == stat_folder.st_dev


def test_s3path_isdir(s3_path, s3_bucket_path, missing_s3_path):
    assert not s3_path.is_dir()
    assert s3_bucket_path.is_dir()
    # even though directories don't matter in s3, count missing prefixes as non directories
    assert not missing_s3_path.is_dir()


def test_s3path_withname(s3_path):
    new_name = "new_file.txt"
    new_path = s3_path.with_name(new_name)
    assert new_path.key == new_name, f"Expected {new_name}, got {new_path.key}"

    s3_path_with_slash = s3_path.with_segments("folder", "old_file.txt")
    new_path_with_slash = s3_path_with_slash.with_name(new_name)
    assert (
        new_path_with_slash.key == "folder/new_file.txt"
    ), f"Expected folder/new_file.txt, got {new_path_with_slash.key}"

    try:
        s3_path.with_name("invalid/name.txt")
    except ValueError as e:
        assert (
            str(e) == "Invalid name 'invalid/name.txt'"
        ), f"Unexpected error message: {e}"


def test_s3path_rmdir(s3_path):
    empty_folder = s3_path / "empty"
    empty_folder.mkdir(parents=True, exist_ok=True)
    empty_folder.rmdir()
    with pytest.raises(NotADirectoryError, match=f"{empty_folder} is not an s3 folder"):
        time.sleep(1)  # S3 needs some time to register the deletion
        empty_folder.rmdir()

    nonempty_folder = s3_path / "nonempty"
    nonempty_folder.mkdir(parents=True, exist_ok=True)
    nonempty_folder._client.add_object("test-key/nonempty/file.txt", b"file")
    with pytest.raises(Exception, match=f"{nonempty_folder} is not empty"):
        nonempty_folder.rmdir()

    nonexistent_folder = s3_path / "nonexistent_folder"
    with pytest.raises(
        NotADirectoryError, match=f"{nonexistent_folder} is not an s3 folder"
    ):
        nonexistent_folder.rmdir()


def test_s3path_unlink(s3_path):
    file = s3_path / "test_file.txt"
    s3_path._client.add_object("test-key/test_file.txt", b"")
    assert file.exists()
    file.unlink()
    time.sleep(1)  # S3 needs some time to register the deletion
    assert not file.exists()

    directory = s3_path / "some_directory"
    directory.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError):
        directory.unlink()

    nonexistent_file = s3_path / "nonexistent_file.txt"
    nonexistent_file.unlink()  # no op
    assert not nonexistent_file.exists()


def test_s3path_mkdir(s3_path):
    test_dir = s3_path / "test_dir"
    test_dir.mkdir(parents=True, exist_ok=False)

    assert test_dir.exists()
    assert test_dir.is_dir()

    with pytest.raises(FileExistsError, match=f"{test_dir} already exists"):
        test_dir.mkdir(parents=True, exist_ok=False)

    test_dir.mkdir(parents=True, exist_ok=True)
    assert test_dir.exists()

    parent_dir = s3_path / "parent_dir"
    sub_dir = parent_dir / "sub_dir"
    sub_dir.mkdir(parents=True, exist_ok=False)

    assert parent_dir.exists()
    assert sub_dir.exists()