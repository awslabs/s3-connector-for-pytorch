import pickle
from typing import Optional

from hypothesis import given, example
from hypothesis.strategies import booleans, integers, none, one_of, builds, text
from s3dataset_s3_client._s3dataset import RestoreStatus, ObjectInfo

restore_status_args = (
    booleans(),
    one_of(none(), integers(min_value=0, max_value=2**128 - 1)),
)

restore_status = builds(RestoreStatus, *restore_status_args)

object_info_args = {
    "key": text(),
    "etag": text(),
    "size": integers(min_value=0, max_value=2**64 - 1),
    "last_modified": integers(min_value=-(2**63), max_value=2**63 - 1),
    "storage_class": one_of(none(), text()),
    "restore_status": one_of(none(), restore_status),
}


@given(*restore_status_args)
@example(False, 2**128 - 1)
def test_restore_status_constructor(in_progress: bool, expiry: Optional[int]):
    restore_status = RestoreStatus(in_progress, expiry)
    assert restore_status.in_progress is in_progress
    assert restore_status.expiry == expiry


@given(*restore_status_args)
def test_restore_status_unpickles(in_progress: bool, expiry: Optional[int]):
    restore_status = RestoreStatus(in_progress, expiry)
    unpickled: RestoreStatus = pickle.loads(pickle.dumps(restore_status))

    assert type(unpickled) is RestoreStatus
    assert restore_status.in_progress is unpickled.in_progress is in_progress
    assert restore_status.expiry == unpickled.expiry == expiry


@given(**object_info_args)
@example("", "", 0, 0, None, None)
@example("", "", 2**64 - 1, 0, None, None)
@example("", "", 0, -(2**63), None, None)
@example("", "", 0, 2**63 - 1, None, None)
def test_object_info_constructor(
    key: str,
    etag: str,
    size: int,
    last_modified: int,
    storage_class: Optional[str],
    restore_status: Optional[RestoreStatus],
):
    object_info = ObjectInfo(
        key, etag, size, last_modified, storage_class, restore_status
    )
    assert object_info.key == key
    assert object_info.size == size
    assert object_info.last_modified == last_modified
    assert object_info.storage_class == storage_class

    if restore_status is None:
        assert object_info.restore_status is None
    else:
        assert object_info.restore_status.expiry == restore_status.expiry
        assert object_info.restore_status.in_progress == restore_status.in_progress


@given(**object_info_args)
def test_object_info_pickles(
    key: str,
    etag: str,
    size: int,
    last_modified: int,
    storage_class: Optional[str],
    restore_status: Optional[RestoreStatus],
):
    object_info = ObjectInfo(
        key, etag, size, last_modified, storage_class, restore_status
    )

    unpickled: ObjectInfo = pickle.loads(pickle.dumps(object_info))

    assert type(unpickled) is ObjectInfo

    assert object_info.key == unpickled.key == key
    assert object_info.size == unpickled.size == size
    assert object_info.last_modified == unpickled.last_modified == last_modified
    assert object_info.storage_class == unpickled.storage_class == storage_class
    if restore_status is None:
        assert object_info.restore_status is unpickled.restore_status is None
    else:
        assert (
            object_info.restore_status.expiry
            == unpickled.restore_status.expiry
            == restore_status.expiry
        )
        assert (
            object_info.restore_status.in_progress
            == unpickled.restore_status.in_progress
            == restore_status.in_progress
        )
        assert (
            object_info.restore_status
            is not unpickled.restore_status
            is not restore_status
        )
