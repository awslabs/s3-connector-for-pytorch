"""Distributed Shared Memory Utility for Dataloader"""

import logging
import mmap
import os
import pickle
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, SupportsIndex, Tuple, Union

import torch.distributed as dist

if pickle.HIGHEST_PROTOCOL < 5:
    del pickle
    import pickle5 as pickle  # type: ignore[no-redef,import-not-found]
from pickle import PickleBuffer

logger = logging.getLogger(__name__)

COMPLETE = ".complete"


__all__ = ["save", "load"]


def save(obj: Any, path: Optional[Path] = None) -> Path:
    """Serialize obj with out-of-band data to path for zero-copy shared memory usage.

    If the object to be serialized itself, or the objects it uses for data
    storage (such as numpy arrays) implement the the pickle protocol version 5
    pickle.PickleBuffer type in __reduce_ex__, then this function can store
    these buffers out-of-band as files in `path` so that they subsequently be
    re-used for zero-copy sharing accross processes.

    Args:
        obj (object):
            Object to serialize. For example a PyArrow Table, a Pandas Dataframe or
            any type that relies on NumPy to store the binary data.
        path (pathlib.Path, optional):
            Empty folder used to save serialized data. Usually a folder in /dev/shm
    Returns:
        pathlib.Path where the data was serialized
    """
    idx = 0
    root: Path = path or generate_path()
    root.mkdir(parents=True, exist_ok=True)

    def buffer_callback(buf: PickleBuffer) -> None:
        nonlocal idx
        with open(root / f"{idx}.bin", "wb") as f:
            f.write(buf)
        idx += 1

    with open(root / "meta.pkl", "wb") as f:
        pickle.dump(obj, f, protocol=5, buffer_callback=buffer_callback)

    # mark as saved
    (root / COMPLETE).touch()
    return root


def generate_path() -> Path:
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    path_str = f"/dev/shm/{global_rank}-{uuid.uuid4()}"
    path = Path(path_str)
    return path


def save_iter(
    it: Iterable[Any],
    path: Optional[Path] = None,
    max_items: int = 0,
    should_stop: Callable[[], bool] = lambda: False,
    truncation_threshold: Optional[int] = None,
) -> Path:
    logger.debug(f"save iter {path} ... started")
    path = path or generate_path()
    next_idx = 0
    for i, e in enumerate(it):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"save iter {path} ...")
        if max_items > 0:
            while (cnt := sum(x.is_dir() for x in path.iterdir()) if path.exists() else 0) >= max_items:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"save iter ... {cnt} dirs of {max_items} stop? {should_stop()}")
                if should_stop():
                    break
                time.sleep(0.001)  # busy wait
        if should_stop():
            break
        if truncation_threshold is not None and i == truncation_threshold:
            logger.info(f"reached {truncation_threshold=}, stop saving microbatches")
            break
        save(e, path / str(i))
        next_idx = i + 1
    save(POISON_PILL, path / str(next_idx))
    logger.debug(f"save iter {path} ... finished after {next_idx} microbatches")
    return path


class Sentinel(object):
    """
    Create a unique sentinel object that is pickled as a constant.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name  # pragma: no cover

    def __copy__(self) -> "Sentinel":
        return self  # pragma: no cover

    def __deepcopy__(self, memo: Any) -> "Sentinel":
        return self  # pragma: no cover

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return self.name

    def __reduce_ex__(self, protocol: SupportsIndex) -> Union[str, Tuple[Any, ...]]:
        return self.name


POISON_PILL = Sentinel("POISON_PILL")


def load(path: Path) -> Any:
    """Load serialized object with out-of-band data from path based on zero-copy shared memory.

    Args:
        path (pathlib.Path):
            Folder used to save serialized data with serialize(). Usually a folder /dev/shm
    Returns:
        Raw deserialized data
    """
    if not saved(path):
        raise RuntimeError(f"The object at {path} is corrupted or not saved")
    buffers: List[Union[PickleBuffer, mmap.mmap]] = []
    num_buffers = len(list(path.iterdir())) - 2  # exclude meta.pkl and .complete
    for idx in range(num_buffers):
        fpath = path / f"{idx}.bin"
        if os.stat(fpath).st_size == 0:
            buffers.append(PickleBuffer(b""))
        else:
            with open(fpath, "rb") as f:
                buffers.append(mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ))
    with open(path / "meta.pkl", "rb") as f:
        obj = pickle.load(f, buffers=buffers)
    shutil.rmtree(path)
    logger.debug(f"removed {path}")
    return obj


def saved(path: Path) -> bool:
    return (path / COMPLETE).exists()


def load_iter(
    path: Path, next_timeout: int = 10 * 60, wait_callback: Callable[[], None] = lambda: None) -> Iterator[Any]:
    idx = 0
    while True:
        start_time = time.time()
        wait_time_threshold = start_time + next_timeout
        chunk_path = path / str(idx)
        while not saved(chunk_path):
            wait_callback()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"waiting for data in {chunk_path}")
            if time.time() > wait_time_threshold:
                logger.error(f"timed out waiting for {chunk_path}")
                raise TimeoutError
            time.sleep(0.001)  # busy wait
        chunk = load(chunk_path)
        if chunk is POISON_PILL:
            logger.debug("poison pill!")
            break
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"fetching microbatch took {time.time() - start_time}s")
        yield chunk
        idx += 1
    return