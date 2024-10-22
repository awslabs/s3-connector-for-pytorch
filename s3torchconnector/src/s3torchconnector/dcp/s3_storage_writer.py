#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import dataclasses
import pickle
import queue
import threading
import os
from typing import List, Optional, Union, Dict, Any
import logging


from attr import dataclass
from torch.distributed.checkpoint import Metadata, StorageWriter, SavePlan, SavePlanner
from torch.distributed.checkpoint.filesystem import _split_by_size_and_type
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future

from .._s3client import S3Client
from .._s3client.s3client_config import S3ClientConfig
from .._s3dataset_common import parse_s3_uri

log = logging.getLogger(__name__)


@dataclass
class _S3StoragePrefix:
    prefix: str


DEFAULT_SUFFIX: str = ".ckpt"


class S3StorageWriter(StorageWriter):

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        # TODO: add implementation
        pass

    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:  # type: ignore
        # TODO: add implementation
        pass

    # TODO: Consider adding optional endpoint for feature parity

    def __init__(
        self,
        region: str,
        s3_uri: str,
        *,
        s3client_config: Optional[S3ClientConfig] = None,
        single_file_per_rank: bool = True,
        thread_count: int = 1,
    ):
        super().__init__()
        # TODO: Add support for multiple files per rank
        if not single_file_per_rank:
            raise ValueError("Multiple files per rank not supported yet.")

        self.region = region
        self.base_uri = s3_uri
        self.bucket, self.prefix = parse_s3_uri(s3_uri)
        self._clients = [
            S3Client(self.region, s3client_config=s3client_config)
            for _ in range(thread_count)
        ]
        self.single_file_per_rank = single_file_per_rank
        self.is_coordinator = False
        self.thread_count = thread_count

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        self.is_coordinator = is_coordinator

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        # S3 path is assumed to exist, no need to create bucket/prefix
        return plan

    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        if self.is_coordinator:
            new_plans = [
                dataclasses.replace(plan, storage_data=_S3StoragePrefix(f"__{i}_"))
                for i, plan in enumerate(plans)
            ]
            return new_plans
        return plans

    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[List[WriteResult]]:

        storage_plan: _S3StoragePrefix = plan.storage_data
        obj_count = 0

        def gen_object_key():
            nonlocal obj_count
            object_key = (
                f"{self.prefix}{storage_plan.prefix}{obj_count}{DEFAULT_SUFFIX}"
            )
            obj_count += 1
            return object_key

        objects_queue: queue.Queue = queue.Queue()

        if self.single_file_per_rank:
            # TODO: This reuses the FileSystem split mechanism
            for write_item in _split_by_size_and_type(self.thread_count, plan.items):
                object_key = gen_object_key()
                # s3_uri = f"{self.base_uri}{object_key}"
                # objects_queue.put((s3_uri, object_key, write_item))
                objects_queue.put((object_key, write_item))
        else:
            # TODO: Test this
            for plan_item in plan.items:
                object_key = gen_object_key()
                # s3_uri = f"{self.base_uri}/{object_key}"
                # objects_queue.put((s3_uri, object_key, [plan_item]))
                objects_queue.put((object_key, [plan_item]))

        results_queue: queue.Queue = queue.Queue()
        threads = []
        for rank in range(self.thread_count):
            t = threading.Thread(
                target=self._write_to_s3, args=(objects_queue, results_queue, rank)
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        res = []
        while not results_queue.empty():
            res.append(results_queue.get())

        futures: Future[List[WriteResult]] = Future()
        futures.set_result(res)
        return futures

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]):
        if self.is_coordinator:
            # Save metadata from coordinator node
            s3_storage_metadata: Dict[Union[str, MetadataIndex], Union[Any, str]] = {}
            for wr_list in results:
                s3_storage_metadata.update(
                    {wr.index: wr.storage_data for wr in wr_list}
                )

            s3_storage_metadata["bucket"] = self.bucket
            s3_storage_metadata["prefix"] = self.prefix
            s3_storage_metadata["single_file_per_rank"] = self.single_file_per_rank
            s3_storage_metadata["thread_count"] = self.thread_count

            metadata.storage_data = s3_storage_metadata
            metadata_key = ".metadata"

            serialized_metadata = pickle.dumps(metadata)
            with self._clients[0].put_object(self.bucket, metadata_key) as s3_writer:
                s3_writer.write(serialized_metadata)

    def _write_to_s3(self, objects_queue, results_queue, rank):
        while not objects_queue.empty():
            object_key, write_items = objects_queue.get()
            serialized = pickle.dumps(write_items)
            log.info(f"Writing checkpoint to object {object_key}")
            s3_writer = self._clients[rank].put_object(self.bucket, object_key)
            result = WriteResult(
                index=rank,
                size_in_bytes=s3_writer.write(serialized),
                storage_data=object_key,
            )
            results_queue.put(result)
