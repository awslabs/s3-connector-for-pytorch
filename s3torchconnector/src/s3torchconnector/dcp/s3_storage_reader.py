#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import pickle
from typing import List, Optional

from torch.distributed.checkpoint import StorageReader, Metadata, LoadPlan, LoadPlanner
from torch.futures import Future

from .._s3client import S3Client
from .._s3client.s3client_config import S3ClientConfig
from .._s3dataset_common import parse_s3_uri


class S3StorageReader(StorageReader):
    def __init__(
        self,
        region: str,
        s3_uri: str,
        *,
        s3client_config: Optional[S3ClientConfig] = None,
        single_file_per_rank: bool = True,
        thread_count: int = 1
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

    def read_metadata(self) -> Metadata:
        metadata_key = ".metadata"
        object_content = self._clients[0].get_object(self.bucket, metadata_key)
        metadata = pickle.load(object_content)
        return metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        self.is_coordinator = is_coordinator
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        return plans

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # TODO: Check expected bucket, prefix etc. in metadata
        pass
