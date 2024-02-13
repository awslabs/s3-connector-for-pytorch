#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import abc
import io
import os
import tarfile
import time
from multiprocessing import Queue
from threading import Thread, Lock, Barrier
from typing import Any, Callable, Iterator, TypeVar, NewType

import boto3
import click
import numpy as np
import prefixed as pr
import yaml
from PIL import Image


class DataGenerator(abc.ABC):
    @abc.abstractmethod
    def create(self, idx_gen: Iterator[int]) -> Iterator[tuple[str, io.BytesIO]]:
        pass


class ImageGenerator(DataGenerator):
    def __init__(self, width: int, height: int, img_format: str = "JPEG"):
        self.width = width
        self.height = height
        self.img_format = img_format

    def create(self, idx_iter: Iterator[int]) -> Iterator[tuple[str, io.BytesIO]]:
        for index in idx_iter:
            img = Image.fromarray(
                np.random.randint(
                    0, high=256, size=(self.height, self.width, 3), dtype=np.uint8
                )
            )
            byte_buf = io.BytesIO()
            img.save(byte_buf, format=self.img_format)
            byte_buf.seek(0)

            yield f"image_{index}.{self.img_format}", byte_buf


class ThreadSafeCounter:
    def __init__(self) -> None:
        self.val = 0
        self.lock = Lock()

    def get_and_inc(self):
        with self.lock:
            self.val += 1
            return self.val


class Utils:
    T = TypeVar("T")

    @staticmethod
    def batcher(
            items: Iterator[T], size_extractor: Callable[[T], int], size_threshold: float
    ) -> Iterator[list[T]]:
        group = []
        total = 0
        for item in items:
            item_size = size_extractor(item)
            new_total = total + item_size
            if new_total > size_threshold:
                yield group
                group = [item]
                total = item_size
            else:
                group.append(item)
                total = new_total
        if len(group) > 0:
            yield group

    @staticmethod
    def get_sample_size(item: tuple[str, io.BytesIO]) -> int:
        nbytes = item[1].getbuffer().nbytes
        return nbytes

    @staticmethod
    def tar_samples(
            samples: list[tuple[str, io.BytesIO]], ctr: ThreadSafeCounter
    ) -> tuple[str, io.BytesIO]:
        tar_fileobj = io.BytesIO()
        with tarfile.open(fileobj=tar_fileobj, mode="w|") as tar:
            for name, sample in samples:
                tf = tarfile.TarInfo(name=name)
                tf.mtime = time.time()
                tf.size = sample.getbuffer().nbytes

                tar.addfile(tf, sample)
        tar_fileobj.seek(0)

        return f"shard_{ctr.get_and_inc()}.tar", tar_fileobj

    @staticmethod
    def parse_resolution(
            ctx: click.Context, param: str, value: str
    ) -> tuple[int, int] | None:
        if value is not None:
            width, _, height = value.replace(" ", "").lower().partition("x")
            return int(width), int(height)

    @staticmethod
    def upload_to_s3(region: str, data: io.BytesIO, bucket: str, key: str):
        click.echo(f"Uploading to {key=}")
        s3_client = boto3.client("s3", region_name=region)
        s3_client.upload_fileobj(data, bucket, key)

    @staticmethod
    def parse_human_readable_bytes(
            ctx: click.Context, param: str, value: str
    ) -> float | None:
        if value is not None:
            return pr.Float(value.rstrip("b").rstrip("B"))

    @classmethod
    def write_dataset_config(cls, disambiguator: str, dataset_cfg: dict[str, Any]):
        cfg_path = f"./configuration/dataset/{disambiguator}.yaml"
        with open(cfg_path, "w") as outfile:
            yaml.dump(dataset_cfg, outfile, default_flow_style=False)
            click.echo(f"Dataset Configuration created at: {cfg_path}")


Sentinel = NewType("Sentinel", None)


class ThreadSafeIterator:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def build_pipeline(num_samples: float, resolution: tuple[int, int],
                   shard_size: float) -> Iterator[tuple[str, io.BytesIO]]:
    # define the pipeline to generate the dataset
    # TODO: parameterize the data generator to allow for creating other kinds of datasets(eg: text).
    sample_generator = ImageGenerator(*resolution)
    pipeline: Iterator[tuple[str, io.BytesIO]] = sample_generator.create(
        i for i in range(int(num_samples))
    )
    if shard_size:
        monotonic_ctr = ThreadSafeCounter()
        pipeline: Iterator[list[tuple[str, io.BytesIO]]] = Utils.batcher(
            items=pipeline,
            size_extractor=lambda item: item[1].getbuffer().nbytes,
            size_threshold=shard_size,
        )
        pipeline: Iterator[tuple[str, io.BytesIO]] = (
            Utils.tar_samples(batch, monotonic_ctr) for batch in pipeline
        )

    return ThreadSafeIterator(pipeline)


def build_producers(num_workers: int, queue: Queue,
                    dataset_generator: Iterator[tuple[str, io.BytesIO]]) -> list[Thread]:
    barrier = Barrier(num_workers)
    out = []
    for i in range(num_workers):
        worker = Thread(target=producer, name=f"DataGenerator-{i}",
                        kwargs={"identifier": i, "queue": queue, "generator": dataset_generator,
                                "barrier": barrier})
        out.append(worker)

    return out


def build_consumers(num_workers: int, queue: Queue, disambiguator: str, region: str, s3_bucket: str) -> list[Thread]:
    return [
        Thread(
            target=consumer,
            name=f"Uploader-{i}",
            kwargs={
                "identifier": i,
                "queue": queue,
                "activity": lambda label, data: Utils.upload_to_s3(
                    region=region,
                    data=data,
                    bucket=s3_bucket,
                    key=f"{disambiguator}/{label}",
                ),
            },
        )
        for i in range(num_workers)
    ]


def producer(generator: Iterator, barrier: Barrier, queue: Queue, identifier: int):
    for item in generator:
        queue.put(item)
    # wait for all producers to finish
    barrier.wait()
    # signal that there are no further items
    if identifier == 0:
        queue.put(Sentinel)


def consumer(
        queue: Queue, activity: Callable[[tuple[str, io.BytesIO]], None], identifier: int
):
    while True:
        # click.echo(f"Consumer running on thread {threading.current_thread().ident}")
        item: tuple[str, io.BytesIO] | Sentinel = queue.get()
        if item is Sentinel:
            # add signal back for other consumers
            queue.put(item)
            break
        activity(*item)


@click.command(context_settings={"show_default": True})
@click.option(
    "-n",
    "--num-samples",
    callback=lambda p1, p2, val: pr.Float(val),
    default="1k",
    help="Number of sample_generator to generate.  Can be supplied as an IEC or SI prefix. Eg: 1k, 2M."
         " Note: these are case-sensitive notations.",
)
@click.option(
    "--resolution",
    callback=Utils.parse_resolution,
    default="496x387",
    help="Resolution written in 'widthxheight' format",
)
@click.option(
    "--shard-size",
    callback=Utils.parse_human_readable_bytes,
    help="If supplied, the images are grouped into tar files of the given size."
         " Size can be supplied as an IEC or SI prefix. Eg: 16Mib, 4Kb, 1Gib."
         " Note: these are case-sensitive notations.",
)
@click.option(
    "--s3-bucket",
    type=str,
    help="S3 Bucket name. Note: Ensure the credentials are made available either through environment"
         " variables or a shared credentials file.",
    required=True,
)
@click.option(
    "--s3-prefix",
    type=str,
    help="Optional S3 Key prefix where the dataset will be uploaded. "
         "Note: a prefix will be autogenerated. eg: s3://<BUCKET>/1k_256x256_16Mib_sharded/",
)
@click.option(
    "--region",
    default="us-east-1",
    type=str,
    help="Region where the S3 bucket is hosted.",
)
def synthesize_dataset(
        num_samples: float,
        resolution: tuple[int, int],
        shard_size: float,
        s3_bucket: str,
        s3_prefix: str,
        region: str,
):
    """
    Synthesizes a dataset that will be used for benchmarking and uploads it to an S3 bucket.
    """
    num_workers = os.cpu_count()
    Q = Queue(num_workers)

    # setup upstream stage to generate the dataset in memory
    pipeline = build_pipeline(num_samples=num_samples, resolution=resolution, shard_size=shard_size)
    producers = build_producers(num_workers=num_workers, queue=Q, dataset_generator=pipeline)

    # setup downstream stage to upload generated datasets to s3
    disambiguator = s3_prefix or f"{num_samples:.0h}_{resolution[0]}x{resolution[1]}_images"
    if shard_size:
        disambiguator = disambiguator + f"_{shard_size:.0h}b_shards"
    consumers = build_consumers(num_workers=num_workers, queue=Q, disambiguator=disambiguator, region=region,
                                s3_bucket=s3_bucket)

    # kick off consumers and producers
    for worker in [*consumers, *producers]:
        worker.start()
    # wait for all threads to finish. Note: order is important since we wait to drain all pending messages from
    # producers first.
    for worker in [*producers, *consumers]:
        worker.join()

    fq_key = f"s3://{s3_bucket}/{disambiguator}/"
    click.echo(f"Dataset uploaded to: {fq_key}")
    # generate hydra dataset config file
    Utils.write_dataset_config(
        disambiguator=disambiguator,
        dataset_cfg={
            "prefix_uri": fq_key,
            "region": region,
            "sharding": bool(shard_size),
        })

    click.echo(
        f"Configure your experiment by setting the entry:\n\tdataset: {disambiguator}"
    )
    click.echo(
        "Alternatively, you can run specify it on the cmd-line when running the benchmark like so:"
    )
    click.echo(f"\tpython benchmark.py -m -cn <CONFIG-NAME> 'dataset={disambiguator}'")


if __name__ == "__main__":
    synthesize_dataset()
