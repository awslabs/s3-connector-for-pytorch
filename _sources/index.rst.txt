Welcome to Amazon S3 Connector for PyTorch's documentation!
===========================================================
The Amazon S3 Connector for PyTorch delivers high throughput for PyTorch training jobs that access or store data in Amazon S3. Using the S3 Connector for PyTorch
automatically optimizes performance when downloading training data from and writing checkpoints to Amazon S3, eliminating the need to write your own code to list S3 buckets and manage concurrent requests.


Amazon S3 Connector for PyTorch provides implementations of PyTorch's dataset primitives that you can use to load training data from Amazon S3.
It supports both map-style datasets for random data access patterns and iterable-style datasets for streaming sequential data access patterns.
The S3 Connector for PyTorch also includes a checkpointing interface to save and load checkpoints directly to Amazon S3, without first saving to local storage.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
