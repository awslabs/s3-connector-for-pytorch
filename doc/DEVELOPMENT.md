# DEVELOPMENT

To develop `s3torchconnector`, you need to have Python, `pip` and `python-venv` installed. 

`s3torchconnector` uses `s3torchconnectorclient` as the underlying S3 Connector. `s3torchconnectorclient` is a 
Python wrapper around MountpointS3Client that uses S3 CRT to optimize performance of S3 read/write
.
Since MountpointS3Client is implemented in Rust, for development and building from source, you will need to install 
`clang`, `cmake` and rust compiler (as detailed below). 

Note: CLI commands for Ubuntu/Debian 
#### Install Python 3.x and pip
```shell
sudo apt update
sudo apt install python3
sudo apt install python3-pip
```
#### Clone project
```shell
  git clone git@github.com:awslabs/s3-connector-for-pytorch.git
```
#### Create  a Python virtual environment
```shell
  cd /path/to/your/project
  python3 -m venv virtual-env
  source virtual-env/bin/activate
```
#### Install clang (needed to build the client)
```shell
  sudo apt install clang
```
#### Install cmake (needed to build the client)
```shell
  sudo apt install cmake
```
#### Install Rust compiler (needed to build the client)
```shell
  curl https://sh.rustup.rs -sSf | sh
  source "$HOME/.cargo/env"
```
#### Install project modules in editable mode
```shell
  pip install -e s3torchconnectorclient
  pip install -e s3torchconnector
```


When you make changes to the Rust code, you need to run `pip install -e s3torchconnectorclient` before changes will 
be viewable from Python.


### Licensing
When developing, ensure to create license headers at the top of each file. This can be automated with Pycharm/Clion 
with the following configuration:

Go to the settings, and find the 'Copyright profiles' section. Create a new one with the following text:

> Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
> 
> // SPDX-License-Identifier: BSD

Then under the 'Copyright' section, create a new scope covering 'all', and assign your new copyright profile.

### Making a commit

Our CI uses `clippy` to lint Rust code changes. Use `cargo clippy --all-targets --all-features` to lint Rust before
pushing new Rust commits.

For Python code changes, run 
```bash
black --verbose .
flake8 s3torchconnector/ --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 s3torchconnectorclient/python --count --select=E9,F63,F7,F82 --show-source --statistics
mypy s3torchconnector/src
mypy s3torchconnectorclient/python/src
```
 to lint.

To run mypy without `lightning` installed, run
```bash
mypy s3torchconnector/src --exclude s3torchconnector/src/s3torchconnector/lightning
mypy s3torchconnectorclient/python/src
```

### Debugging

Either a Python or GDB style debugger will be useful here.

To use a GDB debugger from Rust, just run the Rust test in question with the debugger enabled.

To use a GDB debugger from Python, you need to create a 'Custom Build Application'. 
Fill in the path of the Python executable in your virtual environment (`venv/bin/python`) and fill in the script name 
as the program argument.
Then put a breakpoint in the Rust/C code and try running it.

#### Enabling Debug Logging
The [Python logger](https://docs.python.org/3/library/logging.html) handles logging messages from the Python-side 
of our implementation.
For debug purposes, you can also enable the logs for our Rust components, which are off by default. 
These are handled by [tracing_subscriber](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/) and can be 
configured through the following environment variables:
- `S3_TORCH_CONNECTOR_DEBUG_LOGS` - Configured similarly to the
[RUST_LOG](https://docs.rs/env_logger/latest/env_logger/#enabling-logging) variable for
filtering logs from our Rust components. This includes finer granularity logs from 
[AWS Common Runtime (CRT)](https://docs.aws.amazon.com/sdkref/latest/guide/common-runtime.html).
**Please note that the AWS CRT logs are very noisy. We recommend to filter them out by appending `"awscrt=off"` to
your S3_TORCH_CONNECTOR_DEBUG_LOGS setup.**
- `S3_TORCH_CONNECTOR_LOGS_DIR_PATH` - The path to a local directory where you have write permissions. 
When configured, the logs from the Rust components will be appended to a file at this location. 
This will result in a log file located at 
`${S3_TORCH_CONNECTOR_LOGS_DIR_PATH}/s3torchconnectorclient.log.yyyy-MM-dd-HH`, rolled on an hourly basis. 
The log messages of the latest run are appended to the end of the most recent log file.

**Examples**
- Configure INFO level logs to be written to STDOUT:
```sh
  export S3_TORCH_CONNECTOR_DEBUG_LOGS=info
```

- Enable TRACE level logs (most verbose) to be written at `/tmp/s3torchconnector-logs`:
```sh
  export S3_TORCH_CONNECTOR_DEBUG_LOGS=trace
  export S3_TORCH_CONNECTOR_LOGS_DIR_PATH="/tmp/s3torchconnector-logs"
```
After running your script, you will find the logs under `/tmp/s3torchconnector-logs`.
The file will include AWS CRT logs. 

- Enable TRACE level logs with AWS CRT logs filtered out, written at `/tmp/s3torchconnector-logs`:
```sh
  export S3_TORCH_CONNECTOR_DEBUG_LOGS=trace,awscrt=off
  export S3_TORCH_CONNECTOR_LOGS_DIR_PATH="/tmp/s3torchconnector-logs"
```

- Set up different levels for inner components:
```sh
  export S3_TORCH_CONNECTOR_DEBUG_LOGS=trace,mountpoint_s3_client=debug,awscrt=error
```
This will set the log level to TRACE by default, DEBUG for mountpoint-s3-client and ERROR for AWS CRT.

For more examples please check the
[env_logger documentation](https://docs.rs/env_logger/latest/env_logger/#enabling-logging).

### Fine Tuning
Using S3ClientConfig you can set up the following parameters for the underlying S3 client: 
* `throughput_target_gbps(float)`: Throughput target in Gigabits per second (Gbps) that we are trying to reach.
  **10.0 Gbps** by default (may change in future).

* `part_size(int)`: Size (bytes) of file parts that will be uploaded/downloaded.
  Note: for saving checkpoints, the inner client will adjust the part size to meet the service limits.
  (max number of parts per upload is 10,000, minimum upload part size is 5 MiB).
  Part size must have **values between 5MiB and 5GiB.** Is set by default to **8MiB** (may change in future).

* `unsigned(bool)`: Allows the usage of unsigned clients when accessing public datasets or when other mechanisms are
  in place to grant access.

For example this can be passed in like: 
```py
from s3torchconnector import S3MapDataset, S3ClientConfig

# Setup for DATASET_URI and REGION.
...
# Setting part_size to 5 MiB and throughput_target_gbps to 15 Gbps.
config = S3ClientConfig(part_size=5 * 1024 * 1024, throughput_target_gbps=15)
# Passing this on to an S3MapDataset.
s3_map_dataset = S3MapDataset.from_prefix(DATASET_URI, region=REGION, s3client_config=config)
# Updating the configuration for checkpoints.
# Please note that you can also pass in a different configuration to checkpoints.
s3_checkpoint = S3Checkpoint(region=REGION, s3client_config=config)
# Works similarly for Lightning checkpoints.
s3_lightning_checkpoint = S3LightningCheckpoint(region=REGION, s3client_config=config)

# Use an unsigned S3 client
s3_client = S3Client(region=REGION, s3client_config=S3ClientConfig(unsigned=True))
```

**When modifying the default values for these flags, we strongly recommend to run benchmarking to ensure you are not
introducing a performance regression.**
