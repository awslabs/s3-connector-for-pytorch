# DEVELOPMENT

To develop `s3torchconnector`, you need to have Python, `pip` and `python-venv` installed. 

`s3torchconnector` uses `s3torchconnectorclient` as the underlying S3 Connector. `s3torchconnectorclient` is a Python wrapper around MountpointS3Client that uses S3 CRT to optimize performance
of S3 read/write
.
Since MountpointS3Client is implemented in Rust, for development and building from source, you will need to install `clang`, `cmake` and rust compiler (as detailed below). 

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


When you make changes to the Rust code, you need to run `pip install -e s3torchconnectorclient` before changes will be viewable from 
Python.


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
mypy --strict s3torchconnector/src
mypy --strict s3torchconnectorclient/python/src
```
 to lint.

### Debugging

Either a Python or GDB style debugger will be useful here.

To use a GDB debugger from Rust, just run the Rust test in question with the debugger enabled.

To use a GDB debugger from Python, you need to create a 'Custom Build Application'. 
Fill in the path of the Python executable in your virtual environment (`venv/bin/python`) and fill in the script name 
as the program argument.
Then put a breakpoint in the Rust/C code and try running it.

