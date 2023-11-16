# S3Dataset

S3Dataset is a tool which allows Python code to interface with a performant S3 client.

## Getting started
### TODO: These instructions are probably incomplete
### Build from source
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
  python3 -m venv your-env
  source your-env/bin/activate
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
```
#### Install project modules
```shell
  pip install -e s3dataset_s3_client
  pip install -e s3dataset
```

## Development workflow

When you make changes to the Rust code, you need to run `pip install -e .` before changes will be viewable from 
Python. It's probably worth creating a shell script for this and adding it as part of the pre-build step.

### Making a commit

Our CI uses `clippy` to lint Rust code changes. Use `cargo clippy --all-targets --all-features` to lint Rust before
pushing new Rust commits.

For Python code changes, run 
```bash
black --verbose python/
flake8 python/ --count --select=E9,F63,F7,F82 --show-source --statistics
```
 to lint.

## Debugging

Either a Python or GDB style debugger will be useful here.

To use a GDB debugger from Rust, just run the Rust test in question with the debugger enabled.

To use a GDB debugger from Python, you need to create a 'Custom Build Application'. 
Fill in the path of the Python executable in your virtual environment (`venv/bin/python`) and fill in the script name 
as the program arguments.
Then put a breakpoint in the Rust/C code and try running it.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the LICENSE NAME HERE License.

