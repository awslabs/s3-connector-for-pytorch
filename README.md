# S3Dataset

S3Dataset is a tool which allows Python code to interface with a performant S3 client.

## Building from source
### TODO: These instructions are probably incomplete.

1. Install clang (needed to build the CRT)
2. Install Rust compiler: `curl https://sh.rustup.rs -sSf | sh`
3. Create a Python virtual environment
  - This can be done either through your IDE of choice (CLion works better than Pycharm here, as Pycharm does not 
    include support for GDB level debugging)

## Development workflow

When you make changes to the Rust code, you need to run `pip install -e .` before changes will be viewable from 
Python. It's probably worth creating a shell script for this and adding it as part of the pre-build step.

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

