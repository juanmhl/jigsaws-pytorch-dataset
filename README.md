# Install instructions

1. Install Pytorch by following [this instructions](https://pytorch.org/get-started/locally/). For example, for Linux with CUDA 11.8, run:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
You can substitute `pip3` with `uv pip` if using `uv` virtual environment manager.
2. Install the wheel file with the following command:
```bash
pip3 install <path_to_wheel_file>
```
The wheel file can be found in GitHub releases section.

# Instructions for developers

Instead of installing the package via the wheel file, you can also install it in "editable" mode for development purposes. To do so, clone the repository and run the following command in the root directory of the repository:
```bash
pip3 install -e .
```
This will install the package in editable mode, allowing you to make changes to the source code and have them reflected immediately without needing to reinstall the package.

If you are using `uv` run:
```bash
uv pip install -e .
```

After this you can keep adding dependencies by running the `uv add` command, and they will be automatically added to the `pyproject.toml` file. That is compatible with the installation of the package using the "editable" mode.

To build the wheel file after making changes, run:
```bash
python3 -m build
```
This will create a new wheel file in the `dist` directory.

