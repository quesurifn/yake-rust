### Rust Setup

1. Install [Rust compiler](https://www.rust-lang.org/tools/install)
2. `rustup default stable`
3. `rustup component add rustfmt --toolchain nightly` \
    (we format with `cargo +nightly fmt`)
4. `cargo check`

### Python Setup

This applies to the `python` subdirectory where the python bindings are defined and built into a python distribution package.

1. Install Python (any released version not EOL, see [here](https://devguide.python.org/versions/)); `pip` and `venv` will be included and the following instructions assume that they are used, but you can use, e.g., [`uv`](https://github.com/astral-sh/uv) as well. You may want to upgrade as well, `pip install -U pip wheel`.
2. Go to the `python` subdirectory.
3. `python -m venv venv` to create a virtual environment and `. venv/bin/activate` to activate it (activating it is a bit different from system to system).
4. To run the CI, all you need is [tox](https://tox.wiki/en/4.24.1/); install it with `pip install tox` and run it with `tox`.
5. To build the package, all you need is [build](https://pypi.org/project/build/), `pip install build` to install and `pythom -m build .` to build it. Note that this is not needed for CI, as tox builds the package internally for each environment. This is only in case you want to build it yourself locally. For development, you can also just do an editable install, `pip install -e .`.
6. For code formatting and style, install the development requirements, `pip install -r requirements-dev.txt`. You can then apply code formatting with `ruff format`, detect and fix linting issues with `ruff check --fix`. Note that `mypy` is used for type checking, but since `mypy` does not modify the code and is run in the CI by `tox`, you may not need to run it yourself.
