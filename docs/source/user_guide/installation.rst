Installation
============

DervFlow can be installed from PyPI or built from source using `maturin`.

From PyPI (recommended)
-----------------------

Pre-built wheels are published for CPython 3.10–3.13 on Linux (x86_64, aarch64),
macOS (x86_64, arm64), and Windows (x86_64).

.. code-block:: bash

   pip install dervflow

The wheel bundles the compiled Rust extension together with the Python helpers,
so no additional compilation step is necessary.

From source
-----------

Building DervFlow yourself is useful when working on the Rust crate or when a
pre-built wheel is not available for your platform.

Prerequisites
~~~~~~~~~~~~~

* Rust 1.70 or newer (`rustup` is the easiest way to install it)
* Python 3.10–3.13 with development headers
* `maturin` (`pip install "maturin[patchelf]"` on Linux to avoid rpath warnings)

Build steps
~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/alphavelocity/dervflow.git
   cd dervflow

   # Install maturin
   pip install "maturin[patchelf]"

   # Development install (editable)
   maturin develop --release

   # Or build a wheel for redistribution
   maturin build --release
   pip install target/wheels/dervflow-*.whl

Verifying the installation
--------------------------

.. code-block:: python

   import dervflow
   print(dervflow.__version__)

   model = dervflow.BlackScholesModel()
   price = model.price(100, 100, 0.05, 0.0, 0.2, 1.0, "call")
   print(f"Sample price: {price:.2f}")

Troubleshooting
---------------

* **Import errors** – double-check that NumPy is installed and that the wheel
  matches your Python interpreter and platform.
* **Build errors** – ensure `rustup update` has been run recently and that a C
  toolchain (gcc/clang/MSVC) is available on the system.
* **Performance concerns** – prefer release builds and enable the `python`
  feature when compiling the Rust crate outside of `maturin` (`cargo build
  --features python --release`).

For community support, open an issue on the GitHub tracker or join the
repository discussions.
