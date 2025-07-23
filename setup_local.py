"""Setup tao-wm installation for SLURM environments."""

from os import path as op
import re

from setuptools import find_packages, setup


def _read(f):
    return open(op.join(op.dirname(__file__), f)).read() if op.exists(f) else ""


_meta = _read("tao-wm/__init__.py")


def find_meta(_meta, string):
    l_match = re.search(r"^" + string + r'\s*=\s*"(.*)"', _meta, re.M)
    if l_match:
        return l_match.group(1)
    raise RuntimeError(f"Unable to find {string} string.")


# For SLURM: only install core package, assume dependencies are in conda env
meta = dict(
    name=find_meta(_meta, "__project__"),
    version=find_meta(_meta, "__version__"),
    license=find_meta(_meta, "__license__"),
    description="TAO-WM: ",
    platforms=("Any"),
    zip_safe=False,
    keywords="pytorch tao-wm".split(),
    author=find_meta(_meta, "__author__"),
    author_email=find_meta(_meta, "__email__"),
    url="https://github.com/Erfi/TAO-WM.git",
    packages=find_packages(exclude=["tests"]),
    # Note: No install_requires - dependencies should be pre-installed in SLURM conda env
)

if __name__ == "__main__":
    print("find_package", find_packages(exclude=["tests"]))
    setup(**meta)
