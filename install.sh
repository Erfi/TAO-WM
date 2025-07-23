pip install setuptools==57.5.0
cd lumos
git submodule update --init --recursive
sh install.sh
cd ..
pip install --no-cache-dir -e .
