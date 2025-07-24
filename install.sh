cd lumos
git submodule update --init --recursive
sh install.sh
cd ..
pip install --no-cache-dir -e .
