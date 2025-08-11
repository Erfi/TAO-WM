#!/bin/bash
cd external

# initialize and update the submodules
git submodule update --init --recursive

# Install calvin (will automatically install calvin_env as its submodule)
echo "Installing calvin..."
cd calvin
git submodule update --init --recursive
sh install.sh

cd ..

# Install lumos submodule and its dependencies
echo "Installing lumos..."
cd lumos
git submodule update --init --recursive
sh install.sh

cd ../..

# Install the main package
echo "Installing main package..."
pip install --no-cache-dir -e .
