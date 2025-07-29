#!/bin/bash
cd external

# Install lumos submodule and its dependencies
cd lumos
git submodule update --init --recursive
sh install.sh

cd ..

# Install calvin_env submodule and its dependencies
cd calvin_env
git submodule update --init --recursive
pip install --no-cache-dir -e .

cd ../..

# Install the main package
pip install --no-cache-dir -e .
