#!/bin/bash

export WITH_JUPYTER=NO
export WITH_CUDA=IF_PRESENT
export CONDA_ENV=haskell-torch
export QUICK_GHC=NO
export FAST=

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "$(basename $0) - called from setup.sh"
      exit 0
      ;;
    --with-jupyter)
      export WITH_JUPYTER=YES
      shift
      ;;
    --without-jupyter)
      export WITH_JUPYTER=NO
      shift
      ;;
    --with-cuda)
      export WITH_CUDA=YES
      shift
      ;;
    --without-cuda)
      export WITH_CUDA=NO
      shift
      ;;
    --in-conda-base)
      export CONDA_ENV=base
      shift
      ;;
    --quick-ghc)
      export QUICK_GHC=YES
      shift
      ;;
    --fast)
      export FAST=--fast
      shift
      ;;
    *)
      break
      ;;
  esac
done

eval "$(conda shell.bash hook)"

# Needed to reload the environment
conda deactivate
conda activate $CONDA_ENV

args=()

if [ $WITH_CUDA = "YES" ]; then
    args+=(--with-cuda)
elif [ $WITH_CUDA = "NO" ]; then
    args+=(--without-cuda)
fi

if [ $WITH_JUPYTER = "YES" ]; then
    args+=(--with-jupyter)
elif [ $WITH_JUPYTER = "NO" ]; then
    args+=(--without-jupyter)
fi

python generate-config.py "${args[@]}" || { echo 'Failed to create configuration (stack.yaml and config.yaml) files' ; exit 1; }

echo "======================================================================"
echo "Building"
echo "======================================================================"

stack build haskell-torch $FAST || { echo "Failed to build haskell-torch! :("; exit 1; }
