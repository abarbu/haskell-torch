#!/bin/bash

export WITH_JUPYTER=NO
export WITH_CUDA=IF_PRESENT
export CONDA_ENV=haskell-torch
export QUICK_GHC=NO
export FAST=

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "$(basename $0) - called  from setup.sh"
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

if [ $WITH_JUPYTER = "YES" ]; then
    echo "======================================================================"
    echo "We are now setting up jupyter / ihaskell"
    echo "======================================================================"

    patch --forward -p0 < ihaskell-dynamic.diff || { echo 'Failed to patch IHaskell' ; exit 1; }
    stack install ihaskell $FAST || { echo 'Failed to build IHaskell' ; exit 1; }
    stack exec -- ihaskell install --stack || { echo 'Failed to install IHaskell' ; exit 1; }
    jupyter labextension install jupyterlab-ihaskell
    # TODO This is an alternate extension, not sure
    # jupyter labextension install ihaskell_jupyterlab
    cd ihaskell/ihaskell_labextension
    npm install
    npm run build
    jupyter labextension link .
    cd -
fi
