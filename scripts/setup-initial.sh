#!/bin/bash

export WITH_JUPYTER=NO
export WITH_CUDA=IF_PRESENT
export CONDA_ENV=haskell-torch
export QUICK_GHC=NO
export FAST=

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "$(basename $0) - set up HaskellTorch"
      echo "  by default CUDA will be used if present and jupyter will be installed"
      echo " "
      echo "$(basename $0) [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "--with-jupyter            install jupyter"
      echo "--without-jupyter         don't install jupyter"
      echo "--with-cuda               install CUDA"
      echo "--without-cuda            don't install CUDA"
      echo "--in-conda-base           install in base conda, not haskell-torch"
      echo "--quick-ghc               fast ghc builds, but the compiler will be rather slow"
      echo "--fast                    like stack --fast disables optimizations for app code"
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

echo "======================================================================"
echo "Configuring with"
echo " WITH_JUPYTER=$WITH_JUPYTER"
echo " WITH_CUDA=$WITH_CUDA"
echo " QUICK_GHC=$QUICK_GHC"
echo " FAST=$FAST"
echo "======================================================================"

git submodule update --init
python generate-config.py "${args[@]}" || { echo 'Failed to create configuration (stack.yaml and config.yaml) files' ; exit 1; }
