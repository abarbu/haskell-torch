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

bash scripts/setup-initial.sh "$@"
bash scripts/setup-conda.sh "$@"
bash scripts/setup-haskell.sh "$@"
bash scripts/setup-jupyter.sh "$@"

echo "======================================================================"
echo "                  Haskell-Torch is set up!"
echo ""
echo "Configured with:"
echo " WITH_JUPYTER=$WITH_JUPYTER"
echo " WITH_CUDA=$WITH_CUDA"
echo " QUICK_GHC=$QUICK_GHC"
echo " FAST=$FAST"
echo "======================================================================"
echo
echo " Check above to see if you have CUDA support"
echo " If you want to regenerate the bindings against a new PyTorch, run make"
echo " Next up activate the conda environment. You can later build the code with:"
echo "    conda activate haskell-torch && stack build ${FAST}"
