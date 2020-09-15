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

conda init bash

eval "$(conda shell.bash hook)"

if conda activate $CONDA_ENV ; then
    if [ $WITH_JUPYTER = "YES" ]; then
	if [ $WITH_CUDA = "YES" ]; then
            conda env update -n $CONDA_ENV --file environment-with-jupyter.yml
	else
            conda env update -n $CONDA_ENV --file environment-with-jupyter-cpu-only.yml
	fi
    else
	if [ $WITH_CUDA = "YES" ]; then
            conda env update -n $CONDA_ENV --file environment.yml
	else
            conda env update -n $CONDA_ENV --file environment-cpu-only.yml
	fi
    fi
else
    if [ $WITH_JUPYTER = "YES" ]; then
	if [ $WITH_CUDA = "YES" ]; then
            conda env create -f environment-with-jupyter.yml
	else
            conda env create -f environment-with-jupyter-cpu-only.yml
	fi
    else
	if [ $WITH_CUDA = "YES" ]; then
            conda env create -f environment.yml
	else
            conda env create -f environment.yml
	fi
    fi
fi

if ! conda activate $CONDA_ENV ; then
    echo Cannot activate the $CONDA_ENV environment >&2
    exit 1
fi

mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
echo "export OLD_LD_PRELOAD=\$LD_PRELOAD" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export LD_PRELOAD=\$LD_PRELOAD:$CONDA_PREFIX/lib/libtinfo.so:$CONDA_PREFIX/lib/libtinfow.so:$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so:$CONDA_PREFIX/lib/libmkl_intel_lp64.so" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.6/site-packages/torch/lib/" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
chmod a+x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d/
echo 'export LD_PRELOAD=$OLD_LD_PRELOAD' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'unset OLD_LD_PRELOAD' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=\$OLD_LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo "unset OLD_LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
chmod a+x $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
