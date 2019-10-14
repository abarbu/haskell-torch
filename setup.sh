#!/bin/bash

git submodule update --init

conda init bash

eval "$(conda shell.bash hook)"

if conda activate haskell-torch ; then
    conda env update -n haskell-torch --file environment.yml
else
    conda env create -f environment.yml
fi

if ! conda activate haskell-torch ; then
    echo Cannot activate the haskell-torch environment >&2
    exit 1
fi

mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
echo "export OLD_LD_PRELOAD=\$LD_PRELOAD" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export LD_PRELOAD=\$LD_PRELOAD:$CONDA_PREFIX/lib/libtinfo.so:$CONDA_PREFIX/lib/libtinfow.so:$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CONDA_PREFIX/lib" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
chmod a+x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d/
echo 'export LD_PRELOAD=$OLD_LD_PRELOAD' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'unset OLD_LD_PRELOAD' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=\$OLD_LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo "unset OLD_LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
chmod a+x $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# Needed to reload the environment
conda deactivate
conda activate haskell-torch

python generate-config.py || { echo 'Failed to create configuration (stack.yaml and config.yaml) files' ; exit 1; } 

echo "======================================================================"
echo "                  Haskell-Torch is set up!"
echo " Check above to see if you have CUDA support"
echo " If you want to regenerate the bindings against a new PyTorch, run make"
echo " Next up activate the conda environment and build the code with:"
echo "    conda activate haskell-torch && stack build"
