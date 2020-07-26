#!/bin/bash

export WITH_JUPYTER=NO
export WITH_CUDA=IF_PRESENT

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
    *)
      break
      ;;
  esac
done

echo "======================================================================"
echo "Configuring with"
echo " WITH_JUPYTER=$WITH_JUPYTER"
echo " WITH_CUDA=$WITH_CUDA"
echo "======================================================================"

git submodule update --init

conda init bash

eval "$(conda shell.bash hook)"

if conda activate haskell-torch ; then
    if [ $WITH_JUPYTER = "YES" ]; then
	if [ $WITH_CUDA = "YES" ]; then
            conda env update -n haskell-torch --file environment-with-jupyter.yml
	else
            conda env update -n haskell-torch --file environment-with-jupyter-cpu-only.yml
	fi
    else
	if [ $WITH_CUDA = "YES" ]; then
            conda env update -n haskell-torch --file environment.yml
	else
            conda env update -n haskell-torch --file environment-cpu-only.yml
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

if ! conda activate haskell-torch ; then
    echo Cannot activate the haskell-torch environment >&2
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

# Needed to reload the environment
conda deactivate
conda activate haskell-torch

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

stack build haskell-torch --fast || { echo "Failed to build haskell-torch! :("; exit 1; }

echo "======================================================================"
echo "We are now setting up jupyter / ihaskell"
echo "======================================================================"

if [ $WITH_JUPYTER = "YES" ]; then
    patch --forward -p0 < ihaskell-dynamic.diff || { echo 'Failed to patch IHaskell' ; exit 1; }
    stack install ihaskell --fast || { echo 'Failed to build IHaskell' ; exit 1; }
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

echo "======================================================================"
echo "                  Haskell-Torch is set up!"
echo "======================================================================"
echo
echo " Check above to see if you have CUDA support"
echo " If you want to regenerate the bindings against a new PyTorch, run make"
echo " Next up activate the conda environment. You can later build the code with:"
echo "    conda activate haskell-torch && stack build"
