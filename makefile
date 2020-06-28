TORCH_ROOT=$(shell python -c 'import torch; import inspect; import os; print(os.path.dirname(inspect.getfile(torch)))')

all: haskell-torch-cbindings/src/Torch/C/Tensor.hs

haskell-torch-cbindings/src/Torch/C/Tensor.hs: VariableType.processed
	stack build haskell-torch-tools && \
	cd haskell-torch-cbindings && \
	stack exec haskell-torch-tools-generate-ctensor src/Torch/C/Tensor.hs ../VariableType.processed
