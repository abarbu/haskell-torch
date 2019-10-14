{-# LANGUAGE CPP, FlexibleContexts, QuasiQuotes, ScopedTypeVariables, TemplateHaskell #-}

module Torch.C.CUDA where
import           Data.Monoid           ((<>))
import           Foreign.C.Types
import qualified Language.C.Inline     as C
import qualified Language.C.Inline.Cpp as C
import           Torch.C.Types

C.context (C.cppCtx <> tensorCtx)

#if WITH_CUDA
C.include "<torch/csrc/autograd/generated/VariableType.h>"
C.include "<ATen/cuda/CUDAContext.h>"

C.using "namespace at"
C.using "namespace torch::autograd"

hasCUDA :: IO CBool
hasCUDA = [C.exp|bool{at::cuda::is_available()}|]

currentDevice :: IO CInt
currentDevice = [C.exp|int{at::cuda::current_device()}|]

deviceCount :: IO CInt
deviceCount = [C.exp|int{at::cuda::device_count()}|]
#else
hasCUDA :: IO CBool
hasCUDA = [C.exp|bool{0}|]

currentDevice :: IO CInt
currentDevice = [C.exp|int{0}|]

deviceCount :: IO CInt
deviceCount = [C.exp|int{0}|]
#endif
