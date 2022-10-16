{-# LANGUAGE CPP, QuasiQuotes, TemplateHaskell #-}

module Torch.C.Generator where
import           Data.Monoid           ((<>))
import           Data.Word
import           Foreign.Ptr
import qualified Language.C.Inline     as C
import qualified Language.C.Inline.Cpp as C
import           Torch.C.Types

C.context (tensorCtx <> C.funCtx)

C.include "<torch/csrc/autograd/variable.h>"

C.using "namespace at"

seed :: Ptr CGenerator -> IO Word64
seed g = [C.exp|uint64_t { $(Generator *g)->seed() }|]

initialSeed :: IO Word64
initialSeed = [C.exp|uint64_t { at::detail::getDefaultCPUGenerator().current_seed() }|]

setSeed :: Ptr CGenerator -> Word64 -> IO ()
setSeed g s = [C.exp|void { $(Generator *g)->set_current_seed($(uint64_t s)) }|]

cpuGenerator :: IO (Ptr CGenerator)
cpuGenerator = [C.exp|const Generator *{ &at::globalContext().defaultGenerator(kCPU) }|]

#if WITH_CUDA
cudaGenerator :: IO (Ptr CGenerator)
cudaGenerator = [C.exp|const Generator *{ &at::globalContext().defaultGenerator(kCUDA) }|]
#endif
