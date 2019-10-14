{-# LANGUAGE QuasiQuotes, TemplateHaskell #-}

module Torch.C.Scalar where
import           Data.Int
import           Data.Monoid           ((<>))
import           Foreign.C.Types
import           Foreign.Ptr
import qualified Language.C.Inline     as C
import qualified Language.C.Inline.Cpp as C
import           Torch.C.Types

C.context (C.cppCtx <> tensorCtx)

C.include "<torch/csrc/autograd/variable.h>"

C.using "namespace at"
C.using "namespace torch::autograd"

C.verbatim "extern \"C\" void delete_scalar(Scalar* o) { delete o; }"

foreign import ccall "&delete_scalar" deleteScalar :: FunPtr (Ptr CScalar -> IO ())

mkScalarCPUBool    :: CBool -> IO (Ptr CScalar)
mkScalarCPUBool x    = [C.exp|Scalar *{ new Scalar($(bool x)) }|]

mkScalarCPUByte    :: CUChar -> IO (Ptr CScalar)
mkScalarCPUByte x    = [C.exp|Scalar *{ new Scalar($(unsigned char x)) }|]

mkScalarCPUChar    :: CChar -> IO (Ptr CScalar)
mkScalarCPUChar x    = [C.exp|Scalar *{ new Scalar($(char x)) }|]

mkScalarCPUShort   :: CShort -> IO (Ptr CScalar)
mkScalarCPUShort x   = [C.exp|Scalar *{ new Scalar($(short x)) }|]

mkScalarCPUInt     :: CInt -> IO (Ptr CScalar)
mkScalarCPUInt x     = [C.exp|Scalar *{ new Scalar($(int x)) }|]

mkScalarCPULong    :: CLong -> IO (Ptr CScalar)
mkScalarCPULong x    = [C.exp|Scalar *{ new Scalar($(long x)) }|]

mkScalarCPUHalf    :: Int16 -> IO (Ptr CScalar)
mkScalarCPUHalf x    = [C.exp|Scalar *{ new Scalar($(int16_t x)) }|]

mkScalarCPUFloat   :: CFloat -> IO (Ptr CScalar)
mkScalarCPUFloat x   = [C.exp|Scalar *{ new Scalar($(float x)) }|]

mkScalarCPUDouble  :: CDouble -> IO (Ptr CScalar)
mkScalarCPUDouble x  = [C.exp|Scalar *{ new Scalar($(double x)) }|]

mkScalarCUDABool   :: CBool -> IO (Ptr CScalar)
mkScalarCUDABool x   = [C.exp|Scalar *{ new Scalar($(bool x)) }|]

mkScalarCUDAByte   :: CUChar -> IO (Ptr CScalar)
mkScalarCUDAByte x   = [C.exp|Scalar *{ new Scalar($(unsigned char x)) }|]

mkScalarCUDAChar   :: CChar -> IO (Ptr CScalar)
mkScalarCUDAChar x   = [C.exp|Scalar *{ new Scalar($(char x)) }|]

mkScalarCUDAShort  :: CShort -> IO (Ptr CScalar)
mkScalarCUDAShort x  = [C.exp|Scalar *{ new Scalar($(short x)) }|]

mkScalarCUDAInt    :: CInt -> IO (Ptr CScalar)
mkScalarCUDAInt x    = [C.exp|Scalar *{ new Scalar($(int x)) }|]

mkScalarCUDALong   :: CLong -> IO (Ptr CScalar)
mkScalarCUDALong x   = [C.exp|Scalar *{ new Scalar($(long x)) }|]

mkScalarCUDAHalf   :: Int16 -> IO (Ptr CScalar)
mkScalarCUDAHalf x   = [C.exp|Scalar *{ new Scalar($(int16_t x)) }|]

mkScalarCUDAFloat  :: CFloat -> IO (Ptr CScalar)
mkScalarCUDAFloat x  = [C.exp|Scalar *{ new Scalar($(float x)) }|]

mkScalarCUDADouble :: CDouble -> IO (Ptr CScalar)
mkScalarCUDADouble x = [C.exp|Scalar *{ new Scalar($(double x)) }|]
