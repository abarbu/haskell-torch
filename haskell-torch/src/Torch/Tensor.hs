{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, FlexibleContexts, FlexibleInstances, GADTs, MultiParamTypeClasses #-}
{-# LANGUAGE MultiWayIf, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, RankNTypes, RecordWildCards          #-}
{-# LANGUAGE ScopedTypeVariables, TypeApplications, TypeFamilies, TypeInType, TypeOperators, UndecidableInstances   #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -pgmP cc -optP -E -optP -undef -optP -std=c89 #-}

-- | The main module to create and manipulate tensors. Most inplace operations
-- are in @Tensor.Inplace@
--
-- Arguments follow some conventions to allow for shape inference and building up
-- networks with `>=>`:
--   - Arguments that affect the shape are promoted to types and one should use
--     type application to bind them. These always appear before any other
--     types. Some functionality is split into multiple functions when the shape
--     polymorphism makes no sense.
--   - When multiple such arguments can be provided
--     or when they are not optional (or the operation would be useless if they were)
--     these type-level arguments also appear at the value level.
--     For example as in: @sized (size_ @'[1,sz1]) tensor@
--     The size_ function is a synonym for Size which is an empty datatype
--     like Proxy. It exists only as a carrier for the size and as a way to get
--     named and required arguments at the type level.
--   - Weights come after all other arguments but before the input. If more than
--     one is needed they are passed in through a single argument as a tuple.
--   - Inputs are always the last argument.
-- The last two rules combine together to make it convenient to pipe arguments
-- through networks using @>=>@ Check out the tutorials and examples for more.
--
-- When adding any functions to this module you must always keep storage
-- lifetime in mind: if a tensor shares its storage with another tensor, you
-- must keep a reference to that other tensor alive! This is because tensor
-- storage cane come from Haskell, not just from PyTorch. PyTorch itself will
-- reference count and ensure that tensors that share storage don't delete live
-- storage but when convering a vector to a tensor you can do so without
-- copying. Without the ability to avoid this copy Haskell-Torch would be too
-- slow to be practical because data loaders could never be efficient (we tried,
-- this isn't a theoretical concern). This means that the underlying storage is
-- owned by Haskell and will be freed unless a reference to it continues to
-- exist. This extra reference to storage is the 2nd argument to Tensor. That
-- reference should never be used for anything else!

module Torch.Tensor where
import           Control.Monad
import           Control.Monad.Extra
import qualified Data.ByteString              as BS
import qualified Data.ByteString.Internal     as BS (fromForeignPtr, toForeignPtr)
import           Data.Coerce
import           Data.IORef
import           Data.List                    (zipWith4)
import qualified Data.List
import           Data.Maybe
import           Data.Singletons
import           Data.Singletons.Prelude      as SP
import           Data.Singletons.Prelude.List
import           Data.Singletons.TH
import           Data.Singletons.TypeLits
import           Data.String.ShowIO
import           Data.Text                    (Text)
import qualified Data.Text                    as T
import qualified Data.Text.IO                 as T
import           Data.Type.Equality           hiding (type (==), apply)
import qualified Data.Vector                  as V'
import qualified Data.Vector                  as VB
import           Data.Vector.Storable         (Vector)
import qualified Data.Vector.Storable         as V
import           Data.Word
import           Foreign.C.Types
import           Foreign.ForeignPtr
import           Foreign.ForeignPtr.Unsafe
import           Foreign.Marshal.Alloc
import           Foreign.Marshal.Array
import           Foreign.Ptr
import           Foreign.Storable
import           GHC.Int
import           GHC.TypeLits                 as TL hiding (type (+), type (-))
import           Numeric
import           Prelude                      as P
import           System.IO.Unsafe
import qualified Torch.C.CUDA                 as C
import qualified Torch.C.Generator            as C
import qualified Torch.C.Tensor               as C
import qualified Torch.C.Types                as C
import qualified Torch.C.Variable             as CV
import           Torch.Misc
import           Torch.Types
import           Unsafe.Coerce

-- TODO This is internal
wrapTensorM :: forall ty ki sz. (TensorConstraints ty ki sz)
  => IO (ForeignPtr C.CTensor) -> Maybe (ForeignPtr ()) -> IO (Tensor ty ki sz)
wrapTensorM op store = do
  ten <- op
  pure $ Tensor ten store

-- | Internal. A handy function to create new tensors we'll be writing to
-- shortly
-- | TODO Don't export this
writeToNewTensor :: forall ty ki sz. (TensorConstraints ty ki sz)
  => (ForeignPtr C.CTensor -> IO (ForeignPtr C.CTensor)) -> IO (Tensor ty ki sz)
writeToNewTensor f = do
#if WITH_CUDA
  case demote @ki of
    KCuda ->
      unlessM (cbool <$> C.hasCUDA)
        $ error "Compiled with cuda and a device was requested, but it isn't available at runtime"
    _ -> pure ()
#endif
  o <- C.emptyTensorOptions (cDeviceType (demote @ki))
                           (cScalarType (demote @ty))
                           (boolc False)
  t <- C.empty__aom (demoteNv @sz) o (fromIntegral $ fromEnum C.MemoryFormatContiguous)
  _ <- f t
  pure (Tensor t Nothing)

-- * Scalars

toScalar :: forall ty ki. (TensorConstraints ty ki '[])
         => (TensorTyToHs ty) -> IO (Tensor ty ki '[])
toScalar n = toDevice =<< fromVectorNoCopy (V.singleton (coerce (hsScalarToC n)))

fromScalar :: (IsScalarLike sz ~ True, TensorConstraints ty ki sz)
           => Tensor ty ki sz -> IO (TensorTyToHs ty)
fromScalar t' = do
  t@(Tensor p _) <- toCpu t'
  arr <- castPtr <$> C.data_ptr p
  _ <- peek arr
  r <- peek arr
  touchForeignPtr p
  pure (cScalarToHs r)

isScalar :: Tensor ty ki sz -> Bool
isScalar t = numel t <= 1

cint x = fromIntegral x :: CInt
clong x = fromIntegral x :: CLong
cfloat x = fromIntegral x :: CFloat
cdouble x = fromIntegral x :: CDouble

int x = fromIntegral x :: Int
long x = fromIntegral x :: Int64
float x = fromIntegral x :: Float
double x = fromIntegral x :: Double

-- * Constraints
--
-- These are all the identify function but they're useful for expressing your beliefs about types

-- | Specify the size of this tensor.
sized :: forall sz ty ki. Size sz -> Tensor ty ki sz -> Tensor ty ki sz
sized Size x = x

-- | Specify the type of the tesnor
typed :: forall ty ki sz. Tensor ty ki sz -> Tensor ty ki sz
typed x = x

-- | Specify the storage of the tensor (CPU vs CUDA)
stored :: forall ki ty sz. Tensor ty ki sz -> Tensor ty ki sz
stored x = x

-- | The 2nd tensor is returned, it should have the same type as the first
like :: forall ty ki sz . Tensor ty ki sz -> Tensor ty ki sz -> Tensor ty ki sz
like _ x = x

onCpu :: forall ty sz. Tensor ty KCpu sz -> Tensor ty KCpu sz
onCpu x = x

#if WITH_CUDA
onCuda :: forall ty sz. Tensor ty KCuda sz -> Tensor ty KCuda sz
onCuda x = x
#endif

-- * RNG

setSeed :: Word64 -> IO ()
setSeed s = do
  g <- C.cpuGenerator
  C.setSeed g s
#if WITH_CUDA
  whenM (cbool <$> C.hasCUDA) $ do
      g' <- C.cudaGenerator
      C.setSeed g' s
#endif
  pure ()

generatorFor :: TensorKind -> IO (Ptr C.CGenerator)
generatorFor KCpu  = C.cpuGenerator
#if WITH_CUDA
generatorFor KCuda = C.cudaGenerator
#endif

-- * Comparison / Verification

isCoalesced :: Tensor ty ki size -> IO Bool
isCoalesced t = cbool <$> C.is_coalesced_m (tensorPtr t)

isContiguous :: Tensor ty ki size -> IO Bool
isContiguous t = cbool <$> C.is_contiguous (tensorPtr t)

isCpu :: Tensor ty ki size -> IO Bool
isCpu t = (\x -> toEnum (fromIntegral x) == C.BackendCPU) <$> C.backend (tensorPtr t)

isCuda :: Tensor ty ki size -> IO Bool
isCuda t = (\x -> toEnum (fromIntegral x) == C.BackendCUDA) <$> C.backend (tensorPtr t)

isDistributed :: Tensor ty ki size -> IO Bool
isDistributed t = cbool <$> C.is_distributed__t (tensorPtr t)

isFloatingPoint :: Tensor ty ki size -> IO Bool
isFloatingPoint t = cbool <$> C.is_floating_point__t (tensorPtr t)

isNonzero :: Tensor ty ki size -> IO Bool
isNonzero t = cbool <$> C.is_nonzero__t (tensorPtr t)

isSigned :: Tensor ty ki size -> IO Bool
isSigned t = cbool <$> C.is_signed__t (tensorPtr t)

isSparse :: Tensor ty ki size -> IO Bool
isSparse t = cbool <$> C.is_sparse (tensorPtr t)

size :: forall sz ty ki. Tensor ty ki sz -> Vector Int
size (Tensor _ _) = demoteNv @sz

numel :: Tensor ty ki sz -> Int
numel t = case V.product (size t) of
            0 -> 1
            x -> x

-- * Tensor creation

-- | Create an empty, uninitialized, Tensor
-- >>> numel <$> empty @TFloat @KCpu @'[1,2,3]
-- 6
empty :: forall ty ki sz. (TensorConstraints ty ki sz) => IO (Tensor ty ki sz)
empty = do
#if WITH_CUDA
  case demote @ki of
    KCuda -> do
      b <- cbool <$> C.hasCUDA
      unless b $ error "Compiled with cuda and a device was requested, but it isn't available at runtime"
    _ -> pure ()
#endif
  o <- C.emptyTensorOptions (cDeviceType (demote @ki))
                           (cScalarType (demote @ty))
                           (boolc False)
  tp <- C.empty__aom (demoteNv @sz) o (fromIntegral $ fromEnum C.MemoryFormatContiguous)
  pure $ Tensor tp Nothing

-- | A tensor of all 0s
zeros :: forall ty ki sz. (TensorConstraints ty ki sz) => IO (Tensor ty ki sz)
zeros = writeToNewTensor (\ptr -> C.zeros_out__ta ptr (demoteNv @sz))

-- | A tensor of all 1s
ones :: forall ty ki sz. (TensorConstraints ty ki sz) => IO (Tensor ty ki sz)
ones = writeToNewTensor (\ptr -> C.ones_out__ta ptr (demoteNv @sz))

full :: forall ty ki sz. (TensorConstraints ty ki sz) => TensorTyToHs ty -> IO (Tensor ty ki sz)
full fillValue = do
  fill <- toCScalar @ty @ki (hsScalarToC fillValue)
  writeToNewTensor (\ptr -> C.full_out__tas ptr (demoteNv @sz) fill)

arange :: forall ty ki sz. (TensorConstraints ty ki '[sz], KnownNat sz)
       => TensorTyToHs ty -> TensorTyToHs ty -> IO (Tensor ty ki '[sz])
arange start end = do
  start' <- toCScalar @ty @ki (hsScalarToC start)
  end'   <- toCScalar @ty @ki (hsScalarToC end)
  step'  <- toCScalar @ty @ki (hsScalarToC (fromIntegral $ ceiling ((toDouble end - toDouble start) / fromIntegral (demoteN @sz))))
  writeToNewTensor (\ptr -> C.arange_out__tsss ptr start' end' step')

linspace :: forall ty ki sz. (TensorConstraints ty ki '[sz], KnownNat sz)
       => TensorTyToHs ty -> TensorTyToHs ty -> IO (Tensor ty ki '[sz])
linspace start end = do
  start' <- toCScalar @ty @ki (hsScalarToC start)
  end'   <- toCScalar @ty @ki (hsScalarToC end)
  writeToNewTensor (\ptr -> C.linspace_out__tss6 ptr start' end' (fromIntegral $ demoteN @sz))

logspace :: forall ty ki sz. (TensorConstraints ty ki '[sz], KnownNat sz)
       => TensorTyToHs ty -> TensorTyToHs ty -> Double -> IO (Tensor ty ki '[sz])
logspace start end base = do
  start' <- toCScalar @ty @ki (hsScalarToC start)
  end'   <- toCScalar @ty @ki (hsScalarToC end)
  writeToNewTensor (\ptr -> C.logspace_out__tss6d ptr start' end' (fromIntegral $ demoteN @sz) (coerce base))

eye :: forall ty ki sz0 sz1. (TensorConstraints ty ki sz0, SingI sz1) => IO (Tensor ty ki '[sz0,sz1])
eye = writeToNewTensor (\ptr -> C.eye_out__t66 ptr (demoteN @sz0) (demoteN @sz1))

-- TODO Make generator explicit
randperm :: forall ty ki sz d. (SingI sz, TensorConstraints ty ki '[sz]) => IO (Tensor ty ki '[sz])
randperm = writeToNewTensor (\ptr -> C.randperm_out__t6 ptr (demoteN @sz))

-- | Convert a C pointer to a tensor. Storage is not copied, you are
--   responsible for keeping it alive! You can clone the tensor
--   afterward if you want to have new and separate storage.
--
-- TODO What happens to the storage? Is the above really true? I want it to be left alone.
fromPtr :: forall ty sz. (TensorConstraints ty KCpu sz)
        => Ptr (TensorTyToHsC ty) -> Maybe (ForeignPtr ()) -> IO (Tensor ty KCpu sz)
fromPtr d auxiliary = do
  o <- C.emptyTensorOptions (cDeviceType KCpu)
                           (cScalarType (demote @ty))
                           (boolc False)
  ptr <- C.tensorFromBlob o (castPtr d) (demoteNv @sz)
  pure $ Tensor ptr auxiliary

-- | Like @fromPtr@ this converts a memory block to a tensor but it
--   takes ownership of the storage and will free it when the tensor
--   goes out of scope
fromPtr' :: forall ty sz. (TensorConstraints ty KCpu sz)
         => Ptr (TensorTyToHsC ty) -> IO (Tensor ty KCpu sz)
fromPtr' d = do
  d' <- newForeignPtr finalizerFree (coerce d)
  t <- fromPtr d (Just d')
  pure t

-- | Convert a Storable Vector to a Tensor. This copies the storage!
fromVector :: forall ty sz. (TensorConstraints ty KCpu sz)
           => V.Vector (TensorTyToHsC ty) -> IO (Tensor ty KCpu sz)
fromVector v = do
  unless (product sz == len)
    $ error
    $ "Vector size does not match Tensor storage size, vector is " ++ show (V.length v) ++ " but tensor needs " ++ show sz
  V.unsafeWith v (\ptr -> do
                     ptr' <- mallocArray len
                     copyArray ptr' ptr len
                     fromPtr' ptr')
  where len = V.length v
        sz = demoteNs @sz

-- | Convert a Storable Vector to a Tensor. Storage is not copied. You must not
-- use the vector after this (TODO enforce this once we get linear types).
fromVectorNoCopy :: forall ty sz. (TensorConstraints ty KCpu sz)
                 => V.Vector (TensorTyToHsC ty) -> IO (Tensor ty KCpu sz)
fromVectorNoCopy v = do
  unless (product sz == len)
    $ error
    $ "Vector size does not match Tensor storage size, vector is " ++ show (V.length v) ++ " but tensor needs " ++ show sz
  let (fptr, _) = V.unsafeToForeignPtr0 v
  withForeignPtr fptr (\ptr -> fromPtr ptr (Just (castForeignPtr fptr)))
  where len = V.length v
        sz = demoteNs @sz

toVector :: forall ty sz. Tensor ty 'KCpu sz -> IO (Vector (TensorTyToHsC ty))
toVector t@(Tensor p _) = do
  p' <- C.contiguous_mm p (fromIntegral $ fromEnum C.MemoryFormatContiguous)
  arr <- castPtr <$> C.data_ptr p'
  len <- fromIntegral <$> C.numel p'
  ptr' <- newForeignPtr finalizerFree =<< mallocArray len
  withForeignPtr ptr' (\ptr' -> copyArray ptr' arr len)
  touchForeignPtr p'
  pure $ V.unsafeFromForeignPtr0 ptr' len

-- | Convert a strict ByteString to a Tensor.
fromByteString :: forall sz. (TensorConstraints TByte KCpu sz)
               => BS.ByteString -> IO (Tensor TByte KCpu sz)
fromByteString bs = do
  let (fptr, offset, len) = BS.toForeignPtr bs
  let fptr0 = fptr `plusForeignPtr` offset
  unless (product sz == len)
    $ error
    $ "Bytestring size does not match Tensor storage size, bytestring is " ++ show len ++ " but tensor needs " ++ show sz
  ptr' <- mallocArray len
  withForeignPtr fptr0 (\ptr0 -> copyArray ptr' ptr0 len)
  fromPtr' ptr'
  where sz = demoteNs @sz

-- | Convert a ByteString to a Tensor. Storage is not copied but a reference to
-- it is held. You must not use the bytestring after this (TODO enforce this once we get linear types).
fromByteStringNoCopy :: forall sz. (TensorConstraints TByte KCpu sz)
                     => BS.ByteString -> IO (Tensor TByte KCpu sz)
fromByteStringNoCopy bs = do
  let (fptr, offset, len) = BS.toForeignPtr bs
  let fptr0 = fptr `plusForeignPtr` offset
  unless (product sz == len)
    $ error
    $ "Bytestring size does not match Tensor storage size, bytestring is " ++ show len ++ " but tensor needs " ++ show sz
  withForeignPtr fptr0 (\ptr0 -> fromPtr ptr0 (Just (castForeignPtr fptr0)))
  where sz = demoteNs @sz

-- | Convert a Tensor to a ByteString. Storage is not copied but a reference to
-- it is held.
toByteStringNoCopy :: forall ty sz. Tensor TByte 'KCpu sz -> IO BS.ByteString
toByteStringNoCopy t@(Tensor p _) = do
  p' <- C.data_ptr =<< C.contiguous_mm p (fromIntegral $ fromEnum C.MemoryFormatContiguous)
  p'' <- newForeignPtr_ (castPtr p')
  pure $ BS.fromForeignPtr p'' 0 (numel t)

-- | This pointer is only alive while the tensor is available!
withDataPtr :: forall ty sz a. Tensor ty 'KCpu sz -> (Ptr (TensorTyToHsC ty) -> Int64 -> IO a) -> IO a
withDataPtr t@(Tensor p _) f = do
  p' <- C.contiguous_mm p (fromIntegral $ fromEnum C.MemoryFormatContiguous)
  arr <- castPtr <$> C.data_ptr p'
  len <- C.numel p'
  r <- f arr len
  touchForeignPtr p'
  pure r

-- | Create a uniform tensor between [0,1]
-- >>> numel <$> rand @TFloat @KCpu @'[3,1]
-- 3
rand :: forall ty ki sz. (TensorConstraints ty ki sz) => IO (Tensor ty ki sz)
rand = do
  g <- generatorFor (demote @ki)
  writeToNewTensor (\ptr -> C.rand_out__tag ptr (demoteNv @sz) g)

-- | Create a tensor with normally distributed data
-- >>> numel <$> randn @TFloat @KCpu @'[3,1]
-- 3
randn :: forall ty ki sz. (TensorConstraints ty ki sz, IsFloatTy ty ~ 'True)
      => IO (Tensor ty ki sz)
randn = do
  g <- generatorFor (demote @ki)
  writeToNewTensor (\ptr -> C.randn_out__tag ptr (demoteNv @sz) g)

randint :: forall ty ki sz. (TensorConstraints ty ki sz)
        => Int64 -> Int64 -> IO (Tensor ty ki sz)
randint low high = do
  g <- generatorFor (demote @ki)
  writeToNewTensor (\ptr -> C.randint_out__t66ag ptr low high (demoteNv @sz) g)

-- * AD operations

-- | Enable gradients for a tensor
needGrad :: forall ty ki sz. Tensor ty ki sz -> IO (Tensor ty ki sz)
needGrad t@(Tensor ptr _) = do
  CV.set_requires_grad ptr $ boolc True
  pure t

-- | Enable gradients for a tensor wrapped in AnyTensor, the generic way to
-- store collections of tensors.
needGradAny :: AnyTensor -> IO AnyTensor
needGradAny x@(AnyTensor (Tensor ptr _)) = do
  CV.set_requires_grad ptr $ boolc True
  pure x

-- | Disable gradients for this tensor.
noGrad :: forall ty ki sz. Tensor ty ki sz -> IO (Tensor ty ki sz)
noGrad t@(Tensor ptr _) = do
  CV.set_requires_grad ptr $ boolc False
  pure t

-- | Disable gradients for a tensor wrapped in AnyTensor, the generic way to
-- store collections of tensors.
noGradAny :: AnyTensor -> IO AnyTensor
noGradAny x@(AnyTensor (Tensor ptr _)) = do
  CV.set_requires_grad ptr $ boolc False
  pure x

setRequiresGrad :: forall ty ki sz. Bool -> Tensor ty ki sz -> IO (Tensor ty ki sz)
setRequiresGrad b t@(Tensor ptr _) = do
  CV.set_requires_grad ptr $ boolc b
  pure t

-- | Enable or disable gradients for a tensor wrapped in AnyTensor, the generic
-- way to store collections of tensors.
setRequiresGradAny :: Bool -> AnyTensor -> IO AnyTensor
setRequiresGradAny b x@(AnyTensor (Tensor ptr _)) = do
  CV.set_requires_grad ptr $ boolc b
  pure x

-- | Returns True if the tensor requires a gradient.
requiresGrad :: forall ty ki sz. Tensor ty ki sz -> IO Bool
requiresGrad t@(Tensor ptr x) = cbool <$> CV.requires_grad ptr

-- | Returns True if the generic tensor, wrapped in AnyTensor, requires a
-- gradient.
requiresGradAny :: AnyTensor -> IO Bool
requiresGradAny (AnyTensor (Tensor ptr _)) = cbool <$> CV.requires_grad ptr

backward1 :: Tensor ty ki sz -> Bool -> Bool -> IO ()
backward1 (Tensor tp _) keep create = C.backward1 tp (boolc keep) (boolc create)

withForeignPtrList :: [ForeignPtr C.CTensor] -> (V.Vector (Ptr C.CTensor) -> IO a) -> IO a
withForeignPtrList fps f = do
  r <- f (V.fromList (map unsafeForeignPtrToPtr fps))
  mapM_ touchForeignPtr fps
  pure r

backward :: [AnyTensor] -> Bool -> Bool -> IO ()
backward ts keep create =
  withForeignPtrList (map (\(AnyTensor (Tensor p _)) -> p) ts) (\ps -> C.backwardN ps (boolc keep) (boolc create))

gradient :: Tensor ty ki sz -> IO (Maybe (Tensor ty ki sz))
gradient (Tensor ptr x) = do
  g <- CV.grad (castForeignPtr ptr)
  def <- C.is_defined (castForeignPtr g)
  pure $ if (cbool def) then
           -- Gradients need the parent pointer to be alive, not its storage
           -- grad storage is disjoint from primal storage
           Just (Tensor (castForeignPtr g) (Just (castForeignPtr ptr))) else
           Nothing

clearGradinet :: Tensor ty ki sz -> IO ()
clearGradinet t@(Tensor ptr _) = do
  g <- CV.grad (castForeignPtr ptr)
  def <- C.is_defined (castForeignPtr g)
  when (cbool def) $ (C.zero___t (castForeignPtr g) >> pure ())
  pure ()

disableGrad :: IO ()
disableGrad = CV.set_grad_enabled (boolc False)

unsafeEnableGrad :: IO ()
unsafeEnableGrad = CV.set_grad_enabled (boolc True)

setGradEnabled :: Bool -> IO ()
setGradEnabled b = CV.set_grad_enabled (boolc b)

gradEnabled :: IO Bool
gradEnabled = cbool <$> CV.grad_enabled

withoutGrad :: IO x -> IO x
withoutGrad f = do
  e <- gradEnabled
  disableGrad
  r <- f
  setGradEnabled e
  pure r

withGrad :: IO x -> IO x
withGrad f = do
  e <- gradEnabled
  unsafeEnableGrad
  r <- f
  setGradEnabled e
  pure r

unsafePausingGradForPtr :: ForeignPtr C.CTensor -> IO a -> IO a
unsafePausingGradForPtr p fn = do
  CV.set_requires_grad p (boolc False)
  r <- fn
  CV.set_requires_grad p (boolc True)
  pure r

-- | Free the memory assoicated with this tensor explicitly. You shouldn't need
-- to do this. It's unsafe if you ever access the tensor again but it can be a
-- handy operation in a bind.
unsafeFree :: Tensor ty ki sz -> IO ()
unsafeFree (Tensor tp _) = do
  finalizeForeignPtr tp

detach :: Tensor ty ki sz -> IO (Tensor ty ki sz)
detach r@(Tensor tp _) = C.detach___t tp >> pure r

detachAny :: AnyTensor -> IO AnyTensor
detachAny r@(AnyTensor (Tensor tp _)) = C.detach___t tp >> pure r

-- * Views

-- TODO Does the result of copy need a new require_grad?
--
-- | Copy data from this tensor to a new one that does not share the same
--   storage and store a different kind of value. This can be used to move data
--   between the CPU<->GPU.
copy :: forall ty' ki' sz' ty ki sz.
       (TensorConstraints ty' ki' sz, SingI sz', Product sz ~ Product sz')
     => (Tensor ty ki sz) -> IO (Tensor ty' ki' sz')
copy (Tensor tp _) = writeToNewTensor (\ptr -> C.copy__mtb ptr tp (boolc False))

-- | Copy data from this tensor to a new one that stores a different kind of
-- value and might not share the same storage. Calling this and expecting it to
-- return the same type is free. The first type argument is the destination type.
toType :: forall ty' ty ki sz.
       (TensorConstraints ty' ki sz)
     => (Tensor ty ki sz) -> IO (Tensor ty' ki sz)
toType (Tensor tp x) =
  case ((sing :: Sing ty) %~ (sing :: Sing ty')) of
    (Proved Refl) -> pure $ Tensor tp x -- FIXME This is very sketchy! You can never rely on toType with these semantics.
    _             -> writeToNewTensor (\ptr -> C.copy__mtb ptr tp (boolc False))

-- TODO the API also has "to" which have an option not to copy. What's the semantics of this?
-- | Convert a tensor to another device (either another CUDA device or the CPU)
toDevice :: forall ty ki ki' sz.
           (TensorConstraints ty ki' sz, SingI ki, SingI ki')
         => Tensor ty ki sz -> IO (Tensor ty ki' sz)
toDevice ten@(Tensor tp x) = do
  case ((sing :: Sing ki) %~ (sing :: Sing ki')) of
    (Proved Refl) -> pure $ Tensor tp x
    _ -> do
#if WITH_CUDA
      case demote @ki of
        KCuda -> do
          unlessM (cbool <$> C.hasCUDA)
            $ error "Compiled with cuda and a device was requested, but it isn't available at runtime"
        _ -> pure ()
#endif
      copy ten

-- | Convert a tensor to the CPU
toCpu :: (TensorConstraints ty 'KCpu sz, SingI ki)
      => Tensor ty ki sz -> IO (Tensor ty 'KCpu sz)
toCpu = toDevice

#if WITH_CUDA
-- | Convert a tensor to the GPU
toCuda :: (TensorConstraints ty 'KCpu sz, SingI ki)
      => Tensor ty ki sz -> IO (Tensor ty 'KCuda sz)
toCuda = toDevice
#endif

clone :: forall ty ki sz. Tensor ty ki sz -> IO (Tensor ty ki sz)
clone (Tensor tp _) = C.clone__tm tp (fromIntegral $ fromEnum  C.MemoryFormatPreserve) >>= \tp' -> pure (Tensor tp' Nothing)

expandAs :: forall ty ki ty' ki' sz sz'. (SingI (BroadcastSizes sz sz'))
         => Tensor ty ki sz -> Tensor ty' ki' sz' -> IO (Tensor ty ki (BroadcastSizes sz sz'))
expandAs x@(Tensor tp a) _ = do
  te <- C.expand_mab tp (demoteNv @(BroadcastSizes sz sz')) (boolc True)
  pure $ Tensor te a

-- sz' is an argument
expand :: forall sz' ty ki sz. (SingI sz', ExpandTo sz sz' ~ True)
       => Tensor ty ki sz -> IO (Tensor ty ki sz')
expand x@(Tensor t a) = do
  te <- C.expand_mab t (demoteNv @sz') (boolc True)
  pure $ Tensor te a

-- | This is an escape hatch when you know something is true but GHC can't prove
-- it. You shouldn't need to use this.
unsafeSize :: Tensor ty ki sz -> Tensor ty ki sz'
unsafeSize = unsafeCoerce

broadcast2 :: (SingI (BroadcastSizes sz sz'))
          => Tensor ty ki sz
          -> Tensor ty ki sz'
          -> IO (Tensor ty ki (BroadcastSizes sz sz')
               ,Tensor ty ki (BroadcastSizes sz sz'))
broadcast2 (Tensor t s) (Tensor t' s') = do
  withForeignPtr t
    (\pt -> withForeignPtr t'
      (\pt' -> do
          v <- C.broadcast_tensors__l (V.fromList [pt,pt'])
          let (r,r') = (V.head v, V.last v)
          rp  <- newForeignPtr C.deleteTensor r
          rp' <- newForeignPtr C.deleteTensor r'
          pure (Tensor rp s, Tensor rp' s')))

broadcast3 :: (SingI (BroadcastSizes sz sz'), SingI (BroadcastSizes sz (BroadcastSizes sz' sz'')))
           => Tensor ty ki sz
           -> Tensor ty ki sz'
           -> Tensor ty ki sz''
           -> IO (Tensor ty ki (BroadcastSizes sz (BroadcastSizes sz' sz''))
                ,Tensor ty ki (BroadcastSizes sz (BroadcastSizes sz' sz''))
                ,Tensor ty ki (BroadcastSizes sz (BroadcastSizes sz' sz'')))
broadcast3 (Tensor t s) (Tensor t' s') (Tensor t'' s'') = do
  withForeignPtr t
    (\pt -> withForeignPtr t'
      (\pt' -> withForeignPtr t''
        (\pt'' -> do
          v <- C.broadcast_tensors__l (V.fromList [pt,pt',pt''])
          let (r,r',r'') = (V.head v, V.head (V.tail v), V.last v)
          rp   <- newForeignPtr C.deleteTensor r
          rp'  <- newForeignPtr C.deleteTensor r'
          rp'' <- newForeignPtr C.deleteTensor r''
          pure (Tensor rp s, Tensor rp' s', Tensor rp'' s''))))

cartesianProduct2 :: (SingI sz)
                  => Tensor ty ki '[sz]
                  -> Tensor ty ki '[sz]
                  -> IO (Tensor ty ki '[sz,sz])
cartesianProduct2 (Tensor t _) (Tensor t' _) =
  withForeignPtrs [t,t']
    (\l -> wrapTensorM (C.cartesian_prod__l (V.fromList l)) Nothing)

cartesianProduct3 :: (SingI sz)
                  => Tensor ty ki '[sz]
                  -> Tensor ty ki '[sz]
                  -> Tensor ty ki '[sz]
                  -> IO (Tensor ty ki '[sz,sz,sz])
cartesianProduct3 (Tensor t _) (Tensor t' _) (Tensor t'' _) =
  withForeignPtrs [t,t',t'']
  (\l -> wrapTensorM (C.cartesian_prod__l (V.fromList l)) Nothing)

cross :: forall (dimension :: Nat) ty ki sz.
        (SingI dimension, SelectIndex sz dimension ~ 3)
      => Dimension dimension
      -> Tensor ty ki sz
      -> Tensor ty ki sz
      -> IO (Tensor ty ki sz)
cross Dimension (Tensor tp _) (Tensor tp' _) = do
  wrapTensorM (C.cross__tt6 tp tp' (Just (demoteN @dimension))) Nothing

-- * Selection

flatten :: forall ty ki sz. (SingI (Product sz)) => (Tensor ty ki sz) -> IO (Tensor ty ki '[Product sz])
flatten x@(Tensor t a) = do
  t' <- C.view_ma t (V.fromList $ [demoteN @(Product sz)])
  pure $ Tensor t' a

unflatten :: forall ty ki sz'. (SingI sz') => Tensor ty ki '[Product sz'] -> IO (Tensor ty ki sz')
unflatten x@(Tensor t a) = do
  t' <- C.view_ma t (demoteNv @sz')
  pure $ Tensor t' a

diagonal :: forall ty ki sz sz'. (SingI (Min sz sz')) => Tensor ty ki '[sz, sz'] -> IO (Tensor ty ki '[Min sz sz'])
diagonal x@(Tensor t a) = do
  t' <- C.diag__t6 t 0
  pure $ Tensor t' a

-- offset is an argument
diagonalOffset :: forall offset ty ki sz sz'. (SingI offset, SingI (Min sz sz' - offset), (offset + 1 <=? Min sz sz') ~ True) =>
  Tensor ty ki '[sz, sz'] -> Bool -> IO (Tensor ty ki '[(Min sz sz') - offset])
diagonalOffset x@(Tensor t a) direction = do
  t' <- C.diag__t6 t ((if direction then 1 else -1) * (demoteN @offset))
  pure $ Tensor t' a

toDiagonal :: forall ty ki sz. SingI sz => Tensor ty ki '[sz] -> IO (Tensor ty ki '[sz, sz])
toDiagonal x@(Tensor t a) = do
  t' <- C.diag__t6 t 0
  pure $ Tensor t' a

trace :: (SingI sz)
      => Tensor ty ki sz
      -> IO (Scalar ty ki)
trace (Tensor t _) = wrapTensorM (C.trace__t t) Nothing

tril :: (SingI sz)
      => Tensor ty ki sz
      -> Maybe Int64
      -> IO (Tensor ty ki sz)
tril (Tensor t _) diagonal = wrapTensorM (C.tril__t6 t (fromMaybe 0 diagonal)) Nothing

triu :: (SingI sz)
      => Tensor ty ki sz
      -> Maybe Int64
      -> IO (Tensor ty ki sz)
triu (Tensor t _) diagonal = wrapTensorM (C.triu__t6 t (fromMaybe 0 diagonal)) Nothing

-- * Manipulation

str :: Tensor ty ki sz -> IO Text
str t = T.pack <$> C.str (tensorPtr t)

-- TODO This has some debugging in it, but we might leave it around as a sanity
-- check. Printing is rare and slow anyway, the speed hit is inconsequential.
out :: Tensor ty ki sz -> IO ()
out t = str (debuggingVerifyShape "Verifying while printing" t) >>= T.putStrLn

instance {-# OVERLAPPING #-} (TensorConstraints ty ki sz) => ShowIO (Tensor ty ki sz) where
  showIO x = caseScalar x (\x -> showScalar <$> fromScalar x) (\x -> T.unpack <$> str x) (sing :: Sing sz)

instance {-# OVERLAPPING #-} (TensorConstraints ty ki sz) => ShowIO (Maybe (Tensor ty ki sz)) where
  showIO Nothing  = pure "Nothing"
  showIO (Just x) =  ("Just " ++) `liftM` showIO x

-- * Indexing, slicing, and joining

-- A cuda tensor cannot be viewed as a cpu tensor. They won't be copied to another device by aten!
view :: forall sz' ty ki sz. (SingI sz', Product sz ~ Product sz')
     => (Tensor ty ki sz) -> IO (Tensor ty ki sz')
view x@(Tensor t a) = do
  isCont <- isContiguous x
  if isCont then
    wrapTensorM (C.view_ma t (demoteNv @sz')) a else
    error $ "You should use reshape instead of view because the tensor is not contiguous when converting "
          ++ show ((demoteNv @sz) :: Vector Int) ++ " sized tensor to " ++ show ((demoteNv @sz') :: Vector Int)

reshape :: forall sz' ty ki sz. (SingI sz', Product sz ~ Product sz')
     => (Tensor ty ki sz) -> IO (Tensor ty ki sz')
reshape x@(Tensor t a) =
  wrapTensorM (C.reshape__ta t (demoteNv @sz')) a

squeeze :: forall ty ki sz. (SingI (Squeeze sz)) => Tensor ty ki sz -> IO (Tensor ty ki (Squeeze sz))
squeeze x@(Tensor t a) = do
  t' <- C.squeeze__t t
  pure $ Tensor t' a

-- offset is an argument
squeezeOffset :: forall offset ty ki sz. (SingI offset, SingI (RemoveDimension sz offset))
              => Tensor ty ki sz -> IO (Tensor ty ki (RemoveDimension sz offset))
squeezeOffset x@(Tensor t a) = do
  t' <- C.squeeze__t6 t (demoteN @offset)
  pure $ Tensor t' a

unsqueeze :: forall (dimension :: Nat) ty ki sz.
            (SingI (InsertIndex sz dimension 1), SingI dimension)
          => Dimension dimension
          -> Tensor ty ki sz
          -> IO (Tensor ty ki (InsertIndex sz dimension 1))
unsqueeze Dimension (Tensor tp s) = do
  wrapTensorM (C.unsqueeze__t6 tp (demoteN @dimension)) s

t :: forall ty ki sz sz'. (SingI sz, SingI sz') => Tensor ty ki '[sz,sz'] -> IO (Tensor ty ki '[sz',sz])
t x@(Tensor t a) = do
  t' <- C.t__t t
  pure $ Tensor t' a

transpose :: forall (dim0 :: Nat) (dim1 :: Nat) ty ki sz. (SingI dim0, SingI dim1, SingI (Swap sz dim0 dim1))
          => Tensor ty ki sz -> IO (Tensor ty ki (Swap sz dim0 dim1))
transpose x@(Tensor t a) = do
  t' <- C.transpose__t66 t (demoteN @dim0) (demoteN @dim1)
  pure $ Tensor t' a

cat :: forall (nr :: Nat) (dimension :: Nat) ty ki sz.
         (SingI nr, SingI dimension, (Num (TensorTyToHsC ty)), SingI ty, SingI ki
         ,TensorConstraints ty ki sz
         ,(Storable (TensorTyToHsC ty)), (SingI (MultiplyIndex sz dimension nr)))
       => V'.Vector (Tensor ty ki sz) -> IO (Maybe (Tensor ty ki (MultiplyIndex sz dimension nr)))
cat ts = do
  if V'.length ts == (demoteN @nr) then
    case ts V'.!? 0 of
      Just t -> do
        x <- withForeignPtrList (V'.toList (V'.map tensorPtr ts)) (\vec -> C.cat__l6 vec (demoteN @dimension))
        pure $ Just $ Tensor x Nothing
      Nothing -> pure Nothing else
     pure Nothing

cat2 :: forall (dimension :: Nat) ty ki sz sz'. (SingI dimension, SingI (Cat2 sz sz' dimension), (dimension <=? Length sz)~True)
     => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor ty ki (Cat2 sz sz' dimension))
cat2 t@(Tensor tptr _) t'@(Tensor tptr' _) = do
  x <- withForeignPtrList [tptr, tptr'] (\vec -> C.cat__l6 vec (demoteN @dimension))
  pure $ Tensor x Nothing

-- | Chunk the tensor into N pieces
--
-- TODO Turn this into an example
-- (r :: Tensor 'TFloat 'KCpu '[17,13,4]) <- randn
-- (a,_) <- chunk (dimension_ @0) (chunks_ @3) r
-- :t chunk (dimension_ @0) (chunks_ @3) r
chunk :: forall (dimension :: Nat) (chunks :: Nat) ty ki sz chunkSize remainderSize.
        (SingI dimension, SingI chunks, SingI remainderSize
        ,SingI (ReplaceDimension sz dimension remainderSize)
        ,SingI (ReplaceDimension sz dimension chunkSize)
        ,(DivRoundUp (SelectIndex sz dimension) chunks) ~ chunkSize
        ,(Rem (SelectIndex sz dimension) chunkSize) ~ remainderSize)
      => Dimension dimension
      -> Chunks chunks
      -> Tensor ty ki sz
      -- TODO I hate that this is a list, replace it with a vector
      -- TODO We know the size of the vector
      -> IO ([Tensor ty ki (ReplaceDimension sz dimension chunkSize)]
           -- TODO This can result in tensor with a strange zero
           -- dimension. Protect this more so that we can't access it at all in
           -- some conditions.
           ,Maybe (Tensor ty ki (ReplaceDimension sz dimension remainderSize)))
chunk Dimension Chunks (Tensor tp s) = do
  chunkedTensors <- mapM (newForeignPtr C.deleteTensor) . V.toList  =<< C.chunk__t66 tp (demoteN @chunks) (demoteN @dimension)
  if (demoteN @remainderSize) == 0 then
    pure (map (\x -> Tensor x s) chunkedTensors, Nothing)
  else pure (map (\x -> Tensor x s) (P.take (length chunkedTensors - 1) chunkedTensors)
            ,Just $ Tensor (last chunkedTensors) s)

-- | Split a tensor into parts of the given size
--
-- TODO Turn me into a real example
-- (r :: Tensor 'TFloat 'KCpu '[10,13,4]) <- randn
-- split (dimension_ @0) (size_ @3) r
split :: forall (dimension :: Nat) (size :: Nat) ty ki sz remainderSize.
        (SingI dimension, SingI size, SingI remainderSize
        ,SingI (ReplaceDimension sz dimension remainderSize)
        ,SingI (ReplaceDimension sz dimension size)
        ,(Rem (SelectIndex sz dimension) size) ~ remainderSize)
      => Dimension dimension
      -> Size size
      -> Tensor ty ki sz
      -> IO ([Tensor ty ki (ReplaceDimension sz dimension size)]
           -- TODO This can result in tensor with a strange zero
           -- dimension. Protect this more so that we can't access it at all in
           -- some conditions.
           ,Maybe (Tensor ty ki (ReplaceDimension sz dimension remainderSize)))
split Dimension Size (Tensor tp s) = do
  chunkedTensors <- mapM (newForeignPtr C.deleteTensor) . V.toList  =<< C.split__t66 tp (demoteN @size) (demoteN @dimension)
  if (demoteN @remainderSize) == 0 then
    pure (map (\x -> Tensor x s) chunkedTensors, Nothing)
  else pure (map (\x -> Tensor x s) (P.take (length chunkedTensors - 1) chunkedTensors)
            ,Just $ Tensor (last chunkedTensors) s)

gather :: forall (dimension :: Nat) ty ki sz n.
         (SingI dimension, SingI sz
         ,SingI (ReplaceDimension sz dimension n))
       => Dimension dimension
       -> Tensor ty ki sz
       -> Tensor TLong ki '[n]
       -> IO (Tensor ty ki (ReplaceDimension sz dimension n))
gather Dimension (Tensor t tp) (Tensor index _) =
  wrapTensorM (C.gather__t6tb t (demoteN @dimension) index (boolc False)) tp

take :: Tensor TLong ki '[n] -> Tensor ty ki sz -> IO (Tensor ty ki '[n])
take (Tensor indices _) (Tensor tp _) =
  wrapTensorM (C.take__tt tp indices) Nothing

stack :: forall (nr :: Nat) (dimension :: Nat) ty ki sz.
        (SingI nr, SingI dimension, (Num (TensorTyToHsC ty)), SingI ty, SingI ki
        ,TensorConstraints ty ki sz
        ,(Storable (TensorTyToHsC ty)), (SingI (InsertIndex sz dimension nr)))
      => Groups nr
      -> Dimension dimension
      -> V'.Vector (Tensor ty ki sz) -> IO (Maybe (Tensor ty ki (InsertIndex sz dimension nr)))
stack Groups Dimension ts = do
  if V'.length ts == (demoteN @nr) then
    case ts V'.!? 0 of
      Just t -> do
        x <- withForeignPtrList (V'.toList (V'.map tensorPtr ts)) (\vec -> C.stack__l6 vec (demoteN @dimension))
        pure $ Just $ Tensor x Nothing
      Nothing -> pure Nothing else
     pure Nothing

-- | Index a tensor with a runtime index. Note that this can fail at runtime if
-- the index is too large. We allow negative indices to index backward through
-- the tensor (-1 is the last element).
select :: forall (dimension :: Nat) ty ki sz.
         (NonNull sz ~ True, SingI (SelectIndex sz dimension), SingI dimension, SingI (SelectOtherIndexes sz dimension))
       => Tensor ty ki sz -> Int64 -> IO (Tensor ty ki (SelectOtherIndexes sz dimension))
select t@(Tensor tptr _) n =
  if | n < 0 -> select @dimension t (demoteN @(SelectIndex sz dimension) + n)
     | n >= demoteN @(SelectIndex sz dimension) ->
         error $ "Index too large in select: " ++ show n ++ " vs the size of the tensor " ++ show (demote @(SelectIndex sz dimension))
                 ++ " (full size " ++ (show  $ demoteNs @sz) ++ " on dimension " ++ (show $ demote @dimension)
     | otherwise -> wrapTensorM (C.select__t66 tptr (demoteN @dimension) n) Nothing

unbind :: forall (dimension :: Nat) ty ki sz.
         (SingI (SelectOtherIndexes sz dimension), SingI dimension)
       => Dimension dimension
       -> Tensor ty ki sz
       -> IO [Tensor ty ki (SelectOtherIndexes sz dimension)]
unbind Dimension (Tensor tp _) = do
  ts <- mapM (newForeignPtr C.deleteTensor) =<< (V.toList <$> C.unbind__t6 tp (demoteN @dimension))
  pure $ map (\x -> Tensor x Nothing) ts

-- * Narrowing provides views of the Tensor.
--
-- You might call this slice elsewhere. We have many forms depending on what
-- sorts of compile-time guarantees you want. Check out @Torch.Indexing@ for
-- syntax that is much more convenient.

-- FIXME:
-- , SingI (Narrow sz dimension start length)
narrow :: forall dimension start length ty ki sz.
         (SingI dimension, SingI start, SingI length, SingI (Narrow sz dimension start length))
       => Dimension dimension
       -> Size start
       -> Size length
       -> Tensor ty ki sz -> IO (Tensor ty ki (Narrow sz dimension start length))
narrow Dimension Size Size x@(Tensor t a) = do
  t' <- C.narrow__t666 t (demoteN @dimension) (demoteN @start) (demoteN @length)
  pure $ Tensor t' a

-- | A version of narrow which selects one dimension based on a runtime
-- value. This can fail at runtime rather than compile time.
narrow1 :: forall dimension ty ki sz.
         (SingI dimension, SingI (Narrow sz dimension 0 1))
       => Dimension dimension
       -> Tensor ty ki sz -> Int64 -> IO (Tensor ty ki (Narrow sz dimension 0 1))
narrow1 Dimension x@(Tensor t a) i = do
  sz <- CV.shape t
  unless (-1*(1+sz V.! (demoteN @dimension)) < coerce i
          && coerce i < sz V.! (demoteN @dimension))
    $ error
    $ "narrow1 is trying to index tensor " <> show sz
    <> " at dimension " <> show (demote @dimension)
    <> " with an out of bounds index " <> show i
  t' <- C.narrow__t666 t (demoteN @dimension) i 1
  pure $ Tensor t' a

narrowFrom :: forall dimension start ty ki sz.
         (SingI dimension, SingI start, SingI (Narrow sz dimension start (LengthRemaining sz dimension start))
         ,KnownNat (SelectIndex sz dimension)
         ,(start <=? (sz !! dimension)) ~ True, SingI (sz !! dimension), KnownNat (sz !! dimension), KnownNat start)
       => Dimension dimension
       -> Size start
       -> Tensor ty ki sz
       -> IO (Tensor ty ki (Narrow sz dimension start (LengthRemaining sz dimension start)))
narrowFrom d s x = narrow d s (size_ @(LengthRemaining sz dimension start)) x

narrowTo :: forall dimension end ty ki sz.
         (SingI dimension, SingI end, SingI (Narrow sz dimension 0 end)
         ,(end <=? (sz !! dimension)) ~ True)
       => Dimension dimension
       -> Size end
       -> Tensor ty ki sz
       -> IO (Tensor ty ki (Narrow sz dimension 0 end))
narrowTo d e x = narrow d (size_ @0) e x

narrowFromTo :: forall dimension start end ty ki sz.
         (SingI dimension, SingI start, SingI end, KnownNat end
         ,SingI (Narrow sz dimension start (end - start))
         ,(end <=? (sz !! dimension)) ~ True, (start <=? end) ~ True
         ,SingI (sz !! dimension), KnownNat (sz !! dimension), KnownNat start)
       => Dimension dimension
       -> Size start
       -> Size end
       -> Tensor ty ki sz
       -> IO (Tensor ty ki (Narrow sz dimension start (end - start)))
narrowFromTo d s l x = narrow d s (size_ @(end - start)) x

-- | Like narroFromTo but only the length is taken as a type parameter while the
-- start position is passed in at runtime. Less safe but essential.
narrowFromToByLength :: forall dimension length ty ki sz.
         (SingI dimension, SingI length
         ,SingI (Narrow sz dimension 0 length))
       => Dimension dimension
       -> Size length
       -> Tensor ty ki sz
       -> Int64
       -> IO (Tensor ty ki (Narrow sz dimension 0 length))
narrowFromToByLength Dimension Size x@(Tensor t a) start = do
  let sz = demoteNv @sz
  let dim = demoteN @dimension
  unless ((-1*(1+sz V.! dim)) < coerce start
          && coerce start < sz V.! dim)
    $ error
    $ "narrowFromToByLength is trying to index tensor " <> show sz
    <> " at dimension " <> show (demote @dimension)
    <> " with an out of bounds index " <> show start <> "."
  let endIndex = demoteN @length + if start < 0 then
                                     start + sz V.! dim else
                                     start
  unless (0 <= endIndex && endIndex <= sz V.! dim)
    $ error
    $ "narrowFromToByLength is trying to index tensor " <> show sz
    <> " at dimension " <> show dim
    <> " to produce a tensor of length " <> show (demote @length)
    <> " starting at " <> show start
    <> " but the tensor is too short."
  t' <- C.narrow__t666 t (demoteN @dimension) start (demoteN @length)
  pure $ Tensor t' a

-- * Operations over a single tensor

sum :: forall ty ki sz.
    (Num (TensorTyToHsC ty), Storable (TensorTyToHsC ty), SingI ty)
  => Tensor ty ki sz -> IO (Scalar ty ki)
sum (Tensor t _) = wrapTensorM (C.sum__ts t $ cScalarType' (demote @ty)) Nothing

prod :: forall ty ki sz.
    (Num (TensorTyToHsC ty), Storable (TensorTyToHsC ty), SingI ty)
  => Tensor ty ki sz -> IO (Scalar ty ki)
prod (Tensor t _) = wrapTensorM (C.prod__ts t $ cScalarType' (demote @ty)) Nothing

-- | Prod along a dimension
prodDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (RemoveDimension sz dimension))
        => Dimension dimension
        -> Tensor ty ki sz
        -> IO (Tensor ty ki (RemoveDimension sz dimension))
prodDim Dimension x@(Tensor t _) =
  wrapTensorM (C.prod__t6bs t (demoteN @dimension) (boolc False) $ cScalarType' (demote @ty)) Nothing

-- | Keep the dimension and replace it with size 1
prodKeepDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (ReplaceDimension sz dimension 1))
        => Dimension dimension
        -> Tensor ty ki sz
        -> IO (Tensor ty ki (ReplaceDimension sz dimension 1))
prodKeepDim Dimension x@(Tensor t _) =
  wrapTensorM (C.prod__t6bs t (demoteN @dimension) (boolc True) $ cScalarType' (demote @ty)) Nothing

pow :: forall ty ki sz.
      (Num (TensorTyToHsC ty), Storable (TensorTyToHsC ty), SingI ty)
    => Tensor ty ki sz -> Scalar ty ki -> IO (Tensor ty ki sz)
pow (Tensor t _) s = do
  s' <- toCScalarLike s =<< fromCScalarTensor s
  wrapTensorM (C.pow__ts t s') Nothing

abs :: (SingI ty, SingI ki, SingI sz) =>
       Tensor ty ki sz -> IO (Tensor ty ki sz)
abs x@(Tensor t _) = wrapTensorM (C.abs__t t) Nothing

-- * Reduction operations

-- | Find the maximum along a dimension.
maxDim :: forall (dimension :: Nat) ty ki sz.
      (SingI dimension, SingI (RemoveDimension sz dimension))
    => Tensor ty ki sz
    -> IO (Tensor ty    ki (RemoveDimension sz dimension)
         ,Tensor TLong ki (RemoveDimension sz dimension))
maxDim x@(Tensor t _) = do
  (tv, ti) <- C.max__t6b t (demoteN @dimension) (boolc False)
  pure $ (Tensor tv Nothing, Tensor ti Nothing)

-- | Find the minimum along a dimension.
minDim :: forall (dimension :: Nat) ty ki sz.
      (SingI dimension, SingI (RemoveDimension sz dimension))
    => Tensor ty ki sz
    -> IO (Tensor ty    ki (RemoveDimension sz dimension)
         ,Tensor TLong ki (RemoveDimension sz dimension))
minDim x@(Tensor t _) = do
  (tv, ti) <- C.min__t6b t (demoteN @dimension) (boolc False)
  pure $ (Tensor tv Nothing, Tensor ti Nothing)

-- | The mean of a tensor.
mean :: forall ty ki sz.
       (SingI ty, SingI ki, SingI sz) =>
       Tensor ty ki sz -> IO (Scalar ty ki)
mean x@(Tensor t _) = wrapTensorM (C.mean__ts t $ cScalarType' (demote @ty)) Nothing

-- | Mean along a dimension
meanDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (RemoveDimension sz dimension))
        => Dimension dimension
        -> Tensor ty ki sz
        -> IO (Tensor ty ki (RemoveDimension sz dimension))
meanDim Dimension x@(Tensor t _) =
  wrapTensorM (C.mean__tabs t (V.fromList [demoteN @dimension]) (boolc False) $ cScalarType' (demote @ty)) Nothing

-- | Keep the dimension and replace it with size 1
meanKeepDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (ReplaceDimension sz dimension 1))
        => Dimension dimension
        -> Tensor ty ki sz
        -> IO (Tensor ty ki (ReplaceDimension sz dimension 1))
meanKeepDim Dimension x@(Tensor t _) =
  wrapTensorM (C.mean__tabs t (V.fromList [demoteN @dimension]) (boolc True) $ cScalarType' (demote @ty)) Nothing

-- | The median of a tensor.
median :: forall ty ki sz.
       (SingI ty, SingI ki, SingI sz) =>
       Tensor ty ki sz -> IO (Scalar ty ki)
median x@(Tensor t _) = wrapTensorM (C.median__t t) Nothing

-- | Median along a dimension
medianDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (RemoveDimension sz dimension))
        => Dimension dimension
        -> Tensor ty ki sz
        -> IO (Tensor ty ki (RemoveDimension sz dimension)
             ,Tensor TLong ki (RemoveDimension sz dimension))
medianDim Dimension x@(Tensor t _) = do
  (m,i) <- C.median__t6b t (demoteN @dimension) (boolc False)
  pure (Tensor m Nothing, Tensor i Nothing)

-- | Keep the dimension and replace it with size 1
medianKeepDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (ReplaceDimension sz dimension 1))
        => Dimension dimension
        -> Tensor ty ki sz
        -> IO (Tensor ty ki (ReplaceDimension sz dimension 1)
             ,Tensor TLong ki (ReplaceDimension sz dimension 1))
medianKeepDim Dimension x@(Tensor t _) = do
  (m,i) <- C.median__t6b t (demoteN @dimension) (boolc True)
  pure (Tensor m Nothing, Tensor i Nothing)

-- | Mode along a dimension
modeDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (RemoveDimension sz dimension))
        => Dimension dimension
        -> Tensor ty ki sz
        -> IO (Tensor ty ki (RemoveDimension sz dimension)
             ,Tensor TLong ki (RemoveDimension sz dimension))
modeDim Dimension x@(Tensor t _) = do
  (m,i) <- C.mode__t6b t (demoteN @dimension) (boolc False)
  pure (Tensor m Nothing, Tensor i Nothing)

-- | Keep the dimension and replace it with size 1
modeKeepDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (ReplaceDimension sz dimension 1))
        => Dimension dimension
        -> Tensor ty ki sz
        -> IO (Tensor ty ki (ReplaceDimension sz dimension 1)
             ,Tensor TLong ki (ReplaceDimension sz dimension 1))
modeKeepDim Dimension x@(Tensor t _) = do
  (m,i) <- C.mode__t6b t (demoteN @dimension) (boolc True)
  pure (Tensor m Nothing, Tensor i Nothing)

data Norm = NormFrobenius
          | NormP Double

norm :: forall ty ki sz. Tensor ty ki sz -> Norm -> IO (Scalar ty ki)
norm (Tensor t _) NormFrobenius = wrapTensorM (C.frobenius_norm__t t) Nothing
norm (Tensor t _) (NormP p) = do
  s <- toCScalar @TDouble @ki (hsScalarToC p)
  wrapTensorM (C.norm__ts t s) Nothing

-- | The variance of a tensor
var :: forall ty ki sz.
       (SingI ty, SingI ki, SingI sz)
    => Tensor ty ki sz
    -> Bool
    -> IO (Scalar ty ki)
var x@(Tensor t _) unbiased = wrapTensorM (C.var__tb t $ boolc unbiased) Nothing

-- | Var along a dimension
varDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (RemoveDimension sz dimension))
        => Dimension dimension
        -> Tensor ty ki sz
        -> Bool
        -> IO (Tensor ty ki (RemoveDimension sz dimension))
varDim Dimension x@(Tensor t _) unbiased =
  wrapTensorM (C.var__tabb t (V.fromList [demoteN @dimension]) (boolc unbiased) (boolc False)) Nothing

-- | Keep the dimension and replace it with size 1
varKeepDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (ReplaceDimension sz dimension 1))
        => Dimension dimension
        -> Tensor ty ki sz
        -> Bool
        -> IO (Tensor ty ki (ReplaceDimension sz dimension 1))
varKeepDim Dimension x@(Tensor t _) unbiased =
  wrapTensorM (C.var__tabb t (V.fromList [demoteN @dimension]) (boolc unbiased) (boolc True)) Nothing

std :: forall ty ki sz.
       (SingI ty, SingI ki, SingI sz)
    => Tensor ty ki sz
    -> Bool
    -> IO (Scalar ty ki)
std x@(Tensor t _) unbiased = wrapTensorM (C.std__tb t $ boolc unbiased) Nothing

-- | Along a dimension
stdDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (RemoveDimension sz dimension))
        => Dimension dimension
        -> Tensor ty ki sz
        -> Bool
        -> IO (Tensor ty ki (RemoveDimension sz dimension))
stdDim Dimension x@(Tensor t _) unbiased =
  wrapTensorM (C.std__tabb t (V.fromList [demoteN @dimension]) (boolc unbiased) (boolc False)) Nothing

-- | Keep the dimension and replace it with size 1
stdKeepDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (ReplaceDimension sz dimension 1))
        => Dimension dimension
        -> Tensor ty ki sz
        -> Bool
        -> IO (Tensor ty ki (ReplaceDimension sz dimension 1))
stdKeepDim Dimension x@(Tensor t _) unbiased =
  wrapTensorM (C.std__tabb t (V.fromList [demoteN @dimension]) (boolc unbiased) (boolc True)) Nothing

stdMean :: forall ty ki sz.
       (SingI ty, SingI ki, SingI sz)
    => Tensor ty ki sz
    -> Bool
    -> IO (Scalar ty ki, Scalar ty ki)
stdMean x@(Tensor t _) unbiased = do
  (s,m) <- C.std_mean__tb t $ boolc unbiased
  pure (Tensor s Nothing, Tensor m Nothing)

-- | along a dimension
stdMeanDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (RemoveDimension sz dimension))
        => Dimension dimension
        -> Tensor ty ki sz
        -> Bool
        -> IO (Tensor ty ki (RemoveDimension sz dimension)
             ,Tensor ty ki (RemoveDimension sz dimension))
stdMeanDim Dimension x@(Tensor t _) unbiased = do
  (s,m) <- C.std_mean__tabb t (V.fromList [demoteN @dimension]) (boolc unbiased) (boolc False)
  pure (Tensor s Nothing, Tensor m Nothing)

-- | Keep the dimension and replace it with size 1
stdMeanKeepDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (ReplaceDimension sz dimension 1))
        => Dimension dimension
        -> Tensor ty ki sz
        -> Bool
        -> IO (Tensor ty ki (ReplaceDimension sz dimension 1)
             ,Tensor ty ki (ReplaceDimension sz dimension 1))
stdMeanKeepDim Dimension x@(Tensor t _) unbiased = do
  (s,m) <- C.std_mean__tabb t (V.fromList [demoteN @dimension]) (boolc unbiased) (boolc True)
  pure (Tensor s Nothing, Tensor m Nothing)

varMean :: forall ty ki sz.
       (SingI ty, SingI ki, SingI sz)
    => Tensor ty ki sz
    -> Bool
    -> IO (Scalar ty ki, Scalar ty ki)
varMean x@(Tensor t _) unbiased = do
  (s,m) <- C.var_mean__tb t $ boolc unbiased
  pure (Tensor s Nothing, Tensor m Nothing)

-- | along a dimension
varMeanDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (RemoveDimension sz dimension))
        => Dimension dimension
        -> Tensor ty ki sz
        -> Bool
        -> IO (Tensor ty ki (RemoveDimension sz dimension)
             ,Tensor ty ki (RemoveDimension sz dimension))
varMeanDim Dimension x@(Tensor t _) unbiased = do
  (s,m) <- C.var_mean__tabb t (V.fromList [demoteN @dimension]) (boolc unbiased) (boolc False)
  pure (Tensor s Nothing, Tensor m Nothing)

-- | Keep the dimension and replace it with size 1
varMeanKeepDim :: forall dimension ty ki sz.
          (SingI ty, SingI ki, SingI sz, SingI dimension
          ,SingI (ReplaceDimension sz dimension 1))
        => Dimension dimension
        -> Tensor ty ki sz
        -> Bool
        -> IO (Tensor ty ki (ReplaceDimension sz dimension 1)
             ,Tensor ty ki (ReplaceDimension sz dimension 1))
varMeanKeepDim Dimension (Tensor t _) unbiased = do
  (s,m) <- C.var_mean__tabb t (V.fromList [demoteN @dimension]) (boolc unbiased) (boolc True)
  pure (Tensor s Nothing, Tensor m Nothing)

-- * Comparison operations

eq :: forall ty ki sz sz'.
     (TensorConstraints ty ki sz, TensorConstraints 'TBool ki sz, SingI (SameOrScalar sz sz'))
   => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor 'TBool ki (SameOrScalar sz sz'))
eq t@(Tensor p _) t'@(Tensor p' _) = do
  p <- case (sing :: Sing sz, sing :: Sing sz') of
        (SNil, _) -> scalar t' t
        (_, SNil) -> scalar t t'
        _         -> generic t t'
  pure $ Tensor p Nothing
  where scalar :: forall ty ki sz. Tensor ty ki sz -> Scalar ty ki -> IO (ForeignPtr C.CTensor)
        scalar x@(Tensor xp _) y@(Tensor yp _) = do
          sc <- fromScalar y
          sp <- toCScalar @ty @ki $ hsScalarToC sc
          C.eq__ts xp sp
        generic x@(Tensor xp _) y@(Tensor yp _) = do
          C.ge__tt xp yp

neq :: forall ty ki sz sz'.
        (TensorConstraints ty ki sz, TensorConstraints 'TBool ki sz, SingI (SameOrScalar sz sz'))
      => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor 'TBool ki (SameOrScalar sz sz'))
neq t@(Tensor p _) t'@(Tensor p' _) = do
  p <- case (sing :: Sing sz, sing :: Sing sz') of
        (SNil, _) -> scalar t' t
        (_, SNil) -> scalar t t'
        _         -> generic t t'
  pure $ Tensor p Nothing
  where scalar :: forall ty ki sz. Tensor ty ki sz -> Scalar ty ki -> IO (ForeignPtr C.CTensor)
        scalar x@(Tensor xp _) y@(Tensor yp _) = do
          sc <- fromScalar y
          sp <- toCScalar @ty @ki $ hsScalarToC sc
          C.ne__ts xp sp
        generic x@(Tensor xp _) y@(Tensor yp _) = do
          C.ne__tt xp yp

gtq :: forall ty ki sz sz'.
        (TensorConstraints ty ki sz, TensorConstraints 'TBool ki sz, SingI (SameOrScalar sz sz'))
      => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor 'TBool ki (SameOrScalar sz sz'))
gtq t@(Tensor p _) t'@(Tensor p' _) = do
  p <- case (sing :: Sing sz, sing :: Sing sz') of
        (SNil, _) -> scalar' t' t
        (_, SNil) -> scalar t t'
        _         -> generic t t'
  pure $ Tensor p Nothing
  where scalar :: forall ty ki sz. Tensor ty ki sz -> Scalar ty ki -> IO (ForeignPtr C.CTensor)
        scalar x@(Tensor xp _) y@(Tensor yp _) = do
          sc <- fromScalar y
          sp <- toCScalar @ty @ki $ hsScalarToC sc
          C.ge__ts xp sp
        scalar' :: forall ty ki sz. Tensor ty ki sz -> Scalar ty ki -> IO (ForeignPtr C.CTensor)
        scalar' x@(Tensor xp _) y@(Tensor yp _) = do
          sc <- fromScalar y
          sp <- toCScalar @ty @ki $ hsScalarToC sc
          C.lt__ts xp sp
        generic x@(Tensor xp _) y@(Tensor yp _) = do
          C.ge__tt xp yp

ltq :: forall ty ki sz sz'.
        (TensorConstraints ty ki sz, TensorConstraints 'TBool ki sz, SingI (SameOrScalar sz sz'))
      => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor 'TBool ki (SameOrScalar sz sz'))
ltq t@(Tensor p _) t'@(Tensor p' _) = do
  p <- case (sing :: Sing sz, sing :: Sing sz') of
        (SNil, _) -> scalar' t' t
        (_, SNil) -> scalar t t'
        _         -> generic t t'
  pure $ Tensor p Nothing
  where scalar :: forall ty ki sz. Tensor ty ki sz -> Scalar ty ki -> IO (ForeignPtr C.CTensor)
        scalar x@(Tensor xp _) y@(Tensor yp _) = do
          sc <- fromScalar y
          sp <- toCScalar @ty @ki $ hsScalarToC sc
          C.le__ts xp sp
        scalar' :: forall ty ki sz. Tensor ty ki sz -> Scalar ty ki -> IO (ForeignPtr C.CTensor)
        scalar' x@(Tensor xp _) y@(Tensor yp _) = do
          sc <- fromScalar y
          sp <- toCScalar @ty @ki $ hsScalarToC sc
          C.gt__ts xp sp
        generic x@(Tensor xp _) y@(Tensor yp _) = do
          C.le__tt xp yp

gt :: forall ty ki sz sz'.
       (TensorConstraints ty ki sz, TensorConstraints 'TBool ki sz, SingI (SameOrScalar sz sz'))
     => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor 'TBool ki (SameOrScalar sz sz'))
gt t@(Tensor p _) t'@(Tensor p' _) = do
  p <- case (sing :: Sing sz, sing :: Sing sz') of
        (SNil, _) -> scalar' t' t
        (_, SNil) -> scalar t t'
        _         -> generic t t'
  pure $ Tensor p Nothing
  where scalar :: forall ty ki sz. Tensor ty ki sz -> Scalar ty ki -> IO (ForeignPtr C.CTensor)
        scalar x@(Tensor xp _) y@(Tensor yp _) = do
          sc <- fromScalar y
          sp <- toCScalar @ty @ki $ hsScalarToC sc
          C.gt__ts xp sp
        scalar' :: forall ty ki sz. Tensor ty ki sz -> Scalar ty ki -> IO (ForeignPtr C.CTensor)
        scalar' x@(Tensor xp _) y@(Tensor yp _) = do
          sc <- fromScalar y
          sp <- toCScalar @ty @ki $ hsScalarToC sc
          C.le__ts xp sp
        generic x@(Tensor xp _) y@(Tensor yp _) = do
          C.gt__tt xp yp

lt :: forall ty ki sz sz'.
       (TensorConstraints ty ki sz, TensorConstraints 'TBool ki sz, SingI (SameOrScalar sz sz'))
     => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor 'TBool ki (SameOrScalar sz sz'))
lt t@(Tensor p _) t'@(Tensor p' _) = do
  p <- case (sing :: Sing sz, sing :: Sing sz') of
        (SNil, _) -> scalar' t' t
        (_, SNil) -> scalar t t'
        _         -> generic t t'
  pure $ Tensor p Nothing
  where scalar :: forall ty ki sz. Tensor ty ki sz -> Scalar ty ki -> IO (ForeignPtr C.CTensor)
        scalar x@(Tensor xp _) y@(Tensor yp _) = do
          sc <- fromScalar y
          sp <- toCScalar @ty @ki $ hsScalarToC sc
          C.lt__ts xp sp
        scalar' :: forall ty ki sz. Tensor ty ki sz -> Scalar ty ki -> IO (ForeignPtr C.CTensor)
        scalar' x@(Tensor xp _) y@(Tensor yp _) = do
          sc <- fromScalar y
          sp <- toCScalar @ty @ki $ hsScalarToC sc
          C.ge__ts xp sp
        generic x@(Tensor xp _) y@(Tensor yp _) = do
          C.lt__tt xp yp

allClose :: (Real rtol, Real atol) => Tensor ty ki sz -> Tensor ty ki sz -> rtol -> atol -> Bool -> IO Bool
allClose t t' rtol atol naneq = cbool <$> C.allclose__ttddb (tensorPtr t) (tensorPtr t') (realToFrac rtol) (realToFrac atol) (boolc naneq)

allClose' :: Tensor ty ki sz -> Tensor ty ki sz -> IO Bool
allClose' t t' = allClose t t' (realToFrac 1e-05) (realToFrac 1e-08) False

-- | Reduce the tensor to a scalar, its maximum.
max :: (SingI ty, SingI ki, SingI sz) =>
       Tensor ty ki sz -> IO (Scalar ty ki)
max x@(Tensor t _) = wrapTensorM (C.max__t t) Nothing

-- | Reduce the tensor to a scalar, its minimum.
min :: (SingI ty, SingI ki, SingI sz) =>
       Tensor ty ki sz -> IO (Scalar ty ki)
min x@(Tensor t _) = wrapTensorM (C.min__t t) Nothing

-- * Non-linear activation functions

relu :: Tensor ty ki sz -> IO (Tensor ty ki sz)
relu x@(Tensor t _) = wrapTensorM (C.relu__t t) Nothing

threshold :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> TensorTyToHs ty -> IO (Tensor ty ki sz)
threshold x@(Tensor t _) threshold value = do
  x <- toCScalar @ty @ki (hsScalarToC threshold)
  y <- toCScalar @ty @ki (hsScalarToC value)
  wrapTensorM (C.threshold__tss t x y) Nothing

hardtanh :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> TensorTyToHs ty -> IO (Tensor ty ki sz)
hardtanh x@(Tensor t _) min max = do
  x <- toCScalar @ty @ki (hsScalarToC min)
  y <- toCScalar @ty @ki (hsScalarToC max)
  wrapTensorM (C.hardtanh__tss t x y) Nothing

relu6 :: Num (TensorTyToHs ty) => Tensor ty ki sz -> IO (Tensor ty ki sz)
relu6 t = hardtanh t 0 6

elu :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> TensorTyToHs ty -> IO (Tensor ty ki sz)
elu x@(Tensor t _) alpha scale = do
  x <- toCScalar @ty @ki (hsScalarToC alpha)
  y <- toCScalar @ty @ki (hsScalarToC scale)
  s <- toCScalar @ty @ki 1
  wrapTensorM (C.elu__tsss t x y s) Nothing

selu :: Tensor ty ki sz -> IO (Tensor ty ki sz)
selu x@(Tensor t _) = wrapTensorM (C.selu__t t) Nothing

celu :: forall ty ki sz. TensorTyToHs ty -> Tensor ty ki sz -> IO (Tensor ty ki sz)
celu alpha x@(Tensor t _) = do
  x <- toCScalar @ty @ki (hsScalarToC alpha)
  wrapTensorM (C.celu__ts t x) Nothing

leakyRelu :: forall ty ki sz. TensorTyToHs ty -> Tensor ty ki sz -> IO (Tensor ty ki sz)
leakyRelu negativeSlope x@(Tensor t _) = do
  x <- toCScalar @ty @ki (hsScalarToC negativeSlope)
  wrapTensorM (C.leaky_relu__ts t x) Nothing

rrelu :: forall ty ki sz. TensorTyToHs ty -> TensorTyToHs ty -> DataPurpose -> Tensor ty ki sz -> IO (Tensor ty ki sz)
rrelu lower upper dataPurpose x@(Tensor t _) = do
  x <- toCScalar @ty @ki (hsScalarToC lower)
  y <- toCScalar @ty @ki (hsScalarToC upper)
  gen <- generatorFor (demote @ki)
  wrapTensorM (C.rrelu__tssbg t x y (boolc (dataPurpose == Train)) gen) Nothing

-- TODO Do we want dim to be statically checked?
glu :: forall ty ki sz. Tensor ty ki sz -> Int64 -> IO (Tensor ty ki sz)
glu t@(Tensor p _) dim = wrapTensorM (C.glu__t6 p dim) Nothing

gelu :: forall ty ki sz. Tensor ty ki sz -> IO (Tensor ty ki sz)
gelu t@(Tensor p _) = wrapTensorM (C.gelu__t p) Nothing

logSigmoid :: forall ty ki sz. Tensor ty ki sz -> IO (Tensor ty ki sz)
logSigmoid t@(Tensor p _) = wrapTensorM (C.log_sigmoid__t p) Nothing

-- TODO argument here should always be float? Does this even make sense wiht
-- anything other than floats?
hardshrink :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
hardshrink x@(Tensor t _) lambda = do
  x <- toCScalar @ty @ki (hsScalarToC lambda)
  wrapTensorM (C.hardshrink__ts t x) Nothing

softsign :: forall ty ki sz. (TensorConstraints ty ki sz) => Tensor ty ki sz -> IO (Tensor ty ki sz)
softsign t = do
  one <- toScalar @ty @ki 1
  absT <- Torch.Tensor.abs t
  Torch.Tensor.div t =<< add absT one

softplus :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> TensorTyToHs ty -> IO (Tensor ty ki sz)
softplus x@(Tensor t _) beta threshold = do
  x <- toCScalar @ty @ki (hsScalarToC beta)
  y <- toCScalar @ty @ki (hsScalarToC threshold)
  wrapTensorM (C.softplus__tss t x y) Nothing

softmax :: forall ty ki sz. Tensor ty ki sz -> Int64 -> IO (Tensor ty ki sz)
softmax t@(Tensor p _) dim = wrapTensorM (C.softmax__t6s p dim $ cScalarType' (demote @ty)) Nothing

logSoftmax :: forall ty ki sz. Tensor ty ki sz -> Int64 -> IO (Tensor ty ki sz)
logSoftmax t@(Tensor p _) dim = wrapTensorM (C.log_softmax__t6s p dim $ cScalarType' (demote @ty)) Nothing

softmin :: forall ty ki sz. (TensorConstraints ty ki sz) => Tensor ty ki sz -> Int64 -> IO (Tensor ty ki sz)
softmin t dim = do
  zero <- toScalar @ty @ki 0
  t' <- sub zero t
  softmax t' dim

-- TODO argument here should always be float? Does this even make sense wiht
-- anything other than floats?
softshrink :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
softshrink x@(Tensor t _) lambda = do
  x <- toCScalar @ty @ki (hsScalarToC lambda)
  wrapTensorM (C.softshrink__ts t x) Nothing

sigmoid :: Tensor ty ki sz -> IO (Tensor ty ki sz)
sigmoid x@(Tensor t _) = wrapTensorM (C.sigmoid__t t) Nothing

-- * Normalization layers

-- | Both the weights, if any, and the state are updated inplace
batchNorm2d_ :: (TensorConstraints ty ki [szN, szC, szH, szW]
               ,Fractional (TensorTyToHsC ty), SingI szC, IsFloatTy ty ~ 'True)
             => BatchNormState ty ki '[szC]
             -> Maybe (AffineParam ty ki '[szC])
             -> BNMomentum Double
             -> BNEpsilon Double
             -> DataPurpose
             -> Tensor ty ki '[szN, szC, szH, szW]
             -> IO (Tensor ty ki '[szN, szC, szH, szW])
batchNorm2d_ (BatchNormState bn) affineParams (BNMomentum momentum) (BNEpsilon epsilon) dp x@(Tensor t a) = do
  (w,b) <- case affineParams of
            (Just (AffineParam (Tensor tw _) (Just (Tensor tb _)))) ->
              pure (tw,tb)
            (Just (AffineParam (Tensor tw _) Nothing)) -> do
              tb <- C.undefinedTensor
              pure (tw,tb)
            Nothing -> do
              ten <- C.undefinedTensor
              pure (ten,ten)
  -- TODO The last argument is cudnn_enabled. How do we figure this out?
  bn' <- readIORef bn
  (mean, var)<- case (bnMean bn', bnVariance bn') of
                 (Nothing, Nothing) -> do
                   mean <- zeros
                   var <- zeros
                   writeIORef bn (BatchNormData (Just mean) (Just var))
                   pure (mean,var)
                 (Just mean, Just var) -> pure (mean,var)
                 _ -> error "This is impossible and a bug!"
  r <- C.batch_norm__tttttbddb t w b (tensorPtr mean) (tensorPtr var)
                              (boolc (dp == Train)) (CDouble momentum)
                              (CDouble epsilon) (boolc True)
  pure $ Tensor r a

-- * Linear layers

linear :: forall inF' outF' inF outF ty ki sz1.
      (SingI outF, SingI inF, TensorConstraints ty ki sz1
      , InFeatures inF ~ inF', OutFeatures outF ~ outF'
      , Last sz1 ~ inF, SingI (Head sz1)
      , SingI (BroadcastMatrices sz1 '[inF, outF])
      , SingI (BroadcastSizes (BroadcastMatrices sz1 '[inF, outF]) '[outF])
      , SingI (ReplaceLast sz1 outF)
      ) =>
       InFeatures inF
    -> OutFeatures outF
    -> LinearParam ty ki inF outF
    -> Tensor ty ki sz1
    -> IO (Tensor ty ki (ReplaceLast sz1 outF))
linear InFeatures OutFeatures (LinearParam tw@(Tensor w _) (Just tb@(Tensor b _))) tin@(Tensor i _)  = do
  s <- toCScalar @ty @ki 1
  w' <- C.t__t (tensorPtr tw)
  b' <- expandInC b (demoteNv @'[outF]) (demoteNv @'[Head sz1, outF]) True
  p <- C.addmm__tttss b' i w' s s
  pure $ Tensor p Nothing
linear InFeatures OutFeatures (LinearParam tw bias) tin  = do
  r <- matmul tin =<< t tw
  case bias of
    -- TODO This requires unsafeSize
    -- I don't know how to prove to GHC that broadcasting where the inner
    -- dimensions are the same just removes them and then appends the sizes.
    Nothing   -> pure $ unsafeSize r
    (Just tb) -> unsafeSize <$> add r tb

dropout :: (TensorConstraints ty ki sz
          ,Fractional (TensorTyToHs ty), SingI sz)
        => Double -> DataPurpose -> Tensor ty ki sz
        -> IO (Tensor ty ki sz)
dropout rate dataPurpose (Tensor t _) =
  wrapTensorM (C.dropout__tdb t (coerce rate) (boolc (dataPurpose == Train))) Nothing

dropout_ :: (TensorConstraints ty ki sz
           ,Fractional (TensorTyToHs ty), SingI sz)
         => Double -> DataPurpose -> Tensor ty ki sz
         -> IO (Tensor ty ki sz)
dropout_ rate dataPurpose (Tensor t a) =
  wrapTensorM (C.dropout___tdb t (coerce rate) (boolc (dataPurpose == Train))) a

-- * Recurrent layers

-- | Internal, this is the basic function shared by all. This is not safe.  This
-- is also abused by other code, in particular the initialization code that
-- needs to compact weights in order to make cuda happy.
genericRNN :: forall ty ki gateSize inF hiddenF nrLayers isBidirectional batchFirst cret hiddenTy.
          (Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
          ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty)
          ,SingI ty, SingI ki, SingI nrLayers, SingI isBidirectional)
        => ( ForeignPtr C.CTensor -> hiddenTy
          -> Vector (Ptr C.CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO cret)
        -> InFeatures inF
        -> HiddenFeatures hiddenF
        -> NrLayers nrLayers
        -> IsBidirectional isBidirectional
        -> Double
        -> DataPurpose
        -> Bool
        -> GenericRNNParam ty ki gateSize inF hiddenF nrLayers isBidirectional batchFirst
        -> hiddenTy -- ^ hidden state
        -> ForeignPtr C.CTensor -- ^ input sequence
        -> IO cret
genericRNN fn InFeatures HiddenFeatures NrLayers IsBidirectional dropout dataPurpose batchFirst
        GenericRNNParam{..} statep inp = do
  let lWihN = (map (tensorPtr.debuggingVerifyShape "1") $ toListDirection grnnParamWih0)
            ++ (map (tensorPtr.debuggingVerifyShape "2") $ concat $ VB.toList $ VB.map toListDirection grnnParamWihN1)
  let lWhhN = map (tensorPtr.debuggingVerifyShape "3") $ concat $ VB.toList $ VB.map toListDirection grnnParamWhhN
  let mlBihN = case grnnParamBihN of
                 Nothing -> Nothing
                 Just x  -> Just $ map (tensorPtr.debuggingVerifyShape "4") $ concat $ VB.toList $ VB.map toListDirection x
  let mlBhhN = case grnnParamBhhN of
                 Nothing -> Nothing
                 Just x  -> Just $ map (tensorPtr.debuggingVerifyShape "5") $ concat $ VB.toList $ VB.map toListDirection x
  let params = case (mlBihN, mlBhhN) of
                 (Nothing, Nothing)       -> concat $ zipWith (\x y -> [x,y]) lWihN lWhhN
                 (Just lBihN, Just lBhhN) -> concat $ zipWith4 (\x y z w -> [x,y,z,w]) lWihN lWhhN lBihN lBhhN
  withForeignPtrs params
    (\params' -> do
        fn inp statep (V.fromList params') (boolc $ isJust grnnParamBihN) (demoteN @nrLayers) (coerce dropout)
           (boolc $ dataPurpose == Train) (boolc $ demote @isBidirectional == True) (boolc batchFirst))

rnnRelu :: forall seqLen batch ty ki inF hiddenF nrLayers isBidirectional.
          (Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
          ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty)
          ,SingI ty, SingI ki, SingI nrLayers, SingI isBidirectional
          ,SingI '[seqLen, batch, NrOfRNNDirections isBidirectional TL.* hiddenF]
          ,SingI (NrOfRNNDirections isBidirectional TL.* nrLayers)
          ,SingI batch, SingI hiddenF)
        => InFeatures inF
        -> HiddenFeatures hiddenF
        -> NrLayers nrLayers
        -> IsBidirectional isBidirectional
        -> Double
        -> DataPurpose
        -> RNNParams ty ki inF hiddenF nrLayers isBidirectional False
        -> Tensor ty ki '[batch, NrOfRNNDirections isBidirectional TL.* nrLayers, hiddenF] -- ^ hidden state
        -> Tensor ty ki '[seqLen, batch, inF] -- ^ input sequence
        -> IO (Tensor ty ki '[seqLen, batch, NrOfRNNDirections isBidirectional TL.* hiddenF]
             ,Tensor ty ki '[batch, NrOfRNNDirections isBidirectional TL.* nrLayers, hiddenF]) -- ^ activations & new hidden state
rnnRelu inFeat hiddenFeat layers bidi dropout dataPurpose (RNNParams params) (Tensor statep _) (Tensor inp _) = do
  (act, hs) <- genericRNN C.rnn_relu__ttlb6dbbb inFeat hiddenFeat layers bidi dropout dataPurpose False params statep inp
  pure (Tensor act Nothing, Tensor hs Nothing)

rnnReluBatchFirst :: forall seqLen batch ty ki inF hiddenF nrLayers isBidirectional.
          (Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
          ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty)
          ,SingI ty, SingI ki, SingI nrLayers, SingI isBidirectional
          ,SingI '[batch, seqLen, NrOfRNNDirections isBidirectional TL.* hiddenF]
          ,SingI (NrOfRNNDirections isBidirectional TL.* nrLayers)
          ,SingI batch, SingI hiddenF)
        => InFeatures inF
        -> HiddenFeatures hiddenF
        -> NrLayers nrLayers
        -> IsBidirectional isBidirectional
        -> Double
        -> DataPurpose
        -> RNNParams ty ki inF hiddenF nrLayers isBidirectional True
        -> Tensor ty ki '[NrOfRNNDirections isBidirectional TL.* nrLayers, batch, hiddenF] -- ^ hidden state
        -> Tensor ty ki '[batch, seqLen, inF] -- ^ input sequence
        -> IO (Tensor ty ki '[batch, seqLen, NrOfRNNDirections isBidirectional TL.* hiddenF]
             ,Tensor ty ki '[NrOfRNNDirections isBidirectional TL.* nrLayers, batch, hiddenF]) -- ^ activations & new hidden state
rnnReluBatchFirst inFeat hiddenFeat layers bidi dropout dataPurpose (RNNParams params) (Tensor statep _) (Tensor inp _) = do
  (act, hs) <- genericRNN C.rnn_relu__ttlb6dbbb inFeat hiddenFeat layers bidi dropout dataPurpose True params statep inp
  pure (Tensor act Nothing, Tensor hs Nothing)

rnnTanh :: forall seqLen batch ty ki inF hiddenF nrLayers isBidirectional.
          (Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
          ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty)
          ,SingI ty, SingI ki, SingI nrLayers, SingI isBidirectional
          ,SingI '[seqLen, batch, NrOfRNNDirections isBidirectional TL.* hiddenF]
          ,SingI (NrOfRNNDirections isBidirectional TL.* nrLayers)
          ,SingI batch, SingI hiddenF)
        => InFeatures inF
        -> HiddenFeatures hiddenF
        -> NrLayers nrLayers
        -> IsBidirectional isBidirectional
        -> Double
        -> DataPurpose
        -> RNNParams ty ki inF hiddenF nrLayers isBidirectional False
        -> Tensor ty ki '[batch, NrOfRNNDirections isBidirectional TL.* nrLayers, hiddenF] -- ^ hidden state
        -> Tensor ty ki '[seqLen, batch, inF] -- ^ input sequence
        -> IO (Tensor ty ki '[seqLen, batch, NrOfRNNDirections isBidirectional TL.* hiddenF]
             ,Tensor ty ki '[batch, NrOfRNNDirections isBidirectional TL.* nrLayers, hiddenF]) -- ^ activations & new hidden state
rnnTanh inFeat hiddenFeat layers bidi dropout dataPurpose (RNNParams params) (Tensor statep _) (Tensor inp _) = do
  (act, hs) <- genericRNN C.rnn_tanh__ttlb6dbbb inFeat hiddenFeat layers bidi dropout dataPurpose False params statep inp
  pure (Tensor act Nothing, Tensor hs Nothing)

rnnTanhBatchFirst :: forall seqLen batch ty ki inF hiddenF nrLayers isBidirectional.
          (Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
          ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty)
          ,SingI ty, SingI ki, SingI nrLayers, SingI isBidirectional
          ,SingI '[batch, seqLen, NrOfRNNDirections isBidirectional TL.* hiddenF]
          ,SingI (NrOfRNNDirections isBidirectional TL.* nrLayers)
          ,SingI batch, SingI hiddenF)
        => InFeatures inF
        -> HiddenFeatures hiddenF
        -> NrLayers nrLayers
        -> IsBidirectional isBidirectional
        -> Double
        -> DataPurpose
        -> RNNParams ty ki inF hiddenF nrLayers isBidirectional True
        -> Tensor ty ki '[NrOfRNNDirections isBidirectional TL.* nrLayers, batch, hiddenF] -- ^ hidden state
        -> Tensor ty ki '[batch, seqLen, inF] -- ^ input sequence
        -> IO (Tensor ty ki '[batch, seqLen, NrOfRNNDirections isBidirectional TL.* hiddenF]
             ,Tensor ty ki '[NrOfRNNDirections isBidirectional TL.* nrLayers, batch, hiddenF]) -- ^ activations & new hidden state
rnnTanhBatchFirst inFeat hiddenFeat layers bidi dropout dataPurpose (RNNParams params) (Tensor statep _) (Tensor inp _) = do
  (act, hs) <- genericRNN C.rnn_tanh__ttlb6dbbb inFeat hiddenFeat layers bidi dropout dataPurpose True params statep inp
  pure (Tensor act Nothing, Tensor hs Nothing)

gru :: forall seqLen batch ty ki inF hiddenF nrLayers isBidirectional.
          (Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
          ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty)
          ,SingI ty, SingI ki, SingI nrLayers, SingI isBidirectional
          ,SingI '[seqLen, batch, NrOfRNNDirections isBidirectional TL.* hiddenF]
          ,SingI (NrOfRNNDirections isBidirectional TL.* nrLayers)
          ,SingI batch, SingI hiddenF)
        => InFeatures inF
        -> HiddenFeatures hiddenF
        -> NrLayers nrLayers
        -> IsBidirectional isBidirectional
        -> Double
        -> DataPurpose
        -> GRUParams ty ki inF hiddenF nrLayers isBidirectional False
        -> Tensor ty ki '[batch, NrOfRNNDirections isBidirectional TL.* nrLayers, hiddenF] -- ^ hidden state
        -> Tensor ty ki '[seqLen, batch, inF] -- ^ input sequence
        -> IO (Tensor ty ki '[seqLen, batch, NrOfRNNDirections isBidirectional TL.* hiddenF]
             ,Tensor ty ki '[batch, NrOfRNNDirections isBidirectional TL.* nrLayers, hiddenF]) -- ^ activations & new hidden state
gru inFeat hiddenFeat layers bidi dropout dataPurpose (GRUParams params) (Tensor statep _) (Tensor inp _) = do
  (act, hs) <- genericRNN C.gru__ttlb6dbbb inFeat hiddenFeat layers bidi dropout dataPurpose False params statep inp
  pure (Tensor act Nothing, Tensor hs Nothing)

gruBatchFirst :: forall seqLen batch ty ki inF hiddenF nrLayers isBidirectional.
          (Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
          ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty)
          ,SingI ty, SingI ki, SingI nrLayers, SingI isBidirectional
          ,SingI '[batch, seqLen, NrOfRNNDirections isBidirectional TL.* hiddenF]
          ,SingI (NrOfRNNDirections isBidirectional TL.* nrLayers)
          ,SingI batch, SingI hiddenF)
        => InFeatures inF
        -> HiddenFeatures hiddenF
        -> NrLayers nrLayers
        -> IsBidirectional isBidirectional
        -> Double
        -> DataPurpose
        -> GRUParams ty ki inF hiddenF nrLayers isBidirectional True
        -> Tensor ty ki '[NrOfRNNDirections isBidirectional TL.* nrLayers, batch, hiddenF] -- ^ hidden state
        -> Tensor ty ki '[batch, seqLen, inF] -- ^ input sequence
        -> IO (Tensor ty ki '[batch, seqLen, NrOfRNNDirections isBidirectional TL.* hiddenF]
             ,Tensor ty ki '[NrOfRNNDirections isBidirectional TL.* nrLayers, batch, hiddenF]) -- ^ activations & new hidden state
gruBatchFirst inFeat hiddenFeat layers bidi dropout dataPurpose (GRUParams params) (Tensor statep _) (Tensor inp _) = do
  (act, hs) <- genericRNN C.gru__ttlb6dbbb inFeat hiddenFeat layers bidi dropout dataPurpose True params statep inp
  pure (Tensor act Nothing, Tensor hs Nothing)

lstm :: forall seqLen batch ty ki inF hiddenF nrLayers isBidirectional.
          (Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
          ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty)
          ,SingI ty, SingI ki, SingI nrLayers, SingI isBidirectional
          ,SingI '[seqLen, batch, NrOfRNNDirections isBidirectional TL.* hiddenF]
          ,SingI (NrOfRNNDirections isBidirectional TL.* nrLayers)
          ,SingI batch, SingI hiddenF)
        => InFeatures inF
        -> HiddenFeatures hiddenF
        -> NrLayers nrLayers
        -> IsBidirectional isBidirectional
        -> Double
        -> DataPurpose
        -> LSTMParams ty ki inF hiddenF nrLayers isBidirectional False
        -- TODO This makes zero sense to me.
        -- The documentation at https://pytorch.org/docs/stable/nn.html#
        -- indicates that the order should be (num_layers * num_directions, batch, hidden_size)
        -- and not (num_layers * num_directions, batch, hidden_size)
        -- Note how the order of the input is different!
        -- Internally, something in pytorch must be transpoing these states
        -- This applies to all of the RNNs!
        -> LSTMState ty ki batch isBidirectional nrLayers hiddenF -- ^ hidden & cell state
        -> Tensor ty ki '[seqLen, batch, inF] -- ^ input sequence
        -> IO (Tensor ty ki '[seqLen, batch, NrOfRNNDirections isBidirectional TL.* hiddenF]
             ,LSTMState ty ki batch isBidirectional nrLayers hiddenF) -- ^ activations & new hidden & cell states
lstm inFeat hiddenFeat layers bidi dropout dataPurpose (LSTMParams params) (LSTMState (Tensor hiddenState _) (Tensor cellState _)) (Tensor inp _) = do
  (act, hs, cs) <-
    withForeignPtrs [hiddenState, cellState]
       (\state -> genericRNN C.lstm__tllb6dbbb inFeat hiddenFeat layers bidi dropout dataPurpose False params (V.fromList state) inp)
  pure (Tensor act Nothing, LSTMState (Tensor hs Nothing) (Tensor cs Nothing))

lstmBatchFirst :: forall seqLen batch ty ki inF hiddenF nrLayers isBidirectional.
          (Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
          ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty)
          ,SingI ty, SingI ki, SingI nrLayers, SingI isBidirectional
          ,SingI '[batch, seqLen, NrOfRNNDirections isBidirectional TL.* hiddenF]
          ,SingI (NrOfRNNDirections isBidirectional TL.* nrLayers)
          ,SingI batch, SingI hiddenF)
        => InFeatures inF
        -> HiddenFeatures hiddenF
        -> NrLayers nrLayers
        -> IsBidirectional isBidirectional
        -> Double
        -> DataPurpose
        -> LSTMParams ty ki inF hiddenF nrLayers isBidirectional True
        -- TODO This makes zero sense to me.
        -- The documentation at https://pytorch.org/docs/stable/nn.html#
        -- indicates that the order should be (num_layers * num_directions, batch, hidden_size)
        -- and not (num_layers * num_directions, batch, hidden_size)
        -- Note how the order of the input is different!
        -- Internally, something in pytorch must be transpoing these states
        -- This applies to all of the RNNs!
        -> LSTMStateBatchFirst ty ki batch isBidirectional nrLayers hiddenF -- ^ hidden & cell state
        -> Tensor ty ki '[batch, seqLen, inF] -- ^ input sequence
        -> IO (Tensor ty ki '[batch, seqLen, NrOfRNNDirections isBidirectional TL.* hiddenF]
             ,LSTMStateBatchFirst ty ki batch isBidirectional nrLayers hiddenF) -- ^ activations & new hidden & cell states
lstmBatchFirst inFeat hiddenFeat layers isBidirectional dropout dataPurpose (LSTMParams params)
               (LSTMStateBatchFirst (Tensor hiddenState _) (Tensor cellState _)) (Tensor inp _) = do
  (act, hs, cs) <-
    withForeignPtrs [hiddenState, cellState]
       (\state -> genericRNN C.lstm__tllb6dbbb inFeat hiddenFeat layers isBidirectional dropout dataPurpose True params (V.fromList state) inp)
  pure (Tensor act Nothing, LSTMStateBatchFirst (Tensor hs Nothing) (Tensor cs Nothing))

rnnReluCell :: forall ty ki nr inF hiddenF.
              (Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
              ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty)
              ,SingI ty, SingI ki, SingI nr)
            => InFeatures inF
            -> HiddenFeatures hiddenF
            -> RNNCellParam ty ki inF hiddenF
            -> Tensor ty ki '[nr, inF]
            -> Tensor ty ki '[nr, hiddenF]
            -> IO (Tensor ty ki '[nr, hiddenF])
rnnReluCell InFeatures HiddenFeatures
            (RNNCellParam twih@(Tensor wih _) (Just tbih@(Tensor bih _))
                          twhh@(Tensor whh _) (Just tbhh@(Tensor bhh  _)))
            tin@(Tensor inp _) tstate@(Tensor statep _) = do
  statep' <- C.rnn_relu_cell__tttttt inp statep wih whh bih bhh
  pure (Tensor statep' Nothing)
rnnReluCell InFeatures HiddenFeatures
            (RNNCellParam twih@(Tensor wih _) Nothing
                          twhh@(Tensor whh _) Nothing)
            tin@(Tensor inp _) tstate@(Tensor statep _) = do
  statep' <- C.rnn_relu_cell__tttttt inp statep wih whh (unsafePerformIO C.undefinedTensor) (unsafePerformIO C.undefinedTensor)
  pure (Tensor statep' Nothing)

rnnTanhCell :: forall ty ki nr inF hiddenF.
              (Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
              ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty)
              ,SingI ty, SingI ki, SingI nr)
            => InFeatures inF
            -> HiddenFeatures hiddenF
            -> RNNCellParam ty ki inF hiddenF
            -> Tensor ty ki '[nr, inF]
            -> Tensor ty ki '[nr, hiddenF]
            -> IO (Tensor ty ki '[nr, hiddenF])
rnnTanhCell InFeatures HiddenFeatures
            (RNNCellParam twih@(Tensor wih _) (Just tbih@(Tensor bih _))
                          twhh@(Tensor whh _) (Just tbhh@(Tensor bhh  _)))
            tin@(Tensor inp _) tstate@(Tensor statep _) = do
  statep' <- C.rnn_tanh_cell__tttttt inp statep wih whh bih bhh
  pure (Tensor statep' Nothing)
rnnTanhCell InFeatures HiddenFeatures
            (RNNCellParam twih@(Tensor wih _) Nothing
                          twhh@(Tensor whh _) Nothing)
            tin@(Tensor inp _) tstate@(Tensor statep _) = do
  statep' <- C.rnn_tanh_cell__tttttt inp statep wih whh (unsafePerformIO C.undefinedTensor) (unsafePerformIO C.undefinedTensor)
  pure (Tensor statep' Nothing)

-- * Single-tensor mathematical operations

sin :: Tensor ty ki sz -> IO (Tensor ty ki sz)
sin x@(Tensor t _) = wrapTensorM (C.sin__t t) Nothing

sinh :: Tensor ty ki sz -> IO (Tensor ty ki sz)
sinh x@(Tensor t _) = wrapTensorM (C.sinh__t t) Nothing

asin :: Tensor ty ki sz -> IO (Tensor ty ki sz)
asin x@(Tensor t _) = wrapTensorM (C.asin__t t) Nothing

cos :: Tensor ty ki sz -> IO (Tensor ty ki sz)
cos x@(Tensor t _) = wrapTensorM (C.cos__t t) Nothing

cosh :: Tensor ty ki sz -> IO (Tensor ty ki sz)
cosh x@(Tensor t _) = wrapTensorM (C.cosh__t t) Nothing

acos :: Tensor ty ki sz -> IO (Tensor ty ki sz)
acos x@(Tensor t _) = wrapTensorM (C.acos__t t) Nothing

tan :: Tensor ty ki sz -> IO (Tensor ty ki sz)
tan x@(Tensor t _) = wrapTensorM (C.tan__t t) Nothing

tanh :: Tensor ty ki sz -> IO (Tensor ty ki sz)
tanh x@(Tensor t _) = wrapTensorM (C.tanh__t t) Nothing

atan :: Tensor ty ki sz -> IO (Tensor ty ki sz)
atan x@(Tensor t _) = wrapTensorM (C.atan__t t) Nothing

ceil :: Tensor ty ki sz -> IO (Tensor ty ki sz)
ceil x@(Tensor t _) = wrapTensorM (C.ceil__t t) Nothing

floor :: Tensor ty ki sz -> IO (Tensor ty ki sz)
floor x@(Tensor t _) = wrapTensorM (C.floor__t t) Nothing

clamp :: forall ty ki sz. TensorTyToHs ty -> TensorTyToHs ty -> Tensor ty ki sz -> IO (Tensor ty ki sz)
clamp lower upper  x@(Tensor t _) = do
  l <- toCScalar @ty @ki (hsScalarToC lower)
  u <- toCScalar @ty @ki (hsScalarToC upper)
  wrapTensorM (C.clamp__tss t l u) Nothing

clampMax :: forall ty ki sz. TensorTyToHs ty -> Tensor ty ki sz -> IO (Tensor ty ki sz)
clampMax upper x@(Tensor t _) = do
  u <- toCScalar @ty @ki (hsScalarToC upper)
  wrapTensorM (C.clamp_max__ts t u) Nothing

clampMin :: forall ty ki sz. TensorTyToHs ty -> Tensor ty ki sz -> IO (Tensor ty ki sz)
clampMin lower x@(Tensor t _) = do
  l <- toCScalar @ty @ki (hsScalarToC lower)
  wrapTensorM (C.clamp_min__ts t l) Nothing

digamma :: Tensor ty ki sz -> IO (Tensor ty ki sz)
digamma x@(Tensor t _) = wrapTensorM (C.digamma__t t) Nothing

erf :: Tensor ty ki sz -> IO (Tensor ty ki sz)
erf x@(Tensor t _) = wrapTensorM (C.erf__t t) Nothing

erfc :: Tensor ty ki sz -> IO (Tensor ty ki sz)
erfc x@(Tensor t _) = wrapTensorM (C.erfc__t t) Nothing

erfinv :: Tensor ty ki sz -> IO (Tensor ty ki sz)
erfinv x@(Tensor t _) = wrapTensorM (C.erfinv__t t) Nothing

exp :: Tensor ty ki sz -> IO (Tensor ty ki sz)
exp x@(Tensor t _) = wrapTensorM (C.exp__t t) Nothing

expm1 :: Tensor ty ki sz -> IO (Tensor ty ki sz)
expm1 x@(Tensor t _) = wrapTensorM (C.expm1__t t) Nothing

fmod :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
fmod x@(Tensor t _) div = do
  d <- toCScalar @ty @ki (hsScalarToC div)
  wrapTensorM (C.fmod__ts t d) Nothing

frac :: Tensor ty ki sz -> IO (Tensor ty ki sz)
frac x@(Tensor t _) = wrapTensorM (C.frac__t t) Nothing

log :: Tensor ty ki sz -> IO (Tensor ty ki sz)
log x@(Tensor t _) = wrapTensorM (C.log__t t) Nothing

log10 :: Tensor ty ki sz -> IO (Tensor ty ki sz)
log10 x@(Tensor t _) = wrapTensorM (C.log10__t t) Nothing

log1p :: Tensor ty ki sz -> IO (Tensor ty ki sz)
log1p x@(Tensor t _) = wrapTensorM (C.log1p__t t) Nothing

log2 :: Tensor ty ki sz -> IO (Tensor ty ki sz)
log2 x@(Tensor t _) = wrapTensorM (C.log2__t t) Nothing

mvlgamma :: forall ty ki sz. Tensor ty ki sz -> Int64 -> IO (Tensor ty ki sz)
mvlgamma t@(Tensor p _) dim = wrapTensorM (C.mvlgamma__t6 p dim) Nothing

neg :: Tensor ty ki sz -> IO (Tensor ty ki sz)
neg x@(Tensor t _) = wrapTensorM (C.neg__t t) Nothing

reciprocal :: Tensor ty ki sz -> IO (Tensor ty ki sz)
reciprocal x@(Tensor t _) = wrapTensorM (C.reciprocal__t t) Nothing

remainder :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
remainder x@(Tensor t _) div = do
  d <- toCScalar @ty @ki (hsScalarToC div)
  wrapTensorM (C.remainder__ts t d) Nothing

round :: Tensor ty ki sz -> IO (Tensor ty ki sz)
round x@(Tensor t _) = wrapTensorM (C.round__t t) Nothing

sqrt :: Tensor ty ki sz -> IO (Tensor ty ki sz)
sqrt x@(Tensor t _) = wrapTensorM (C.sqrt__t t) Nothing

rsqrt :: Tensor ty ki sz -> IO (Tensor ty ki sz)
rsqrt x@(Tensor t _) = wrapTensorM (C.rsqrt__t t) Nothing

sign :: Tensor ty ki sz -> IO (Tensor ty ki sz)
sign x@(Tensor t _) = wrapTensorM (C.sign__t t) Nothing

trunc :: Tensor ty ki sz -> IO (Tensor ty ki sz)
trunc x@(Tensor t _) = wrapTensorM (C.trunc__t t) Nothing

isNaN :: Tensor ty ki sz -> IO (Tensor TBool ki sz)
isNaN x@(Tensor t _) = wrapTensorM (C.isnan__t t) Nothing

isInf :: Tensor ty ki sz -> IO (Tensor ty ki sz)
isInf x@(Tensor t _) = wrapTensorM (C.trunc__t t) Nothing

matrixRank :: Tensor ty ki '[szh, szw] -> Maybe Double -> Bool -> IO (Scalar ty ki)
matrixRank x@(Tensor t _) Nothing isSymmetric =
  wrapTensorM (C.matrix_rank__tb t (boolc isSymmetric)) Nothing
matrixRank x@(Tensor t _) (Just tol) isSymmetric =
  wrapTensorM (C.matrix_rank__tdb t (coerce tol) (boolc isSymmetric)) Nothing

matrixPower :: SquareBatches sz ~ True
            => Tensor ty ki sz -> Int64 -> IO (Tensor ty ki sz)
matrixPower x@(Tensor t _) n = wrapTensorM (C.matrix_power__t6 t n) Nothing

inverse :: SquareBatches sz ~ True => Tensor ty ki sz -> IO (Tensor ty ki sz)
inverse x@(Tensor t _) = wrapTensorM (C.inverse__t t) Nothing

det :: (SingI (RemoveLastTwoDims sz), SquareBatches sz ~ True)
    => Tensor ty ki sz -> IO (Tensor ty ki (RemoveLastTwoDims sz))
det x@(Tensor t _) = wrapTensorM (C.det__t t) Nothing

logdet :: (SingI (RemoveLastTwoDims sz), SquareBatches sz ~ True)
       => Tensor ty ki sz -> IO (Tensor ty ki (RemoveLastTwoDims sz))
logdet x@(Tensor t _) = wrapTensorM (C.logdet__t t) Nothing

-- * Operations over multiple tensors

set_ :: Tensor ty ki sz -> Tensor ty ki sz -> IO (Tensor ty ki sz)
set_ to@(Tensor toptr _) from@(Tensor fromptr _) = do
  _ <- C.copy__mtb toptr fromptr (boolc False)
  pure to

dot :: forall ty ki sz. Tensor ty ki '[sz] -> Tensor ty ki '[sz] -> IO (Tensor ty ki '[sz])
dot x@(Tensor t _) (Tensor t' _) = wrapTensorM (C.dot__tt t t') Nothing

-- TODO this is internal and should go elsewhere
expandInC :: ForeignPtr C.CTensor -> Vector Int64 -> Vector Int64 -> Bool -> IO (ForeignPtr C.CTensor)
expandInC t sz targetSz implicit | sz == targetSz = pure t
                                 | otherwise = C.expand_mab t targetSz (boolc implicit)

-- | Add two tensors with broadcasting, if one is a scalar some optimizations will be applied.
add :: forall ty ki sz sz'.
      (SingI (BroadcastSizes sz sz'), TensorConstraints ty ki sz)
      => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor ty ki (BroadcastSizes sz sz'))
add (Tensor t _) (Tensor t' _) = do
  let szExpanded = demoteNv @(BroadcastSizes sz sz')
  let sz = demoteNv @sz
  te <- expandInC t sz szExpanded True
  let sz' = demoteNv @sz'
  t'e <- expandInC t' sz' szExpanded True
  s <- toCScalar @ty @ki 1
  rt <- C.add__tts te t'e s
  pure $ Tensor rt Nothing

-- | Add two tensors _without_ broadcasting
add' :: forall ty ki sz. Tensor ty ki sz -> Tensor ty ki sz -> IO (Tensor ty ki sz)
add' (Tensor p _) (Tensor p' _) = do
  s <- toCScalar @ty @ki 1
  rt <- C.add__tts p p' s
  pure $ Tensor rt Nothing

-- | Subtract two tensors with broadcasting, if one is a scalar some optimizations will be applied.
sub :: forall ty ki sz sz'.
      (SingI (BroadcastSizes sz sz'), TensorConstraints ty ki sz)
      => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor ty ki (BroadcastSizes sz sz'))
sub (Tensor t _) (Tensor t' _) = do
  let szExpanded = demoteNv @(BroadcastSizes sz sz')
  let sz = demoteNv @sz
  te <- expandInC t sz szExpanded True
  let sz' = demoteNv @sz'
  t'e <- expandInC t' sz' szExpanded True
  s <- toCScalar @ty @ki 1
  rt <- C.sub__tts te t'e s
  pure $ Tensor rt Nothing

-- | Subtract two tensors _without_ broadcasting
sub' :: forall ty ki sz. Tensor ty ki sz -> Tensor ty ki sz -> IO (Tensor ty ki sz)
sub' (Tensor p _) (Tensor p' _) = do
  s <- toCScalar @ty @ki 1
  rt <- C.sub__tts p p' s
  pure $ Tensor rt Nothing

-- | Multiply two tensors with broadcasting. This is very general and also handles m*v and v*m products.
matmul :: forall ty ki sz sz'.
         SingI (BroadcastMatrices sz sz')
         => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor ty ki (BroadcastMatrices sz sz'))
matmul (Tensor p _) (Tensor p' _) = do
  r <- C.matmul__tt p p'
  pure $ Tensor r Nothing

-- | Elementwise multiplication of two tensors with broadcasting.
mul :: forall ty ki sz sz'. (SingI (BroadcastSizes sz sz'))
      => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor ty ki (BroadcastSizes sz sz'))
mul (Tensor t _) (Tensor t' _) = do
  let szExpanded = demoteNv @(BroadcastSizes sz sz')
  te <- C.expand_mab t szExpanded (boolc True)
  t'e <- C.expand_mab t' szExpanded (boolc True)
  rt <- C.mul__tt te t'e
  pure $ Tensor rt Nothing

-- | Add input to the elementwise multiplication of two other tensors with a scaling factor using broadcasting
-- input + value * tensor1 * tensor 2
addcmul :: forall ty ki sz sz' sz''.
          (SingI (BroadcastSizes sz (BroadcastSizes sz' sz'')))
        => Tensor ty ki sz
        -> TensorTyToHs ty
        -> Tensor ty ki sz'
        -> Tensor ty ki sz''
        -> IO (Tensor ty ki (BroadcastSizes sz (BroadcastSizes sz' sz'')))
addcmul (Tensor i _) val (Tensor t1 _) (Tensor t2 _) = do
  let szExpanded = demoteNv @(BroadcastSizes sz (BroadcastSizes sz' sz''))
  t1'  <- C.expand_mab t1 szExpanded (boolc True)
  t2'  <- C.expand_mab t2 szExpanded (boolc True)
  val' <- toCScalar @ty @ki $ hsScalarToC val
  rt <- C.addcmul__ttts i t1' t2' val'
  pure $ Tensor rt Nothing

-- | Matrix multiplication
mm :: forall ty ki sz1 szin sz2. (SingI sz1, SingI szin, SingI sz2)
     => Tensor ty ki '[sz1, szin] -> Tensor ty ki '[szin,sz2] -> IO (Tensor ty ki '[sz1,sz2])
mm (Tensor t _) (Tensor t' _) = do
  r <- C.mm__tt t t'
  pure $ Tensor r Nothing

-- | Matrix-vector multiplication
mv :: forall ty ki sz1 sz2. (SingI sz1, SingI sz2)
     => Tensor ty ki '[sz1, sz2] -> Tensor ty ki '[sz2] -> IO (Tensor ty ki '[sz1])
mv (Tensor t _) (Tensor t' _) = do
  r <- C.mv__tt t t'
  pure $ Tensor r Nothing

-- | Vector-Matrix multiplication
vm :: forall ty ki sz1 sz2. (SingI sz1, SingI sz2)
     => Tensor ty ki '[sz1] -> Tensor ty ki '[sz1,sz2] -> IO (Tensor ty ki '[sz2])
vm x y = do
  x' <- view x
  r <- mm (sized (size_ @'[1,sz1]) x') y
  view r

-- | Elementwise division of two tensors with broadcasting, if one is a scalar some optimizations will be applied.
div :: forall ty ki sz sz'. (SingI (BroadcastSizes sz sz'))
      => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor ty ki (BroadcastSizes sz sz'))
div (Tensor t _ ) (Tensor t' _) = do
  let szExpanded = demoteNv @(BroadcastSizes sz sz')
  te <- C.expand_mab t szExpanded (boolc True)
  t'e <- C.expand_mab t' szExpanded (boolc True)
  rt <- C.div__tt te t'e
  pure $ Tensor rt Nothing


-- | Add input to the elementwise division of two other tensors with a scaling factor using broadcasting
-- input + value * tensor1 / tensor 2
addcdiv :: forall ty ki sz sz' sz''.
          (SingI (BroadcastSizes sz (BroadcastSizes sz' sz'')))
        => Tensor ty ki sz
        -> TensorTyToHs ty
        -> Tensor ty ki sz'
        -> Tensor ty ki sz''
        -> IO (Tensor ty ki (BroadcastSizes sz (BroadcastSizes sz' sz'')))
addcdiv (Tensor i _) val (Tensor t1 _) (Tensor t2 _) = do
  let szExpanded = demoteNv @(BroadcastSizes sz (BroadcastSizes sz' sz''))
  t1'  <- C.expand_mab t1 szExpanded (boolc True)
  t2'  <- C.expand_mab t2 szExpanded (boolc True)
  val' <- toCScalar @ty @ki $ hsScalarToC val
  rt <- C.addcdiv__ttts i t1' t2' val'
  pure $ Tensor rt Nothing

atan2 :: forall ty ki sz sz'.
        SingI (BroadcastMatrices sz sz')
      => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor ty ki (BroadcastMatrices sz sz'))
atan2 (Tensor p _) (Tensor p' _) = do
  r <- C.atan2__tt p p'
  pure $ Tensor r Nothing

pinverse :: Tensor ty ki sz
         -> Double
         -> IO (Tensor ty ki sz)
pinverse (Tensor t _) rcond = wrapTensorM (C.pinverse__td t (coerce rcond)) Nothing

-- * Probability distributions

-- | Sample from a bernoulli distribution where the probabilities are stored in
-- the tensor.
-- TODO Is this differentiable?
bernoulli :: forall ty ki sz. (IsFloatTy ty ~ True)
          => Tensor ty ki sz -> IO (Tensor ty ki sz)
bernoulli (Tensor input _) =
  wrapTensorM (C.bernoulli__tg input =<< generatorFor (demote @ki)) Nothing

uniform :: forall ty ki sz. (IsFloatTy ty ~ True, SingI sz, TensorConstraints ty ki sz)
        => Double -> Double -> IO (Tensor ty ki sz)
uniform l h = do
  t@(Tensor ptr _) <- empty
  _ <- C.uniform__mddg ptr (CDouble l) (CDouble h) =<< generatorFor (demote @ki)
  pure t

-- TODO need to audit all the uses of IsFloat, not sure we actually need them
multinomialVector :: forall (size :: Nat) (replacement :: Bool) ty ki n.
                    (IsFloatTy ty ~ True, SingI size, SingI replacement
                    ,EnoughIndicesForReplacement replacement size n ~ True)
                  => Size size -> Replacement replacement -> Tensor ty ki '[n] -> IO (Tensor TLong ki '[size])
multinomialVector Size Replacement (Tensor t _) =
  wrapTensorM (C.multinomial_m6bg t (demoteN @size) (boolc (demote @replacement)) =<< generatorFor (demote @ki)) Nothing

-- TODO need to audit all the uses of IsFloat, not sure we actually need them
multinomialMatrix :: forall (size :: Nat) (replacement :: Bool) ty ki n m.
                    (IsFloatTy ty ~ True, SingI size, SingI replacement, SingI n, SingI m
                    ,EnoughIndicesForReplacement replacement size n ~ True)
                  => Size size -> Replacement replacement -> Tensor ty ki '[m,n] -> IO (Tensor TLong ki '[m,size])
multinomialMatrix Size Replacement (Tensor t _) =
  wrapTensorM (C.multinomial_m6bg t (demoteN @size) (boolc (demote @replacement)) =<< generatorFor (demote @ki)) Nothing

where' :: Tensor TBool ki sz -> Tensor ty ki sz -> Tensor ty ki sz -> IO (Tensor ty ki sz)
where' (Tensor b _) (Tensor t _) (Tensor f _) =
  wrapTensorM (C.where__ttt b t f) Nothing


-- * Convolution layers

-- stride padding dilation groups are arguments
conv1d :: forall ty ki nr kernelSize outChans inChans inLen  outLen stride padding dilation groups.
         ( Conv1DSize inLen kernelSize stride padding dilation ~ outLen
         , NonzeroSing stride, SingI padding, NonzeroSing dilation, NonzeroSing groups
         , NonzeroSing nr, NonzeroSing outChans, NonzeroSing outLen) =>
       InChannels inChans
       -> OutChannels outChans
       -> Kernel kernelSize
       -> Stride stride
       -> Padding padding
       -> Dilation dilation
       -> Groups groups
       -> (Tensor ty ki '[outChans, Div inChans groups, kernelSize]
         ,Maybe (Tensor ty ki '[outChans]))
       -> Tensor ty ki '[nr, inChans, inLen] -> IO (Tensor ty ki '[nr, outChans, outLen])
conv1d InChannels OutChannels Kernel Stride Padding Dilation Groups ((Tensor tw _), bias) (Tensor ti _) = do
  tb <- case bias of
         Nothing -> tensorPtr <$> (zeros @ty @ki @'[outChans])
         Just x  -> pure $ tensorPtr x
  t' <- C.conv1d__tttaaa6 ti tw tb
                (V.singleton (demoteN @stride))
                (V.singleton (demoteN @padding))
                (V.singleton (demoteN @dilation))
                (demoteN @groups)
  pure $ Tensor t' Nothing

-- Height then Width (it's the torch way it seems)
-- stride padding dilation groups are arguments
conv2d :: forall inChans ty ki nr outChans inH inW outH outW groups
         kernelW kernelH strideH strideW paddingH paddingW dilationH dilationW.
         ( Conv1DSize inH kernelH strideH paddingH dilationH ~ outH
         , Conv1DSize inW kernelW strideW paddingW dilationW ~ outW
         , NonzeroSing2 strideH strideW, Sing2 paddingH paddingW
         , NonzeroSing2 dilationH dilationW
         , NonzeroSing groups, NonzeroSing nr, NonzeroSing outChans
         , NonzeroSing2 outW outH) =>
       InChannels inChans
       -> OutChannels outChans
       -> Kernel '(kernelH, kernelW)
       -> Stride '(strideH, strideW)
       -> Padding '(paddingH, paddingW)
       -> Dilation '(dilationH, dilationW)
       -> Groups groups
       -> (ConvParam ty ki outChans '[Div inChans groups, kernelH, kernelW])
       -> Tensor ty ki '[nr, inChans, inH, inW]
       -> IO (Tensor ty ki '[nr, outChans, outH, outW])
conv2d InChannels OutChannels Kernel Stride Padding Dilation Groups (ConvParam (Tensor tw _) bias) (Tensor ti _)  = do
  tb <- case bias of
         Nothing -> tensorPtr <$> zeros @ty @ki @'[outChans]
         Just x  -> pure $ tensorPtr x
  t' <- C.conv2d__tttaaa6 ti tw tb
                (V.fromList [demoteN @strideH, demoteN @strideW])
                (V.fromList [demoteN @paddingH, demoteN @paddingW])
                (V.fromList [demoteN @dilationH, demoteN @dilationW])
                (demoteN @groups)
  pure $ Tensor t' Nothing

-- Depth, Height, then Width (it's the torch way it seems)
-- stride padding dilation groups are arguments
conv3d :: forall ty ki nr inChans inD inH inW outChans outD outH outW
         kernelD kernelW kernelH strideD strideH strideW
         paddingD paddingH paddingW dilationD dilationH dilationW groups.
         ( Conv1DSize inH kernelH strideH paddingH dilationH ~ outH
         , Conv1DSize inW kernelW strideW paddingW dilationW ~ outW
         , Conv1DSize inD kernelD strideD paddingD dilationD ~ outD
         , (dilationH > 0)  ~ True, (dilationW > 0)  ~ True, (dilationW > 0)  ~ True
         , NonzeroSing3 strideD strideH strideW, Sing3 paddingD paddingH paddingW
         , NonzeroSing3 dilationD dilationH dilationW, NonzeroSing groups, NonzeroSing nr, NonzeroSing outChans
         , NonzeroSing3 outD outW outH) =>
         InChannels inChans
       -> OutChannels outChans
       -> Kernel '(kernelD, kernelH, kernelW)
       -> Stride '(strideD, strideH, strideW)
       -> Padding '(paddingD, paddingH, paddingW)
       -> Dilation '(dilationD, dilationH, dilationW)
       -> Groups groups
       -> (ConvParam ty ki outChans '[Div inChans groups, kernelH, kernelW])
       -> Tensor ty ki '[nr, inChans, inD, inH, inW]
       -> IO (Tensor ty ki '[nr, outChans, outD, outH, outW])
conv3d InChannels OutChannels Kernel Stride Padding Dilation Groups (ConvParam (Tensor weights _) bias) (Tensor input _) = do
  tb <- case bias of
         Nothing -> tensorPtr <$> zeros @ty @ki @'[outChans]
         Just x  -> pure $ tensorPtr x
  t' <- C.conv3d__tttaaa6 input weights tb
                (V.fromList [demoteN @strideD, demoteN @strideH, demoteN @strideW])
                (V.fromList [demoteN @paddingD, demoteN @paddingH, demoteN @paddingW])
                (V.fromList [demoteN @dilationD, demoteN @dilationH, demoteN @dilationW])
                (demoteN @groups)
  pure $ Tensor t' Nothing

-- * Pooling layers

-- TODO We don't support ceil mode here!
-- kernel stride padding ceilMode are arguments
avgPool1d :: forall kernel stride padding ceilMode ty ki nr chans inW outW
            kernelW strideW paddingW.
            ( AvgPool1DSize inW kernelW strideW paddingW ~ outW
            -- TODO Implement ceilMode; goes into AvgPool1Dsize
            , ceilMode ~ False, SingI ceilMode
            , SingI kernelW, SingI strideW, SingI paddingW
            , SingI nr, SingI chans, SingI outW
            , kernel ~ Kernel kernelW
            , stride ~ Stride strideW
            , padding ~ Padding paddingW) =>
             Kernel kernelW
          -> Stride strideW
          -> Padding paddingW
          -> CeilMode ceilMode
          -> Bool -- This defaults to true, TODO Wrap me!
          -> Tensor ty ki '[nr, chans, inW]
          -> IO (Tensor ty ki '[nr, chans, outW])
avgPool1d Kernel Stride Padding CeilMode countIncludePad (Tensor input _) = do
  t' <- C.avg_pool1d__taaabb input
                          (V.fromList [demoteN @kernelW])
                          (V.fromList [demoteN @strideW])
                          (V.fromList [demoteN @paddingW])
                          (if (demote @ceilMode) then 1 else 0)
                          (if countIncludePad then 1 else 0)
  pure $ Tensor t' Nothing

-- TODO We don't support ceil mode here!
-- kernel stride padding ceilMode are arguments
avgPool2d :: forall kernel stride padding ceilMode ty ki nr chans inH inW outH outW
            kernelH kernelW strideH strideW paddingH paddingW.
            ( AvgPool1DSize inH kernelH strideH paddingH ~ outH
            , AvgPool1DSize inW kernelW strideW paddingW ~ outW
            -- TODO Implement ceilMode; goes into AvgPool1Dsize
            , ceilMode ~ False, SingI ceilMode
            , SingI kernelH, SingI kernelW, SingI strideH, SingI strideW, SingI paddingH, SingI paddingW
            , SingI nr, SingI chans, SingI outH, SingI outW
            , kernel ~ Kernel '(kernelH, kernelW)
            , stride ~ Stride '(strideH, strideW)
            , padding ~ Padding '(paddingH, paddingW)) =>
            Kernel '(kernelH, kernelW)
          -> Stride '(strideH, strideW)
          -> Padding '(paddingH, paddingW)
          -> CeilMode ceilMode
          -> Bool -- This defaults to true, TODO Wrap me!
          -> Tensor ty ki '[nr, chans, inH, inW]
          -> IO (Tensor ty ki '[nr, chans, outH, outW])
avgPool2d Kernel Stride Padding CeilMode countIncludePad (Tensor input _) = do
  t' <- C.avg_pool2d__taaabb6 input
                          (V.fromList [demoteN @kernelH, demoteN @kernelW])
                          (V.fromList [demoteN @strideH, demoteN @strideW])
                          (V.fromList [demoteN @paddingH, demoteN @paddingW])
                          (if (demote @ceilMode) then 1 else 0)
                          (if countIncludePad then 1 else 0)
                          Nothing
  pure $ Tensor t' Nothing

-- TODO We don't support ceil mode here!
-- kernel stride padding ceilMode are arguments
avgPool3d :: forall kernel stride padding ceilMode ty ki nr chans inD inH inW outD outH outW
            kernelD kernelH kernelW strideD strideH strideW paddingD paddingH paddingW.
            ( AvgPool1DSize inH kernelD strideD paddingD ~ outD
            , AvgPool1DSize inH kernelH strideH paddingH ~ outH
            , AvgPool1DSize inW kernelW strideW paddingW ~ outW
            -- TODO Implement ceilMode; goes into AvgPool1Dsize
            , ceilMode ~ False, SingI ceilMode
            , SingI kernelD, SingI kernelH, SingI kernelW
            , SingI strideD, SingI strideH, SingI strideW
            , SingI paddingD, SingI paddingH, SingI paddingW
            , SingI nr, SingI chans, SingI outD, SingI outH, SingI outW
            , kernel ~ Kernel '(kernelD, kernelH, kernelW)
            , stride ~ Stride '(strideD, strideH, strideW)
            , padding ~ Padding '(paddingD, paddingH, paddingW)) =>
             Kernel kernel
          -> Stride stride
          -> Padding padding
          -> CeilMode ceilMode
          -> Bool
          -> Tensor ty ki '[nr, chans, inD, inH, inW]
          -> IO (Tensor ty ki '[nr, chans, outD, outH, outW])
avgPool3d Kernel Stride Padding CeilMode countIncludePad (Tensor input _) = do
  t' <- C.avg_pool3d__taaabb6 input
                          (V.fromList [demoteN @kernelD, demoteN @kernelH, demoteN @kernelW])
                          (V.fromList [demoteN @strideD, demoteN @strideH, demoteN @strideW])
                          (V.fromList [demoteN @paddingD, demoteN @paddingH, demoteN @paddingW])
                          (if (demote @ceilMode) then 1 else 0)
                          (if countIncludePad then 1 else 0)
                          Nothing
  pure $ Tensor t' Nothing

-- | Adaptive 1D average pooling takes an output size and finds max pooling
-- parameters to achieve that size.
adaptiveAvgPool1d :: forall outF' outF ty ki nr chans inW outW.
                  (OutFeatures outF ~ outF', outW ~ outF
                  ,SingI nr, SingI chans, SingI outW, SingI outF
                  ,TensorConstraints ty ki '[nr, chans, inW])
                => OutFeatures outF
                -> Tensor ty ki '[nr, chans, inW]
                -> IO (Tensor ty ki '[nr, chans, outW])
adaptiveAvgPool1d OutFeatures (Tensor t _) = do
  wrapTensorM (C.adaptive_avg_pool1d__ta t (V.fromList [demoteN @outW])) Nothing

-- | Adaptive 2D average pooling takes an output size and finds max pooling
-- parameters to achieve that size.
adaptiveAvgPool2d :: forall outF' outF ty ki nr chans inH inW outH outW.
                  (OutFeatures outF ~ outF', '[outH, outW] ~ outF
                  ,SingI nr, SingI chans, SingI outH, SingI outW, SingI outF
                  ,TensorConstraints ty ki '[nr, chans, inH, inW])
                => OutFeatures outF
                -> Tensor ty ki '[nr, chans, inH, inW]
                -> IO (Tensor ty ki '[nr, chans, outH, outW])
adaptiveAvgPool2d OutFeatures (Tensor t _) = do
  wrapTensorM (C.adaptive_avg_pool2d__ta t (V.fromList [demoteN @outH, demoteN @outW])) Nothing

-- | Adaptive 3D average pooling takes an output size and finds max pooling
-- parameters to achieve that size.
adaptiveAvgPool3d :: forall outF' outF ty ki nr chans inD inH inW outD outH outW.
                  (OutFeatures outF ~ outF', '[outD, outH, outW] ~ outF
                  ,SingI nr, SingI chans, SingI outD, SingI outH, SingI outW, SingI outF
                  ,TensorConstraints ty ki '[nr, chans, inD, inH, inW])
                => OutFeatures outF
                -> Tensor ty ki '[nr, chans, inD, inH, inW]
                -> IO (Tensor ty ki '[nr, chans, outD, outH, outW])
adaptiveAvgPool3d OutFeatures (Tensor t _) = do
  wrapTensorM (C.adaptive_avg_pool3d__ta t (V.fromList [demoteN @outD, demoteN @outH, demoteN @outW])) Nothing

-- | Max pooling
-- TODO We don't support ceil mode here!
-- kernel stride padding ceilMode are arguments
maxPool1d :: forall ceilMode ty ki nr chans inW outW
           kernelW strideW paddingW dilationW.
         ( Conv1DSize inW kernelW strideW paddingW dilationW ~ outW
         , ceilMode ~ False, SingI ceilMode
         , NonzeroSing kernelW, NonzeroSing strideW, SingI paddingW
         , NonzeroSing dilationW, NonzeroSing nr, NonzeroSing chans
         , NonzeroSing outW, TensorConstraints 'TLong ki '[]) =>
        Kernel kernelW
      -> Stride strideW
      -> Padding paddingW
      -> CeilMode ceilMode
      -> Tensor ty ki '[nr, chans, inW]
      -> IO (Tensor ty     ki '[nr, chans, outW]
            ,Tensor 'TLong ki '[nr, chans, outW])
maxPool1d Kernel Stride Padding CeilMode (Tensor input _) = do
  (t',ti') <- C.max_pool1d_with_indices__taaaab input
                                             (V.fromList [demoteN @kernelW])
                                             (V.fromList [demoteN @strideW])
                                             (V.fromList [demoteN @paddingW])
                                             (V.fromList [demoteN @dilationW])
                                             (boolc (demote @ceilMode))
  pure $ (Tensor t' Nothing, Tensor ti' Nothing)

-- | 2D Max pooling
-- TODO We don't support ceil mode here!
-- kernel stride padding ceilMode are arguments
maxPool2d :: forall ceilMode ty ki nr inH inW chans outH outW
           kernelH kernelW strideH strideW paddingH paddingW dilationH dilationW.
         ( Conv1DSize inH kernelH strideH paddingH dilationH ~ outH
         , Conv1DSize inW kernelW strideW paddingW dilationW ~ outW
         , ceilMode ~ False, SingI ceilMode
         , NonzeroSing2 kernelH kernelW, NonzeroSing2 strideH strideW, Sing2 paddingH paddingW
         , NonzeroSing2 dilationH dilationW, NonzeroSing nr, NonzeroSing chans
         , SingI outW, SingI outH, TensorConstraints 'TLong ki '[]
         ) =>
         Kernel '(kernelH, kernelW)
      -> Stride '(strideH, strideW)
      -> Padding '(paddingH, paddingW)
      -> Dilation '(dilationH, dilationW)
      -> CeilMode ceilMode
      -> Tensor ty ki '[nr, chans, inH, inW]
      -> IO (Tensor ty     ki '[nr, chans, outH, outW]
            ,Tensor 'TLong ki '[nr, chans, outH, outW])
maxPool2d Kernel Stride Padding Dilation CeilMode (Tensor input _) = do
  (t',ti') <- C.max_pool2d_with_indices__taaaab input
                                             (V.fromList [demoteN @kernelH, demoteN @kernelW])
                                             (V.fromList [demoteN @strideH, demoteN @strideW])
                                             (V.fromList [demoteN @paddingH, demoteN @paddingW])
                                             (V.fromList [demoteN @dilationH, demoteN @dilationW])
                                             (boolc (demote @ceilMode))
  pure $ (Tensor t' Nothing, Tensor ti' Nothing)

-- | 3D Max pooling
-- TODO We don't support ceil mode here!
-- kernel stride padding ceilMode are arguments
maxPool3d :: forall ceilMode ty ki nr inD inH inW chans outD outH outW
           kernelD kernelH kernelW strideD strideH strideW paddingD paddingH paddingW dilationD dilationH dilationW.
         ( Conv1DSize inD kernelD strideD paddingD dilationD ~ outD
         , Conv1DSize inH kernelH strideH paddingH dilationH ~ outH
         , Conv1DSize inW kernelW strideW paddingW dilationW ~ outW
         , ceilMode ~ False, SingI ceilMode
         , NonzeroSing3 kernelD kernelH kernelW
         , Sing3 strideD strideH strideW
         , Sing3 paddingD paddingH paddingW
         , NonzeroSing3 dilationD dilationH dilationW, NonzeroSing nr, NonzeroSing chans
         , NonzeroSing3 outD outW outH, TensorConstraints 'TLong ki '[]) =>
        Kernel '(kernelD, kernelH, kernelW)
      -> Stride '(strideD, strideH, strideW)
      -> Padding '(paddingD, paddingH, paddingW)
      -> CeilMode ceilMode
      -> Tensor ty ki '[nr, chans, inD, inH, inW]
      -> IO (Tensor ty     ki '[nr, chans, outD, outH, outW]
            ,Tensor 'TLong ki '[nr, chans, outD, outH, outW])
maxPool3d Kernel Stride Padding CeilMode (Tensor input _) = do
  (t',ti') <- C.max_pool3d_with_indices__taaaab input
                                             (V.fromList [demoteN @kernelD, demoteN @kernelH, demoteN @kernelW])
                                             (V.fromList [demoteN @strideD, demoteN @strideH, demoteN @strideW])
                                             (V.fromList [demoteN @paddingD, demoteN @paddingH, demoteN @paddingW])
                                             (V.fromList [demoteN @dilationD, demoteN @dilationH, demoteN @dilationW])
                                             (boolc (demote @ceilMode))
  pure $ (Tensor t' Nothing, Tensor ti' Nothing)

-- | Adaptive 1D max pooling takes an output size and finds max pooling parameters
-- to achieve that size.
adaptiveMaxPool1d :: forall outF' outF ty ki nr chans inW outW.
                  (OutFeatures outF ~ outF', outW ~ outF
                  ,SingI nr, SingI chans, SingI outW, SingI outF
                  ,TensorConstraints ty ki '[nr, chans, inW])
                => OutFeatures outF
                -> Tensor ty ki '[nr, chans, inW]
                -> IO (Tensor ty ki '[nr, chans, outW], Tensor 'TLong ki '[nr, chans, outW])
adaptiveMaxPool1d OutFeatures (Tensor t _) = do
  (t', ti') <- C.adaptive_max_pool1d__ta t (V.fromList [demoteN @outW])
  pure $ (Tensor t' Nothing, Tensor ti' Nothing)

-- | Adaptive 2D max pooling takes an output size and finds max pooling parameters
-- to achieve that size.
adaptiveMaxPool2d :: forall outF' outF ty ki nr chans inH inW outH outW.
                  (OutFeatures outF ~ outF', '[outH, outW] ~ outF
                  ,SingI nr, SingI chans, SingI outH, SingI outW, SingI outF
                  ,TensorConstraints ty ki '[nr, chans, inH, inW])
                => OutFeatures outF
                -> Tensor ty ki '[nr, chans, inH, inW]
                -> IO (Tensor ty ki '[nr, chans, outH, outW]
                     ,Tensor TLong ki '[nr, chans, outH, outW])
adaptiveMaxPool2d OutFeatures (Tensor t _) = do
  (t', ti') <- C.adaptive_max_pool2d__ta t (V.fromList [demoteN @outH, demoteN @outW])
  pure $ (Tensor t' Nothing, Tensor ti' Nothing)

-- | Adaptive 3D max pooling takes an output size and finds max pooling parameters
-- to achieve that size.
adaptiveMaxPool3d :: forall outF' outF ty ki nr chans inD inH inW outD outH outW.
                  (OutFeatures outF ~ outF', '[outD, outH, outW] ~ outF
                  ,SingI nr, SingI chans, SingI outD, SingI outH, SingI outW, SingI outF
                  ,TensorConstraints ty ki '[nr, chans, outD, inD, inH, inW])
                => OutFeatures outF
                -> Tensor ty ki '[nr, chans, inD, inH, inW]
                -> IO (Tensor ty ki '[nr, chans, outD, outH, outW]
                     ,Tensor TLong ki '[nr, chans, outD, outH, outW])
adaptiveMaxPool3d OutFeatures (Tensor t _) = do
  (t', ti') <- C.adaptive_max_pool3d__ta t (V.fromList [demoteN @outD, demoteN @outH, demoteN @outW])
  pure $ (Tensor t' Nothing, Tensor ti' Nothing)

-- * Padding

constantPad1d :: forall padLeft padRight ty ki n c inw.
                (SingI '[n, c, inw + padLeft + padRight], SingI padLeft, SingI padRight)
              => Tensor ty ki '[n,c,inw]
              -> TensorTyToHs ty ->
                IO (Tensor ty ki '[n,c,inw+padLeft+padRight])
constantPad1d t@(Tensor p _) value = do
  v <- toCScalar @ty @ki (hsScalarToC value)
  grad <- requiresGrad t
  setRequiresGrad grad =<< wrapTensorM (C.constant_pad_nd__tas p (V.fromList [demoteN @padLeft, demoteN @padRight]) v) Nothing

constantPad2d :: forall padTop padBottom padLeft padRight ty ki n c inh inw.
                (SingI '[n, c, inh + padTop + padBottom, inw + padLeft + padRight]
                , SingI padLeft, SingI padRight, SingI padTop, SingI padBottom)
              => Tensor ty ki '[n,c,inh,inw]
              -> TensorTyToHs ty
              -> IO (Tensor ty ki '[n, c, inh + padTop + padBottom, inw + padLeft + padRight])
constantPad2d t@(Tensor p _) value = do
  v <- toCScalar @ty @ki (hsScalarToC value)
  grad <- requiresGrad t
  setRequiresGrad grad =<< wrapTensorM (C.constant_pad_nd__tas p (V.fromList [demoteN @padTop, demoteN @padBottom, demoteN @padLeft, demoteN @padRight]) v) Nothing

constantPad3d :: forall padFront padBack padTop padBottom padLeft padRight ty ki n c ind inh inw.
                (SingI '[n, c, ind + padFront + padBack, inh + padTop + padBottom, inw + padLeft + padRight]
                , SingI padLeft, SingI padRight, SingI padTop, SingI padBottom, SingI padFront, SingI padBack)
              => Tensor ty ki '[n,c,ind,inh,inw]
              -> TensorTyToHs ty
              -> IO (Tensor ty ki '[n, c, ind + padFront + padBack, inh + padTop + padBottom, inw + padLeft + padRight])
constantPad3d t@(Tensor p _) value = do
  v <- toCScalar @ty @ki (hsScalarToC value)
  grad <- requiresGrad t
  setRequiresGrad grad =<< wrapTensorM (C.constant_pad_nd__tas p (V.fromList [demoteN @padFront, demoteN @padBack, demoteN @padTop, demoteN @padBottom, demoteN @padLeft, demoteN @padRight]) v) Nothing

reflectionPad1d :: forall padLeft padRight ty ki n c inw.
                (SingI '[n, c, inw + padLeft + padRight], SingI padLeft, SingI padRight)
              => Tensor ty ki '[n,c,inw]
              -> IO (Tensor ty ki '[n,c,inw+padLeft+padRight])
reflectionPad1d t@(Tensor p _) = do
  grad <- requiresGrad t
  setRequiresGrad grad =<< wrapTensorM (C.reflection_pad1d__ta p (V.fromList [demoteN @padLeft, demoteN @padRight])) Nothing

reflectionPad2d :: forall padTop padBottom padLeft padRight ty ki n c inh inw.
                (SingI '[n, c, inh + padTop + padBottom, inw + padLeft + padRight]
                , SingI padLeft, SingI padRight, SingI padTop, SingI padBottom)
              => Tensor ty ki '[n,c,inh,inw]
              -> IO (Tensor ty ki '[n, c, inh + padTop + padBottom, inw + padLeft + padRight])
reflectionPad2d t@(Tensor p _) = do
  grad <- requiresGrad t
  setRequiresGrad grad =<< wrapTensorM (C.reflection_pad2d__ta p (V.fromList [demoteN @padTop, demoteN @padBottom, demoteN @padLeft, demoteN @padRight])) Nothing

replicationPad1d :: forall padLeft padRight ty ki n c inw.
                (SingI '[n, c, inw + padLeft + padRight], SingI padLeft, SingI padRight)
              => Tensor ty ki '[n,c,inw]
              -> IO (Tensor ty ki '[n,c,inw+padLeft+padRight])
replicationPad1d t@(Tensor p _) = do
  grad <- requiresGrad t
  setRequiresGrad grad =<< wrapTensorM (C.replication_pad1d__ta p (V.fromList [demoteN @padLeft, demoteN @padRight])) Nothing

replicationPad2d :: forall padTop padBottom padLeft padRight ty ki n c inh inw.
                (SingI '[n, c, inh + padTop + padBottom, inw + padLeft + padRight]
                , SingI padLeft, SingI padRight, SingI padTop, SingI padBottom)
              => Tensor ty ki '[n,c,inh,inw]
              -> IO (Tensor ty ki '[n, c, inh + padTop + padBottom, inw + padLeft + padRight])
replicationPad2d t@(Tensor p _) = do
  grad <- requiresGrad t
  setRequiresGrad grad =<< wrapTensorM (C.replication_pad2d__ta p (V.fromList [demoteN @padTop, demoteN @padBottom, demoteN @padLeft, demoteN @padRight])) Nothing

replicationPad3d :: forall padFront padBack padTop padBottom padLeft padRight ty ki n c ind inh inw.
                (SingI '[n, c, ind + padFront + padBack, inh + padTop + padBottom, inw + padLeft + padRight]
                , SingI padLeft, SingI padRight, SingI padTop, SingI padBottom, SingI padFront, SingI padBack)
              => Tensor ty ki '[n,c,ind,inh,inw]
              -> IO (Tensor ty ki '[n, c, ind + padFront + padBack, inh + padTop + padBottom, inw + padLeft + padRight])
replicationPad3d t@(Tensor p _) = do
  grad <- requiresGrad t
  setRequiresGrad grad =<< wrapTensorM (C.replication_pad3d__ta p (V.fromList [demoteN @padFront, demoteN @padBack, demoteN @padTop, demoteN @padBottom, demoteN @padLeft, demoteN @padRight])) Nothing

-- * Statistics

-- | Histogram of a tensor
histc :: forall bins ty ki sz. (TensorConstraints ty ki sz, SingI bins)
  => Tensor ty ki sz -> TensorTyToHs ty -> TensorTyToHs ty -> IO (Tensor ty ki '[bins])
histc (Tensor t _) min max = do
  min' <- toCScalar @ty @ki (hsScalarToC min)
  max' <- toCScalar @ty @ki (hsScalarToC max)
  t' <- C.histc__t6ss t (demoteN @bins) min' max'
  pure $ Tensor t' Nothing

-- * Embeddings

-- TODO No support for sparse embeddings
-- NB This function can change the weights!
embedding :: (SingI (AddDimension sz embeddingDim))
          => NrEmbeddings nrEmbeddings
          -> EmbeddingDimensions embeddingDim
          -> Maybe Int64 -- TODO This is a runtime index
          -> Maybe (Double, Double) -- the max norm and a p-norm (like 2 would be the l2 norm)
          -> Bool
          -> EmbeddingParam ty ki nrEmbeddings embeddingDim
          -> Tensor TLong ki sz
          -> IO (Tensor ty ki (AddDimension sz embeddingDim))
embedding NrEmbeddings EmbeddingDimensions paddingIdx maxNorm scaleGradByFreq
          (EmbeddingParam (Tensor t _)) (Tensor input _ ) = do
  case maxNorm of
    Nothing -> pure ()
    Just (maxNormValue, pNorm) ->
      withoutGrad $ C.embedding_renorm___ttdd t input (coerce maxNormValue) (coerce pNorm) >> pure ()
  wrapTensorM (C.embedding__tt6bb t input (fromMaybe (-100) paddingIdx) (boolc scaleGradByFreq) (boolc False)) Nothing

padEmbedding_ :: (TensorConstraints ty ki '[nrEmbeddings, embeddingDim]
                ,SingI nrEmbeddings, SingI embeddingDim)
              => EmbeddingParam ty ki nrEmbeddings embeddingDim
              -> Int64
              -> IO (EmbeddingParam ty ki nrEmbeddings embeddingDim)
padEmbedding_ e@(EmbeddingParam t) paddingIdx = do
  z <- select @0 t paddingIdx
  _ <- fill_ z 0
  pure e
  where fill_ :: forall ty ki sz. (TensorConstraints ty ki sz)
              => Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
        fill_ t@(Tensor ptr _) c = do
          fill <- toCScalar @ty @ki (hsScalarToC c)
          _ <- C.full_out__tas ptr (demoteNv @sz) fill
          pure t

-- * Loss functions

cosineSimilarity :: forall ty ki sz dim. (TensorConstraints ty ki sz, SingI (RemoveDimension sz dim), SingI dim)
                 => Dimension dim
                 -> Maybe Double
                 -> Tensor ty ki sz
                 -> Tensor ty ki sz
                 -> IO (Tensor ty ki (RemoveDimension sz dim))
cosineSimilarity _ eps (Tensor t _) (Tensor t' _) =
  wrapTensorM (C.cosine_similarity__tt6d t t'
                                   (demoteN @dim)
                                   (case eps of
                                         Nothing -> 1e-08
                                         Just e  -> coerce e)) Nothing

pairwiseDistance :: forall ty ki n d. (TensorConstraints ty ki '[n,d], SingI n)
                 => Double
                 -> Maybe Double
                 -> Tensor ty ki '[n,d]
                 -> Tensor ty ki '[n,d]
                 -> IO (Tensor ty ki '[n])
pairwiseDistance pNormDegree eps (Tensor t _) (Tensor t' _) =
  wrapTensorM (C.pairwise_distance__ttddb t t' (coerce pNormDegree)
                         (case eps of
                             Nothing -> 1e-08
                             Just e  -> coerce e)
                         (boolc False)) Nothing

l1Loss :: forall ty ki sz. (TensorConstraints ty ki sz)
       => Tensor ty ki sz
       -> SizeAverage Bool
       -> Tensor ty ki sz
       -> IO (Scalar ty ki)
l1Loss (Tensor target _) (SizeAverage sa) (Tensor input _) =
  wrapTensorM (C.l1_loss__tt6 input target (fromIntegral $ fromEnum $ if sa then
                                                                   C.ReductionMean else
                                                                   C.ReductionSum)) Nothing

l1Losses :: forall ty ki sz sz'. (TensorConstraints ty ki '[])
          => Tensor ty ki sz
          -> Tensor ty ki sz'
          -> IO (Tensor ty ki sz)
l1Losses (Tensor target _) (Tensor input _) = do
  wrapTensorM (C.l1_loss__tt6 input target (fromIntegral $ fromEnum C.ReductionNone)) Nothing

-- sizeAverage defaults to true
mseLoss :: forall ty ki sz. (TensorConstraints ty ki sz)
        => Tensor ty ki sz
        -> SizeAverage Bool
        -> Tensor ty ki sz
        -> IO (Scalar ty ki)
mseLoss (Tensor target _) (SizeAverage sa) (Tensor input _) =
  wrapTensorM (C.mse_loss__tt6 input target (fromIntegral $ fromEnum $ if sa then
                                                                    C.ReductionMean else
                                                                    C.ReductionSum))
              Nothing

mseLosses :: forall ty ki sz sz'. (TensorConstraints ty ki '[])
          => Tensor ty ki sz
          -> Tensor ty ki sz'
          -> IO (Tensor ty ki sz)
mseLosses (Tensor target _) (Tensor input _) = do
  wrapTensorM (C.mse_loss__tt6 input target (fromIntegral $ fromEnum C.ReductionNone)) Nothing

-- TODO k-dimensional version
nllLoss :: forall ty ki sz nrClasses.
          (TensorConstraints ty ki '[nrClasses])
        => Tensor TLong ki '[sz]                  -- ^ target
        -> Maybe (Tensor TFloat ki '[nrClasses])
        -> SizeAverage Bool
        -> Maybe Int64
        -> Tensor ty ki '[sz,nrClasses]           -- ^ input
        -> IO (Scalar ty ki)
nllLoss (Tensor target _) rescaleWeights (SizeAverage sa) ignoreIndex (Tensor input _) = do
  trescale <- case rescaleWeights of
               Nothing           -> C.undefinedTensor
               Just (Tensor x _) -> pure x
  l <- C.nll_loss__ttt66 
                input target trescale
                (fromIntegral $ fromEnum $ if sa then
                                             C.ReductionMean else
                                             C.ReductionSum)
                (case ignoreIndex of
                    Nothing -> -100
                    Just x  -> x)
  pure $ Tensor l Nothing

nllLosses :: forall ty ki sz nrClasses.
          (TensorConstraints ty ki '[nrClasses])
        => Tensor TLong ki '[sz]                     -- ^ target
        -> Maybe (Tensor TFloat ki '[nrClasses])
        -> Maybe Int64
        -> Tensor ty ki '[sz,nrClasses]              -- ^ input
        -> IO (Tensor ty ki '[sz])
nllLosses (Tensor target _) rescaleWeights ignoreIndex (Tensor input _) = do
  trescale <- case rescaleWeights of
               Nothing           -> C.undefinedTensor
               Just (Tensor x _) -> pure x
  wrapTensorM (C.nll_loss__ttt66 input target trescale (fromIntegral $ fromEnum C.ReductionNone)
                (case ignoreIndex of
                    Nothing -> -100
                    Just x  -> x)) Nothing

nllLoss2d :: forall ty ki sz nrClasses szh szw.
            (TensorConstraints ty ki '[nrClasses,szh,szw])
          => Tensor TLong ki '[sz,szh,szw]               -- ^ target
          -> Maybe (Tensor TFloat ki '[nrClasses])
          -> SizeAverage Bool
          -> Maybe Int64
          -> Tensor ty ki '[sz,nrClasses,szh,szw]        -- ^ input
          -> IO (Scalar ty ki)
nllLoss2d (Tensor target _) rescaleWeights (SizeAverage sa) ignoreIndex (Tensor input _) = do
  trescale <- case rescaleWeights of
               Nothing           -> C.undefinedTensor
               Just (Tensor x _) -> pure x
  l <- C.nll_loss2d__ttt66
                  input target trescale
                  (fromIntegral $ fromEnum $ if sa then
                                               C.ReductionMean else
                                               C.ReductionSum)
                  (case ignoreIndex of
                      Nothing -> -100
                      Just x  -> x)
  pure $ Tensor l Nothing

nllLossess2d :: forall ty ki sz nrClasses szh szw.
            (TensorConstraints ty ki '[nrClasses,szh,szw])
          => Tensor TLong ki '[sz,szh,szw]          -- ^ target
          -> Maybe (Tensor TFloat ki '[nrClasses])
          -> Maybe Int64
          -> Tensor ty ki '[sz,nrClasses,szh,szw]   -- ^ input
          -> IO (Scalar ty ki)
nllLossess2d (Tensor target _) rescaleWeights ignoreIndex (Tensor input _) = do
  trescale <- case rescaleWeights of
               Nothing           -> C.undefinedTensor
               Just (Tensor x _) -> pure x
  wrapTensorM (C.nll_loss2d__ttt66 input target trescale (fromIntegral $ fromEnum $ C.ReductionNone)
                (case ignoreIndex of
                    Nothing -> -100
                    Just x  -> x)) Nothing

-- TODO k-dimensional version
-- TODO crossEntropy target is marked as _assert_no_grad() what else is?
crossEntropyLoss
  :: (TensorConstraints ty ki sz, KnownNat nrClasses) =>
     Tensor TLong ki '[sz]
     -> Maybe (Tensor TFloat ki '[nrClasses])
     -> SizeAverage Bool
     -> Maybe Int64
     -> Tensor ty ki '[sz, nrClasses]
     -> IO (Scalar ty ki)
crossEntropyLoss target rescaleWeights sa ignoreIndex input = do
  input' <- logSoftmax input 1
  nllLoss target rescaleWeights sa ignoreIndex input'

crossEntropyLosses
  :: (TensorConstraints ty ki sz, KnownNat nrClasses) =>
     Tensor TLong ki '[sz]
     -> Maybe (Tensor TFloat ki '[nrClasses])
     -> Maybe Int64
     -> Tensor ty ki '[sz, nrClasses]
     -> IO (Tensor ty ki '[sz])
crossEntropyLosses target rescaleWeights ignoreIndex input = do
  input' <- logSoftmax input 1
  nllLosses target rescaleWeights ignoreIndex input'

-- | Compute BCE
binaryCrossEntropyLoss :: (SingI sz)
                       => Tensor ty ki (n:sz)
                       -> Maybe (Tensor ty ki '[n])
                       -> SizeAverage Bool
                       -> Tensor ty ki (n:sz)
                       -> IO (Scalar ty ki)
binaryCrossEntropyLoss t@(Tensor target _) weight (SizeAverage sa) (Tensor i _) = do
  whenM (requiresGrad t) $
    error "BCE loss is not differntiable with respect to the target, mark the target as not requiring a gradient with noGrad.\n One day this will be enforced at compile time."
  w <- (case weight of
          Nothing           -> C.undefinedTensor
          Just (Tensor x _) -> pure x)
  wrapTensorM (C.binary_cross_entropy__ttt6 i target w (fromIntegral $ fromEnum $ if sa then
                                                             C.ReductionMean else
                                                             C.ReductionSum)) Nothing

-- | Compute BCE
binaryCrossEntropyLosses :: (SingI sz)
                     => Tensor ty ki (n:sz)
                     -> Maybe (Tensor ty ki '[n])
                     -> Tensor ty ki (n:sz)
                     -> IO (Tensor ty ki (n:sz))
binaryCrossEntropyLosses t@(Tensor target _) weight (Tensor i _) = do
  whenM (requiresGrad t) $
    error "BCE loss is not differntiable with respect to the target, mark the target as not requiring a gradient with noGrad.\n One day this will be enforced at compile time."
  w <- (case weight of
          Nothing           -> C.undefinedTensor
          Just (Tensor x _) -> pure x)
  wrapTensorM (C.binary_cross_entropy__ttt6 i target w (fromIntegral $ fromEnum $ C.ReductionNone)) Nothing

-- * Debugging

-- | This is a handy print function for debugging
-- TODO Make this look better
shortPrintTensor :: Tensor ty ki sz -> IO ()
shortPrintTensor ten@(Tensor t _) = do
                      shape <- CV.shape t
                      putStrLn $ "    size" <> show shape
                      briefPrint (debuggingVerifyShape "shortPrintTensor" ten)
  where briefPrint ten = do
          v <- toVector =<< toCpu ten
          let f t = putStrLn $ Data.List.intercalate ", " $ map (flip (showEFloat (Just 4)) "" . toDouble . cScalarToHs) $ V.toList t
          f $ V.take 10 v
          putStrLn "  ... "
          f $ V.drop (V.length v - 10) v
