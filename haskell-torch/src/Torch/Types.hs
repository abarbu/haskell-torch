{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, DeriveFoldable, DeriveFunctor, DeriveGeneric, DeriveTraversable     #-}
{-# LANGUAGE EmptyCase, ExistentialQuantification, FlexibleContexts, FlexibleInstances, FunctionalDependencies, GADTs, KindSignatures  #-}
{-# LANGUAGE MultiParamTypeClasses, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, QuasiQuotes, RankNTypes     #-}
{-# LANGUAGE RecordWildCards, ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies, TypeInType #-}
{-# LANGUAGE TypeOperators, UndecidableInstances, StandaloneKindSignatures                                                             #-}
{-# OPTIONS_GHC -pgmP cc -optP -E -optP -undef -optP -std=c89 -Wall -fno-warn-unticked-promoted-constructors -fno-warn-orphans #-}

-- | The basic types and type-level operators shared by all of Haskell-Torch.
--
-- Conventions:
--  - scalar arguments that impact shape/size/etc of the computation are passed as lifted types,
--    they will be the first types of any function that needs them
--  - equivalent operators to ones that exit in prelude for numbers are defined, they are prefixed by
--    a period, as in .>, and return a value wrapped in IO. Operations that only apply to vectors are
--    prefixed by ^ and operations that only apply to matrices are prefixed with #.
--  - some highly polymorphic torch operations have simplified semantics, see matmul for example.
--    This is not becuase we can't support the full semantics, it is trivial to do so. But it is
--    against the spirit of keeping mathematical and logical operations clear to conflate such
--    different concepts as dot products and matrix-matrix products.
--  - operations come in two kinds, the .> version and the ..> version. Double colon
--    ops take as input IO and produce IO. So you can't write w .* x .+ b normally
--    because the results are in IO but the colon variant is sufficiently polymorphic that you can.
--    w ..* x ..+ b.

module Torch.Types where
import           Control.Monad
import           Data.Coerce
import           Data.Default
import           Data.IORef
import           Data.Singletons
import           Data.Singletons.Prelude      as SP
import           Data.Singletons.TH
import           Data.Singletons.TypeLits
import           Data.Text                    (Text)
import qualified Data.Text                    as T
import qualified Data.Vector                  as VB
import qualified Data.Vector.Storable         as V
import           Data.Word
import           Foreign.C.String
import           Foreign.C.Types
import           Foreign.ForeignPtr
import           Foreign.Ptr
import           Foreign.Storable
import           Generics.Eot
import           GHC.Float
import           GHC.Int
import           GHC.TypeLits                 as TL hiding (type (+), type (-))
import           Numeric.Half
import           System.IO.Unsafe
import qualified Torch.C.Scalar               as C
import qualified Torch.C.Tensor               as C
import qualified Torch.C.Types                as C
import qualified Torch.C.Variable             as C
import           Torch.Misc
import           Plugin.DefaultType
import           GHC.TypeLits(Nat)

-- | The storage typeof the tensor we're dealing with. This will appearat the
-- type level regularly.
data TensorType = TBool
                | TByte
                | TChar
                | TShort
                | TInt
                | TLong
                | THalf
                | TFloat
                | TDouble
                deriving (Show, Eq)

-- | Is this type fixed or floating?
type family IsFloatTy (a :: TensorType) :: Bool where
  IsFloatTy TFloat = True
  IsFloatTy TDouble = True
  IsFloatTy THalf = True
  IsFloatTy _ = False

-- | Expand a fixed type to its largest counterpart. All other types, the
-- floating point ones, are left alone.
type family ExpandType (a :: TensorType) where
  ExpandType TBool   = TLong
  ExpandType TByte   = TLong
  ExpandType TChar   = TLong
  ExpandType TShort  = TLong
  ExpandType TInt    = TLong
  ExpandType TLong   = TLong
  ExpandType THalf   = THalf
  ExpandType TFloat  = TFloat
  ExpandType TDouble = TDouble

-- | These appear at the type level and describe the device that holds the
-- tensor. KCuda is only available when compiled with cuda support. If you don't
-- have a device, this code will fail to type check!
data TensorKind = KCpu
#if WITH_CUDA
                | KCuda
#endif
                deriving (Show, Eq)

-- | If you are disciplined about only using this instead of KCuda or KCpu, you
-- will end up with models that run on whichever device is available. Note that
-- this isn't easy. Your compiler will not complain if your code is not
-- sufficiently polymorphic, it only cares that the current best device fits.
-- TODO Do something about best devices and find a better scheme.
type KBest =
#if WITH_CUDA
              KCuda
#else
              KCpu
#endif

-- | Internal
cDeviceType :: TensorKind -> C.Backend
cDeviceType KCpu  = C.BackendCPU
#if WITH_CUDA
cDeviceType KCuda = C.BackendCUDA
#endif

genSingletons [''TensorType, ''TensorKind]

singDecideInstances [''TensorType, ''TensorKind]

-- * Many singletons for type-level operations.
-- We would love to do this:
--
-- $(singletonsOnly [d|
--   isScalarLike :: [Nat] -> Bool
--   isScalarLike []    = True
--   isScalarLike (h:t) = if h == 1 then isScalarLike t else False
--  |])
--
-- But the resulting code leads to far worse types than writing everything by
-- hand.

-- | Are these two shapes expandable?
type family ExpandTo (a::[Nat]) (b::[Nat]) :: Bool where
  ExpandTo '[] _         = True
  ExpandTo _ '[]         = True
  ExpandTo (i:is) (i:js) = ExpandTo is js
  ExpandTo (1:is) (_:js) = ExpandTo is js
  ExpandTo (_:is) (1:js) = ExpandTo is js
  ExpandTo (_:_)  (_:_)  = False

-- | Remove a dimension
type family RemoveDimension (a::[Nat]) (b::Nat) :: [Nat] where
  RemoveDimension '[] 0   = '[]
  RemoveDimension '[] _   = TypeError (TL.Text "Index out of range for RemoveDimension")
  RemoveDimension (h:t) 0 = t
  RemoveDimension (h:t) n = h : RemoveDimension t (n - 1)

-- | Remove all of the 1 dimensions
type family Squeeze (a::[Nat]) :: [Nat] where
  Squeeze '[] = '[]
  Squeeze '[x] = '[x]
  Squeeze (1:t) = Squeeze t
  Squeeze (h:t) = h : Squeeze t

type family SameOrScalar (a::[Nat]) (b::[Nat]) :: [Nat] where
  SameOrScalar '[] x = x
  SameOrScalar x '[] = x
  SameOrScalar x x   = x
  SameOrScalar _   _ = TypeError (TL.Text "Types are not compatible")

type family ReplaceLast (a::[Nat]) (b::Nat) :: [Nat] where
  ReplaceLast '[]   _ = TypeError (TL.Text "Empty list when replacing last item")
  ReplaceLast '[h]  x = '[x]
  ReplaceLast (h:t) x = h : ReplaceLast t x

type family Conv1DSize (lin::Nat) (kernelSize::Nat) (stride::Nat) (padding::Nat) (dilation::Nat) :: Nat where
  Conv1DSize lin kernelSize stride padding dilation = (Div (lin + 2 TL.* padding - dilation TL.* (kernelSize - 1) - 1) stride) + 1

type family AvgPool1DSize (lin::Nat) (kernelSize::Nat) (stride::Nat) (padding::Nat) :: Nat where
  AvgPool1DSize lin kernelSize stride padding = (Div (lin + 2 TL.* padding - kernelSize) stride) + 1

type family InsertIndex (l::[Nat]) (dim::Nat) (i::Nat) :: [Nat] where
  InsertIndex '[] 0 i   = '[i]
  InsertIndex '[] n _   = TypeError (TL.Text "Index too large for insertIndex")
  InsertIndex (h:t) 0 i = i:h:t
  InsertIndex (h:t) n i = h : InsertIndex t (n - 1) i

type family MultiplyIndex (l::[Nat]) (dim::Nat) (i::Nat) :: [Nat] where
  MultiplyIndex '[] 0 i   = '[i]
  MultiplyIndex '[] n _   = TypeError (TL.Text "Index too large for insertIndex")
  MultiplyIndex (h:t) 0 i = i TL.* h : t
  MultiplyIndex (h:t) n i = h : MultiplyIndex t (n - 1) i

-- TODO Rename me when time comes
type family ReplaceDimension (a::[Nat]) (dim::Nat) (b::Nat) :: [Nat] where
  ReplaceDimension '[]  0 x = '[x] -- TODO Is this Kosher? Should it fail?
  ReplaceDimension (h:t) 0 x = x : t
  ReplaceDimension (h:t) n x = h : ReplaceDimension t (n - 1) x

type family LengthRemaining (l::[Nat]) (i::Nat) (start::Nat) :: Nat where
  LengthRemaining l i start = SelectIndex l i - start

-- | Is a list null?
type family NonNull (a :: [Nat]) :: Bool where
  NonNull '[] = False
  NonNull _   = True

type family Narrow (l::[Nat]) (dim::Nat) (start::Nat) (len::Nat) where
  Narrow l dim start len = ReplaceDimension l dim len

-- TODO
  -- ErrorWhen (dim >= Length l)
  --                              (TL.Text "Dimension too large when narrowing")
  --                              (ErrorWhen (SelectDimension l dim < start + len)
  --                               (TL.Text "start + length is larger than the size of the tensor")
  --                               (ErrorWhen (len < 1)
  --                                 (TL.Text "length must be larger than 1")
  --                                 (ReplaceDimension l dim len)))

type family ErrorWhen (b::Bool) (s::ErrorMessage) ok where
  ErrorWhen True em _  = TypeError em
  ErrorWhen False _ ok = ok

type family IsScalarLike (l::[Nat]) :: Bool where
  IsScalarLike '[]  = True
  IsScalarLike (1:t) = IsScalarLike t
  IsScalarLike (n:_) = False

-- TODO This function is a disaster, I don't know how to write it more cleanly
caseScalar :: forall ty ki (sz :: [Nat]) a.
             Tensor ty ki sz
           -> (IsScalarLike sz ~ True => Tensor ty ki sz -> a)
           -> (Tensor ty ki sz -> a)
           -> Sing sz
           -> a
caseScalar t y _ SNil = y t
caseScalar t y n s =
  case s %~ SCons (sing :: Sing 1) SNil of
    Proved Refl -> y t
    _ -> case s %~ SCons (sing :: Sing 1) (SCons (sing :: Sing 1) SNil) of
          Proved Refl -> y t
          _ -> case s %~ SCons (sing :: Sing 1) (SCons (sing :: Sing 1) (SCons (sing :: Sing 1) SNil)) of
                Proved Refl -> y t
                _ -> case s %~ SCons (sing :: Sing 1) (SCons (sing :: Sing 1) (SCons (sing :: Sing 1) (SCons (sing :: Sing 1) SNil))) of
                      Proved Refl -> y t
                      _ -> case s %~ SCons (sing :: Sing 1) (SCons (sing :: Sing 1)
                                                            (SCons (sing :: Sing 1)
                                                             (SCons (sing :: Sing 1)
                                                              (SCons (sing :: Sing 1) SNil)))) of
                            Proved Refl -> y t
                            _ -> n t

type family DivRoundUp (a::Nat) (b::Nat) :: Nat where
  DivRoundUp a b = Div (a + b - 1) b

type family Cat2 (a::[Nat]) (b::[Nat]) (dim::Nat) :: [Nat] where
  Cat2 '[] '[] _ = TypeError (TL.Text "Can't cat two scalars")
  Cat2 (h:t) (h':t)  0 = h+h':t
  Cat2 (h:t) (h':t') 0 = TypeError (TL.Text "Tensors must agree on all of the dimensions aside from the one being concatenated")
  Cat2 (h:t) (h:t')  n = Cat2 t t' (n-1)
  Cat2 (h:_) (h':_)  n = TypeError (TL.Text "Tensors must agree on all of the dimensions aside from the one being concatenated")

type family SelectDimension (a :: [Nat]) (i::Nat) :: Nat where
  SelectDimension '[]  0   = 1 -- TODO Shoud this fail?
  SelectDimension '[1] 0   = 1 -- TODO Shoud this fail?
  SelectDimension '[_] _   = TypeError (TL.Text "Can't select out of bounds dimension")
  SelectDimension (h:t) n = SelectDimension t (n - 1)

type family Swap (a :: [Nat]) (i::Nat) (j::Nat) :: [Nat] where
  Swap l i j = ReplaceDimension (ReplaceDimension l i (SelectDimension l j)) j (SelectDimension l i)

type family AdvancedIndex1 (a :: [Nat]) (b :: [Nat]) :: [Nat] where
  AdvancedIndex1 '[] '[]  = '[]
  AdvancedIndex1 '[] b    = TypeError (TL.Text "Can't index a scalar")
  AdvancedIndex1 a '[]    = a
  AdvancedIndex1 (a:as) b = b ++ as

type family Reverse' l where
  Reverse' (a:t) = Reverse' t ++ '[a]

-- This helps with some basic equational reasoning that GHC has difficulty
-- figuring out from the above.
type family BroadcastSizes a b where
  BroadcastSizes a '[] = a
  BroadcastSizes '[] a = a
  BroadcastSizes a a   = a
  BroadcastSizes a b   = Reverse (BroadcastSizes' (Reverse a) (Reverse b))

type family BroadcastSizes' (i::[Nat]) (j::[Nat]) :: [Nat] where
  BroadcastSizes' '[] j = j
  BroadcastSizes' i '[] = i
  BroadcastSizes' (x:is) (x:js) = x : BroadcastSizes' is js
  BroadcastSizes' (1:is) (x:js) = x : BroadcastSizes' is js
  BroadcastSizes' (x:is) (1:js) = x : BroadcastSizes' is js
  BroadcastSizes' _ _ = TypeError (TL.Text "Types are not broadcastable")

type family ExpandSizes a b where
  ExpandSizes a '[] = a
  ExpandSizes '[] a = a
  ExpandSizes a a   = a
  ExpandSizes a b   = Reverse (ExpandSizes' (Reverse a) (Reverse b))

type family ExpandSizes' (x :: [Nat]) (y :: [Nat]) :: [Nat] where
  ExpandSizes' '[] j = j
  ExpandSizes' i '[] = i
  ExpandSizes' (x:is) (x:js) = x : ExpandSizes' is js
  ExpandSizes' (1:is) (x:js) = x : ExpandSizes' is js
  ExpandSizes' (x:is) (1:js) = x : ExpandSizes' is js
  ExpandSizes' (x:is) (0:js) = x : ExpandSizes' is js -- NB This was -1 in PyTorch, but it's 0 for us!
  ExpandSizes' _ _ = TypeError (TL.Text "Types are not expandable")

-- Our matrix broadcast operations are greatly reduced from what torch allows
type family BroadcastMatrices (a::[Nat]) (b::[Nat]) :: [Nat] where
  BroadcastMatrices '[i,j] '[j,k] = '[i,k] -- If both arguments are 2-dimensional, the matrix-matrix product is returned.
  BroadcastMatrices '[i] '[j]     = TypeError (TL.Text "Use dot to 'multiply' 1-D matrices")
  BroadcastMatrices _ '[]         = TypeError (TL.Text "Use 'mul' to multiply with scalars")
  BroadcastMatrices '[] _         = TypeError (TL.Text "Use 'mul' to multiply scalars")
  BroadcastMatrices '[i,j] '[j]   = '[i]
  BroadcastMatrices '[i] '[i,j]   = '[j]
  BroadcastMatrices a b           = BroadcastSizes a b

type family GridRows (imagesPerRow::Nat) (images::Nat) where
  GridRows imagesPerRow images = (Div images imagesPerRow) + (If (Mod images imagesPerRow == 0) 0 1)

type family SelectIndex (l :: [Nat]) (i :: Nat) :: Nat where
  SelectIndex (h:t) 0 = h
  SelectIndex (h:t) i = SelectIndex t (i - 1)
  SelectIndex '[] _   = TypeError (TL.Text "Too many dimensions when selecting a tensor (SelectIndex)")

type family SelectOtherIndexes (l :: [Nat]) (i :: Nat) :: [Nat] where
  SelectOtherIndexes (h:t) 0 = t
  SelectOtherIndexes (h:t) i = h : SelectOtherIndexes t (i - 1)
  SelectOtherIndexes '[] _   = TypeError (TL.Text "Too many dimensions when selecting a tensor (SelectOtherIndexes)")

type family SquareBatches (l :: [Nat]) :: Bool where
  SquareBatches '[]       = False
  SquareBatches (a:'[])   = False
  SquareBatches (a:a:'[]) = True
  SquareBatches (a:b:'[]) = False
  SquareBatches (h:t)     = SquareBatches t

type family RemoveLastTwoDims (l :: [Nat]) :: [Nat] where
  RemoveLastTwoDims '[]       = '[]
  RemoveLastTwoDims (a:'[])   = '[]
  RemoveLastTwoDims (a:b:'[]) = '[]
  RemoveLastTwoDims (h:t)     = h : RemoveLastTwoDims t

type family AddDimension (l :: [Nat]) (a :: Nat) :: [Nat] where
  AddDimension '[]   a = '[a]
  AddDimension (h:t) a = h : AddDimension t a

type family EnoughIndicesForReplacement (replacement :: Bool) (sz :: Nat) (indices :: Nat) :: Bool where
  EnoughIndicesForReplacement False  s i = s <=? i
  EnoughIndicesForReplacement True _ _   = True

-- | Internal
cScalarType :: TensorType -> C.ScalarType
cScalarType TBool   = C.ScalarTypeBool
cScalarType TByte   = C.ScalarTypeByte
cScalarType TChar   = C.ScalarTypeChar
cScalarType TShort  = C.ScalarTypeShort
cScalarType TInt    = C.ScalarTypeInt
cScalarType TLong   = C.ScalarTypeLong
cScalarType THalf   = C.ScalarTypeHalf
cScalarType TFloat  = C.ScalarTypeFloat
cScalarType TDouble = C.ScalarTypeDouble

-- | Internal
cScalarType' :: Num b => TensorType -> b
cScalarType' = fromIntegral . fromEnum . cScalarType

-- | Internal, maps a Haskell storage type to a C storage type
type family TensorTyToHsC (a :: TensorType) = result | result -> a where
    TensorTyToHsC 'TBool   = CBool
    TensorTyToHsC 'TByte   = CUChar
    TensorTyToHsC 'TChar   = CChar
    TensorTyToHsC 'TShort  = CShort
    TensorTyToHsC 'TInt    = CInt
    TensorTyToHsC 'TLong   = CLong
    TensorTyToHsC 'THalf   = CUShort
    TensorTyToHsC 'TFloat  = CFloat
    TensorTyToHsC 'TDouble = CDouble

-- | Internal, maps a C storage type to a Haskell storage type
type family TensorTyToHs (a :: TensorType) = result | result -> a where
    TensorTyToHs 'TBool   = CBool -- You may wonder if this is CBool instead of Bool?
                                  -- We want all of these to have Num types, and Bool doesn't
    TensorTyToHs 'TByte   = Word8
    TensorTyToHs 'TChar   = Char
    TensorTyToHs 'TShort  = Int16
    TensorTyToHs 'TInt    = Int32
    TensorTyToHs 'TLong   = Int64
    TensorTyToHs 'THalf   = Half
    TensorTyToHs 'TFloat  = Float
    TensorTyToHs 'TDouble = Double

-- | Tensors have types, kinds, and size.
--
-- They are implemented as a pointer to a C tensor and another,
-- pontentially-null, auxiliary pointer for GC purposes. This auxiliary pointer
-- is used to make sharing explicit to GHC. If a tensor shares storage with any
-- other source it must keep a reference to the auxiliary pointer of that source
-- (note, to the auxiliary pointer not to the source itself!). This pointer is
-- never accessed and serves no other role.
--
-- It is used to make various Haskell<->Torch operations cheaper and avoid copying
-- particularly when it comes to consuming storable vectors.
data Tensor (ty :: TensorType) (ki :: TensorKind) (size :: [Nat]) where
  Tensor :: (Num (TensorTyToHs ty), Storable (TensorTyToHs ty),
            Num (TensorTyToHsC ty), Storable (TensorTyToHsC ty), SingI ty, SingI ki, SingI size) =>
    !(ForeignPtr C.CTensor) -> !(Maybe (ForeignPtr ())) -> Tensor ty ki size

-- | Hold any Tensor for heterogenous lists.
data AnyTensor = forall ty ki sz. TensorConstraints ty ki sz => AnyTensor (Tensor ty ki sz)

-- | Just a shortcut for a tensor with no dimensions. Note that the story about
-- what a scalar is, is far more complex. PyTorch has 3 scalars (zero-size
-- tensors, 1-dim tensors of size 1, and a special sclar type) but we only
-- expose two to the user. You will only see the two tensor types, never the
-- special scalar type. In any case, there is an alternative Scalar, which is
-- @Tensor ty ki '[1]@.
type Scalar ty ki = Tensor ty ki '[]

-- | A handy bundle for all of the constraints that make up a tensor. These appear in many functions.
type TensorConstraints ty ki sz = (SingI ty, SingI ki, SingI sz
                                  ,Num (TensorTyToHs ty), Storable (TensorTyToHs ty)
                                  ,Num (TensorTyToHsC ty), Storable (TensorTyToHsC ty))

-- | A handy bundle for all of the constraints that make up a scalar tensor. These appear in many functions.
type ScalarConstraints ty ki = (SingI ty, SingI ki
                               ,Num (TensorTyToHs ty), Storable (TensorTyToHs ty)
                               ,Num (TensorTyToHsC ty), Storable (TensorTyToHsC ty))

-- | You will often care about the real floating point part of tensors, this
-- lets you treat them as doubles.
-- TODO Audit all of the uses of toDouble when we support complex tensors
toDouble :: forall (ty :: TensorType). (SingI ty)
         => TensorTyToHs ty -> Double
toDouble arg = case (sing :: Sing ty) of
                   STBool   -> fromIntegral arg
                   STByte   -> fromIntegral arg
                   STChar   -> fromIntegral $ fromEnum arg
                   STShort  -> fromIntegral arg
                   STInt    -> fromIntegral arg
                   STLong   -> fromIntegral arg
                   STHalf   -> float2Double $ fromHalf arg
                   STFloat  -> float2Double arg
                   STDouble -> coerce arg

-- | Convert a double back to any tensor scalar type.
-- TODO Audit all of the uses of toDouble when we support complex tensors
fromDouble :: forall (ty :: TensorType). (SingI ty)
           => Double -> TensorTyToHs ty
fromDouble arg = case (sing :: Sing ty) of
                   STBool   -> round arg
                   STByte   -> round arg
                   STChar   -> toEnum $ round arg
                   STShort  -> round arg
                   STInt    -> round arg
                   STLong   -> round arg
                   STHalf   -> toHalf (double2Float arg)
                   STFloat  -> double2Float arg
                   STDouble -> arg

-- | Internal
toCScalar :: forall (ty :: TensorType) (ki :: TensorKind). (SingI ki, SingI ty) => TensorTyToHsC ty -> IO (ForeignPtr C.CScalar)
toCScalar arg = (case (sing :: Sing ty, demote @ki) of
                   (STBool, KCpu)    -> C.mkScalarCPUBool arg
                   (STByte, KCpu)    -> C.mkScalarCPUByte arg
                   (STChar, KCpu)    -> C.mkScalarCPUChar arg
                   (STShort, KCpu)   -> C.mkScalarCPUShort arg
                   (STInt, KCpu)     -> C.mkScalarCPUInt arg
                   (STLong, KCpu)    -> C.mkScalarCPULong arg
                   (STHalf, KCpu)    -> C.mkScalarCPUHalf $ fromIntegral arg
                   (STFloat, KCpu)   -> C.mkScalarCPUFloat arg
                   (STDouble, KCpu)  -> C.mkScalarCPUDouble arg
#if WITH_CUDA
                   (STBool, KCuda)   -> C.mkScalarCUDABool arg
                   (STByte, KCuda)   -> C.mkScalarCUDAByte arg
                   (STChar, KCuda)   -> C.mkScalarCUDAChar arg
                   (STShort, KCuda)  -> C.mkScalarCUDAShort arg
                   (STInt, KCuda)    -> C.mkScalarCUDAInt arg
                   (STLong, KCuda)   -> C.mkScalarCUDALong arg
                   (STHalf, KCuda)   -> C.mkScalarCUDAHalf $ fromIntegral arg
                   (STFloat, KCuda)  -> C.mkScalarCUDAFloat arg
                   (STDouble, KCuda) -> C.mkScalarCUDADouble arg
#endif
                ) >>= newForeignPtr C.deleteScalar

-- | Internal
toCScalarLike :: forall ty ki sz. (TensorConstraints ty ki sz)
              => Tensor ty ki sz -> TensorTyToHsC ty -> IO (ForeignPtr C.CScalar)
toCScalarLike _ s = toCScalar @ty @ki s

-- | Internal
cScalarToHs :: forall (ty :: TensorType). (SingI ty)
            => TensorTyToHsC ty -> TensorTyToHs ty
cScalarToHs arg = case (sing :: Sing ty) of
                   STBool   -> arg
                   STByte   -> fromIntegral arg
                   STChar   -> toEnum $ fromIntegral arg
                   STShort  -> fromIntegral arg
                   STInt    -> fromIntegral arg
                   STLong   -> fromIntegral arg
                   STHalf   -> fromIntegral arg
                   STFloat  -> coerce arg
                   STDouble -> coerce arg

-- | Internal
hsScalarToC :: forall (ty :: TensorType). (SingI ty)
            => TensorTyToHs ty -> TensorTyToHsC ty
hsScalarToC arg = case (sing :: Sing ty) of
                   STBool   -> arg
                   STByte   -> fromIntegral arg
                   STChar   -> fromIntegral $ fromEnum arg
                   STShort  -> fromIntegral arg
                   STInt    -> fromIntegral arg
                   STLong   -> fromIntegral arg
                   STHalf   -> getHalf arg
                   STFloat  -> coerce arg
                   STDouble -> coerce arg

-- | Internal
toDoubleC :: forall (ty :: TensorType). (SingI ty)
         => TensorTyToHsC ty -> Double
toDoubleC arg = case (sing :: Sing ty) of
                   STBool   -> fromIntegral arg
                   STByte   -> fromIntegral arg
                   STChar   -> fromIntegral arg
                   STShort  -> fromIntegral arg
                   STInt    -> fromIntegral arg
                   STLong   -> fromIntegral arg
                   STHalf   -> fromIntegral arg
                   STFloat  -> float2Double (coerce arg)
                   STDouble -> coerce arg

-- | Internal
showScalarC :: forall (ty :: TensorType). (SingI ty)
            => TensorTyToHsC ty -> String
showScalarC arg = case (sing :: Sing ty) of
                   STBool   -> show $ (fromIntegral arg :: Int)
                   STByte   -> show $ (fromIntegral arg :: Int)
                   STChar   -> show $ (fromIntegral arg :: Int)
                   STShort  -> show $ (fromIntegral arg :: Int)
                   STInt    -> show $ (fromIntegral arg :: Int)
                   STLong   -> show $ (fromIntegral arg :: Int)
                   STHalf   -> show $ arg
                   STFloat  -> show $ float2Double (coerce arg)
                   STDouble -> show $ (coerce arg :: Double)

-- | Show any haskell number that is of the storage class of a tensor. If you
-- have a number derived from a tensor and want to keep your function
-- polymorphic, this is a good idea.
showScalar :: forall (ty :: TensorType). (SingI ty)
           => TensorTyToHs ty -> String
showScalar arg = case (sing :: Sing ty) of
                   STBool   -> show arg
                   STByte   -> show arg
                   STChar   -> show arg
                   STShort  -> show arg
                   STInt    -> show arg
                   STLong   -> show arg
                   STHalf   -> show arg
                   STFloat  -> show arg
                   STDouble -> show arg

-- | Internal
fromCScalarTensor :: (IsScalarLike sz ~ True)
                  => Tensor ty ki sz -> IO (TensorTyToHsC ty)
fromCScalarTensor (Tensor p _ ) = do
  arr <- castPtr <$> C.data_ptr p
  _ <- peek arr
  r <- peek arr
  touchForeignPtr p
  pure r

-- | This is pointer equality. Two tensors that contain the same data but are
-- not clones of each other will not be equal.
instance Eq (Tensor ty ki sz) where
  (Tensor x _) == (Tensor y _) = x == y

-- | Internal

tensorPtr :: Tensor ty ki size -> ForeignPtr C.CTensor
tensorPtr (Tensor x _) = x

tensorAux :: Tensor ty ki size -> Maybe (ForeignPtr ())
tensorAux (Tensor _ y) = y

-- | Internal
cbool :: CBool -> Bool
cbool (CBool 0) = False
cbool _         = True

-- | Internal
boolc :: Bool -> CBool
boolc False = CBool 0
boolc True  = CBool 1

-- * Various useful constructors for named variables (both types and
-- * values). These help us organize the many arguments that functions
-- * take. It's a simple way to have named arguments.

data Stride x = Stride
stride_ :: forall x. Stride x
stride_ = Stride
data Padding x = Padding
padding_ :: forall x. Padding x
padding_ = Padding
data Dilation x = Dilation
dilation_ :: forall x. Dilation x
dilation_ = Dilation
data Groups x = Groups
groups_ :: forall x. Groups x
groups_ = Groups
data Kernel x = Kernel
kernel_ :: forall x. Kernel x
kernel_ = Kernel
data OutChannels x = OutChannels
outChannels_ :: forall x. OutChannels x
outChannels_ = OutChannels
outC_ :: forall x. OutChannels x
outC_ = OutChannels
data InChannels x = InChannels
inChannels_ :: forall x. InChannels x
inChannels_ = InChannels
inC_ :: forall x. InChannels x
inC_ = InChannels
data InFeatures x = InFeatures
inFeatures_ :: forall x. InFeatures x
inFeatures_ = InFeatures
inF_ :: forall x. InFeatures x
inF_ = InFeatures
data OutFeatures x = OutFeatures
outFeatures_ :: forall x. OutFeatures x
outFeatures_ = OutFeatures
outF_ :: forall x. OutFeatures x
outF_ = OutFeatures
data HiddenFeatures x = HiddenFeatures
hiddenFeatures_ :: forall x. HiddenFeatures x
hiddenFeatures_ = HiddenFeatures
hiddenF_ :: forall x. HiddenFeatures x
hiddenF_ = HiddenFeatures
data CeilMode x = CeilMode
ceilMode_ :: forall x. CeilMode x
ceilMode_ = CeilMode
data BatchSize x = BatchSize
batchSize_ :: forall x. BatchSize x
batchSize_ = BatchSize
data Size x = Size
size_ :: forall x. Size x
size_ = Size
data Sections x = Sections
sections_ :: forall x. Sections x
sections_ = Sections
data Dimension x = Dimension
dimension_ :: forall x. Dimension x
dimension_ = Dimension
data NrLayers x = NrLayers
nrLayers_ :: forall x. NrLayers x
nrLayers_ = NrLayers
data IsBidirectional x = IsBidirectional
isBidirectional_ :: forall x. IsBidirectional x
isBidirectional_ = IsBidirectional
data NrEmbeddings x = NrEmbeddings
nrEmbeddings_ :: forall x. NrEmbeddings x
nrEmbeddings_ = NrEmbeddings
data EmbeddingDimensions x = EmbeddingDimensions
embeddingDimensions_ :: forall x. EmbeddingDimensions x
embeddingDimensions_ = EmbeddingDimensions
embeddingDims_ :: forall x. EmbeddingDimensions x
embeddingDims_ = EmbeddingDimensions
data Chunks x = Chunks
chunks_ :: forall x. Chunks x
chunks_ = Chunks
data Replacement x = Replacement
replacement_ :: forall x. Replacement x
replacement_ = Replacement

data SizeAverage x = SizeAverage x
data RescaleWeights x = RescaleWeights x
data BNMomentum x = BNMomentum x
data BNEpsilon x = BNEpsilon x

instance Default (SizeAverage Bool) where
  def = SizeAverage True

instance Default (BNMomentum Double) where
  def = BNMomentum 0.1

instance Default (BNEpsilon Double) where
  def = BNEpsilon 1e-5

-- * Define "layers"
--
-- These are pieces of networks that hold values which later operations
-- use. Many operations logically hold multiple related values, like a liner
-- layer has weights and biases. Keeping all of them separate but in sync is a
-- mess. Instead, we create a small abstraction that ties these shapes
-- together. Note that biases are still optional. Many other mechanisms hook
-- into these definitions. For example, saving and loading tensors and
-- models. If you create more layers that have relationships between multiple
-- arguments add them here. Note that not all arguments are optimizable, for
-- example, BatchNorm, appears here and the mechanism is used to store internal
-- state rather than optimizable values.
--
-- When adding layers you should make a Generic instance here and then custom
-- instances in StoredModel (ParameterNames,Stored) and Initialization
-- (ToTensors, ToParameters, Initialize, ShowIO), as in
--
-- deriving (ParameterNames,Stored,ToTensors,ToParameters,Initialize,ShowIO)

data LinearParam ty ki inF outF =
  LinearParam !(Tensor ty ki '[outF, inF]) !(Maybe (Tensor ty ki '[outF]))
  deriving(Generic)

data ConvParam ty ki outChans otherSz =
  ConvParam !(Tensor ty ki (outChans : otherSz)) !(Maybe (Tensor ty ki '[outChans]))
  deriving(Generic)

data AffineParam ty ki sz =
  AffineParam !(Tensor ty ki sz) !(Maybe (Tensor ty ki sz))
  deriving(Generic)

data BatchNormState ty ki sz =
  BatchNormState !(IORef (BatchNormData (Tensor ty ki sz)))
  deriving(Generic)

data BatchNormData t = BatchNormData { bnMean     :: !(Maybe t)
                                     , bnVariance :: !(Maybe t) }
  deriving(Generic)

instance {-# OVERLAPPING #-} Default (IO (BatchNormState ty ki sz)) where
  def = BatchNormState <$> newIORef (BatchNormData Nothing Nothing)

data EmbeddingParam ty ki nrEmbeddings embeddingDim =
  EmbeddingParam !(Tensor ty ki '[nrEmbeddings, embeddingDim])
  deriving(Generic)

-- * Recurrent layers
--
-- We define a family of current layers which includes RNNs, LSTMs, and
-- GRUs. They all share the same structure with a single gate size parameter
-- that changes between them (RNN = 1, LSTM = 4, GRU = 3).

type family NrOfRNNDirections (isBidirectional :: Bool) = r | r -> isBidirectional where
  NrOfRNNDirections True  = 2
  NrOfRNNDirections False = 1

data BiDirectional x  = BiDirectional x x
  deriving (Generic, Functor, Foldable, Traversable, Show)
data UniDirectional x = UniDirectional x
  deriving (Generic, Functor, Foldable, Traversable, Show)

-- | This is a rather complicated setup for the RNNs. We want to keep either a
-- single weight, if the RNN is unidirectional, or pairs of weights, if the RNN
-- is bidirectional. We want the types to verify that we're doing the right
-- thing and aren't missing any corner cases. PerDirection is a type family that
-- takes care of this either we store one value or two depending on
-- isBidirectional.
type family PerDirection (isBidirectional :: Bool) = r | r -> isBidirectional where
  PerDirection True  = BiDirectional
  PerDirection False = UniDirectional

mapMDirection :: forall isBidirectional m x y. (Monad m, SingI isBidirectional)
              => (x -> m y) -> PerDirection isBidirectional x -> m (PerDirection isBidirectional y)
mapMDirection f x = case sing :: Sing isBidirectional of
                      STrue  -> mapM f x
                      SFalse -> mapM f x

zipWithMDirection :: forall isBidirectional m x y z. (Monad m, SingI isBidirectional)
                  => (x -> z -> m y) -> PerDirection isBidirectional x -> [z] -> m (PerDirection isBidirectional y)
zipWithMDirection f x l = case (sing :: Sing isBidirectional, x, l) of
                        (STrue, BiDirectional x x', [z,z']) -> BiDirectional <$> f x z <*> f x' z'
                        (SFalse, UniDirectional x, [z])     -> UniDirectional <$> f x z

mapDirection :: forall isBidirectional x y. (SingI isBidirectional)
             => (x -> y) -> PerDirection isBidirectional x -> PerDirection isBidirectional y
mapDirection f x = case sing :: Sing isBidirectional of
                      STrue  -> fmap f x
                      SFalse -> fmap f x

repeatMDirection :: forall isBidirectional m x. (Monad m, SingI isBidirectional)
                 => m x -> m (PerDirection isBidirectional x)
repeatMDirection x = case sing :: Sing isBidirectional of
                      STrue  -> BiDirectional <$> x <*> x
                      SFalse -> UniDirectional <$> x

fromListMDirection :: forall isBidirectional m y z. (Monad m, SingI isBidirectional)
                   => (z -> m y) -> [z] -> m (PerDirection isBidirectional y)
fromListMDirection f l = case (sing :: Sing isBidirectional, l) of
                        (STrue, [z,z']) -> BiDirectional <$> f z <*> f z'
                        (SFalse, [z])   -> UniDirectional <$> f z

toListDirection :: forall isBidirectional x. (SingI isBidirectional)
                => PerDirection isBidirectional x -> [x]
toListDirection x = case sing :: Sing isBidirectional of
                      STrue -> let (BiDirectional a a') = x
                              in [a,a']
                      SFalse -> let (UniDirectional a) = x
                               in [a]

-- | This is internal, you won't be using it directly unless you build a new RNN
-- that is somewhat compatible with RNNs/LSTMs/GRUs. There are some strange
-- parameters that don't seem to have anything to do with the content.
-- nrLayers & batchFirst in particular.
--
-- Both of these are needed to compact the model weights, so we need them here.
data GenericRNNParam ty ki gateSize inF hiddenF (nrLayers :: Nat) isBidirectional (batchFirst :: Bool) =
  GenericRNNParam
           { -- The first layer is special, it needs to process the input
             grnnParamWih0  :: !(PerDirection isBidirectional (Tensor ty ki '[gateSize TL.* hiddenF, inF]))
             -- All other layers have size that processes only the hidden state
             -- rnnParamWihN1 stores 1 less element than all of the other vectors because
             -- its zero elements are in rnnParamWih0!
             -- TODO This vector is not statically-sized, but we know its size.
           , grnnParamWihN1 :: !(VB.Vector        (PerDirection isBidirectional (Tensor ty ki '[gateSize TL.* hiddenF
                                                                                             ,NrOfRNNDirections isBidirectional TL.* hiddenF])))
           , grnnParamBihN  :: !(Maybe (VB.Vector (PerDirection isBidirectional (Tensor ty ki '[gateSize TL.* hiddenF]))))
           , grnnParamWhhN  :: !(VB.Vector        (PerDirection isBidirectional (Tensor ty ki '[gateSize TL.* hiddenF, hiddenF])))
           , grnnParamBhhN  :: !(Maybe (VB.Vector (PerDirection isBidirectional (Tensor ty ki '[gateSize TL.* hiddenF]))))
           }
  deriving(Generic)

data RNNParams ty ki inF hiddenF nrLayers isBidirectional batchFirst =
  RNNParams (GenericRNNParam ty ki 1 inF hiddenF nrLayers isBidirectional batchFirst)
  deriving(Generic)

data LSTMParams ty ki inF hiddenF nrLayers isBidirectional batchFirst =
  LSTMParams (GenericRNNParam ty ki 4 inF hiddenF nrLayers isBidirectional batchFirst)
  deriving(Generic)

data GRUParams ty ki inF hiddenF nrLayers isBidirectional batchFirst =
  GRUParams (GenericRNNParam ty ki 3 inF hiddenF nrLayers isBidirectional batchFirst)
  deriving(Generic)

-- | We use the same state for GRUs. These have standard instances and can be
-- used in your structures but they don't expose their internals as optimizable
-- parameters.
data RNNState ty ki batch isBidirectional nrLayers hiddenF =
  RNNState { rnnHiddenState :: Tensor ty ki '[NrOfRNNDirections isBidirectional TL.* nrLayers, batch, hiddenF] }
  deriving(Generic)

-- | We use the same state for GRUs. These have standard instances and can be
-- used in your structures but they don't expose their internals as optimizable
-- parameters.
data RNNStateBatchFirst ty ki batch isBidirectional nrLayers hiddenF =
  RNNStateBatchFirst { rnnHiddenStateBatchFirst :: Tensor ty ki '[batch, NrOfRNNDirections isBidirectional TL.* nrLayers, hiddenF] }
  deriving(Generic)

-- | These have standard instances and can be used in your structures but they
-- don't expose their internals as optimizable parameters.
data LSTMState ty ki batch isBidirectional nrLayers hiddenF =
  LSTMState { lstmHiddenState :: Tensor ty ki '[batch, NrOfRNNDirections isBidirectional TL.* nrLayers, hiddenF]
            , lstmCellState   :: Tensor ty ki '[batch, NrOfRNNDirections isBidirectional TL.* nrLayers, hiddenF] }
  deriving(Generic)

-- | These have standard instances and can be used in your structures but they
-- don't expose their internals as optimizable parameters.
data LSTMStateBatchFirst ty ki batch isBidirectional nrLayers hiddenF =
  LSTMStateBatchFirst
            { lstmHiddenStateBatchFirst :: Tensor ty ki '[NrOfRNNDirections isBidirectional TL.* nrLayers, batch, hiddenF]
            , lstmCellStateBatchFirst   :: Tensor ty ki '[NrOfRNNDirections isBidirectional TL.* nrLayers, batch, hiddenF] }
  deriving(Generic)

-- * Single-cell recurrent networks
-- This is a more low-level interface than the one above

data RNNCellParam ty ki inF hiddenF =
  RNNCellParam { rnnCellParamWih :: !(Tensor ty ki '[hiddenF, inF])
               , rnnCellParamBih :: !(Maybe (Tensor ty ki '[hiddenF]))
               , rnnCellParamWhh :: !(Tensor ty ki '[hiddenF, hiddenF])
               , rnnCellParamBhh :: !(Maybe (Tensor ty ki '[hiddenF])) }
  deriving(Generic)

unRNNCellParam :: RNNCellParam ty ki inF hiddenF
               -> IO (Tensor ty ki '[hiddenF, inF],     Maybe (Tensor ty ki '[hiddenF])
                    ,Tensor ty ki '[hiddenF, hiddenF], Maybe (Tensor ty ki '[hiddenF]))
unRNNCellParam (RNNCellParam wih bih whh bhh) =
  pure $ (wih,bih,whh,bhh)

-- * Working with heterogenous values in optimizers.

-- | Any optimizer has to let us perform one step of the optimization in a
-- destructive way over the parameters of the network. Non-destructive
-- operations else would be too inefficient and pointless to implement.
class Optimizer a where
  step_ :: a -> IO a

-- | Hold any optimizer, useful for collections of optimizers. We model networks
-- with many parameters as having collections of optimizers.
data AnyOptimizer = forall a. Optimizer a => AnyOptimizer a

instance Optimizer AnyOptimizer where
  step_ (AnyOptimizer a) = AnyOptimizer <$> step_ a

-- | Many optimizers have learning rates. Having access to this rate and tuning
-- it can be critical.
class Optimizer a => OptimizerWithLR a where
  getLearningRate  :: a -> IO Float
  setLearningRate_ :: a -> Float -> IO a

-- | Hold any optimizer that has a learning rate associated with it, useful for
-- collections of optimizers or models with many parameters.
data AnyOptimizerWithLR = forall a. OptimizerWithLR a => AnyOptimizerWithLR a

instance Optimizer AnyOptimizerWithLR where
  step_ (AnyOptimizerWithLR a) = AnyOptimizerWithLR <$> step_ a

instance OptimizerWithLR AnyOptimizerWithLR where
  getLearningRate (AnyOptimizerWithLR a)     = getLearningRate a
  setLearningRate_ (AnyOptimizerWithLR a) lr = AnyOptimizerWithLR <$> setLearningRate_ a lr

-- | Technically this more properly should go into Torch.Datasets.Types but that
-- would cause a cyclic dependency. Optimizers and many other operations behave
-- differently at test and training time. We tag data with its type to allow
-- these layers to know what to do. In addition, this makes sure that you don't
-- test on the training data by mistake.
data DataPurpose = Train | Test
  deriving (Show, Eq)

genSingletons [''DataPurpose]

-- * Debugging

-- | You should never need this. It verifies the runtime shape vs the shape we
-- compute statically in the types. It's only useful for testing that the types
-- are correctly written. It is used internally in various operations that we
-- are debugging and in various operations where there is a possibilty that user
-- error will lead to the tensors being read back from C having the wrong size
-- (for example, when doing file I/O).

-- TODO Better error messages
debuggingVerifyShape :: forall ty ki sz. Text -> Tensor ty ki sz -> Tensor ty ki sz
debuggingVerifyShape msg t@(Tensor p _) = unsafePerformIO $ do
  tys' <- C.getTypeString p
  let tys = cvarname (demote @ty) (demote @ki)
  if tys' == "UndefinedType" then
    pure t else do
    let sz = demoteNv @sz
    sz' <- C.shape p
    unless (sz == sz')   $ error $ "Shape mismatch! types expect " ++ show sz ++ " but the computed shape is " ++ show sz' ++ "\n" ++ T.unpack msg
    unless (tys == tys') $ error $ "Type mismatch! types expect " ++ show tys ++ " but the computed type is " ++ show tys' ++ "\n" ++ T.unpack msg
    pure t
  where
    cvarname TBool   KCpu  = "CPU Bool"
    cvarname TByte   KCpu  = "CPU Byte"
    cvarname TChar   KCpu  = "CPU Char"
    cvarname TShort  KCpu  = "CPU Short"
    cvarname TInt    KCpu  = "CPU Int"
    cvarname TLong   KCpu  = "CPU Long"
    cvarname THalf   KCpu  = "CPU Half"
    cvarname TFloat  KCpu  = "CPU Float"
    cvarname TDouble KCpu  = "CPU Double"
#if WITH_CUDA
    cvarname TBool   KCuda = "CUDA Bool"
    cvarname TByte   KCuda = "CUDA Byte"
    cvarname TChar   KCuda = "CUDA Char"
    cvarname TShort  KCuda = "CUDA Short"
    cvarname TInt    KCuda = "CUDA Int"
    cvarname TLong   KCuda = "CUDA Long"
    cvarname THalf   KCuda = "CUDA Half"
    cvarname TFloat  KCuda = "CUDA Float"
    cvarname TDouble KCuda = "CUDA Double"
#endif

-- | Internal. This is an internal test. TODO Move me
debuggingPrintADGraph :: Tensor ty ki size -> IO ()
debuggingPrintADGraph loss = do
  fn <- C.gradient_function (tensorPtr loss)
  printADGraphLoop fn
  where
    printADGraphLoop fn = do
      if nullPtr == fn then
        pure () else
        do
          fname <- C.function_name fn
          print fname
          ins <- C.function_nr_inputs fn
          outs <- C.function_nr_outputs fn
          print (ins, outs)
          es <- C.function_next_edges fn
          V.mapM_ (\e -> do
                      efn <- C.edge_function e
                      printADGraphLoop efn) es

-- * Default types
-- These are used by -fplugin Plugin.DefaultType
-- Types are gussed in reverse order from the list here.

instance DefaultType Nat 1024
instance DefaultType Nat 512
instance DefaultType Nat 256
instance DefaultType Nat 128
instance DefaultType Nat 64
instance DefaultType Nat 32
instance DefaultType Nat 16
instance DefaultType Nat 15
instance DefaultType Nat 14
instance DefaultType Nat 13
instance DefaultType Nat 12
instance DefaultType Nat 11
instance DefaultType Nat 10
instance DefaultType Nat 9
instance DefaultType Nat 8
instance DefaultType Nat 7
instance DefaultType Nat 6
instance DefaultType Nat 5
instance DefaultType Nat 4
instance DefaultType Nat 3
instance DefaultType Nat 2
instance DefaultType Nat 1
instance DefaultType Nat 0

-- Prefer CUDA if available
#if WITH_CUDA
instance DefaultType TensorKind KCuda
#endif
instance DefaultType TensorKind KCpu

instance DefaultType TensorType TInt
instance DefaultType TensorType TLong
instance DefaultType TensorType TDouble
instance DefaultType TensorType TFloat
instance DefaultType TensorType TBool
instance DefaultType TensorType TByte
instance DefaultType TensorType TChar
instance DefaultType TensorType TShort
instance DefaultType TensorType THalf
