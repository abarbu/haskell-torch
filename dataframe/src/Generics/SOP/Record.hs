{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fno-warn-unticked-promoted-constructors #-}
{-# OPTIONS_GHC -fno-warn-redundant-constraints #-}
module Generics.SOP.Record
  ( -- * A suitable representation for single-constructor records
    FieldLabel
  , RecordCode
  , Record
  , RecordRep
    -- * Computing the record code
  , RecordCodeOf
  , IsRecord
  , ValidRecordCode
  , ExtractTypesFromRecordCode
  , ExtractLabelsFromRecordCode
  , RecombineRecordCode
    -- * Conversion between a type and its record representation.
  , toRecord
  , fromRecord
    -- * Utilities
  , P(..)
  , Snd
  )
  where

import Control.DeepSeq
import Generics.SOP.BasicFunctors
import Generics.SOP.NP
import Generics.SOP.NS
import Generics.SOP.Universe
import Generics.SOP.Sing
import Generics.SOP.Type.Metadata
import qualified GHC.Generics as GHC
import GHC.TypeLits
import GHC.Types
import Unsafe.Coerce

--------------------------------------------------------------------------
-- A suitable representation for single-constructor records.
--------------------------------------------------------------------------

-- | On the type-level, we represent fiel labels using symbols.
type FieldLabel = Symbol

-- | The record code deviates from the normal SOP code in two
-- ways:
--
-- - There is only one list, because we require that there is
--   only a single constructor.
--
-- - In addition to the types of the fields, we store the labels
--   of the fields.
--
type RecordCode = [(FieldLabel, Type)]

-- | The record representation of a type is a record indexed
-- by the record code.
--
type RecordRep (a :: Type) = Record (RecordCodeOf a)

-- | The representation of a record is just a product indexed by
-- a record code, containing elements of the types indicated
-- by the code.
--
-- Note that the representation is deliberately chosen such that
-- it has the same run-time representation as the product part
-- of the normal SOP representation.
--
type Record (r :: RecordCode) = NP P r

--------------------------------------------------------------------------
-- Computing the record code
--------------------------------------------------------------------------

-- | This type-level function takes the type-level metadata provided
-- by generics-sop as well as the normal generics-sop code, and transforms
-- them into the record code.
--
-- Arguably, the record code is more usable than the representation
-- directly on offer by generics-sop. So it's worth asking whether
-- this representation should be included in generics-sop ...
--
-- The function will only reduce if the argument type actually is a
-- record, meaning it must have exactly one constructor, and that
-- constructor must have field labels attached to it.
--
type RecordCodeOf a = ToRecordCode_Datatype a (DatatypeInfoOf a) (Code a)

-- | Helper for 'RecordCodeOf', handling the datatype level. Both
-- datatypes and newtypes are acceptable. Newtypes are just handled
-- as one-constructor datatypes for this purpose.
--
type family
  ToRecordCode_Datatype (a :: Type) (d :: DatatypeInfo) (c :: [[Type]]) :: RecordCode where
#if MIN_VERSION_generics_sop(0,5,0)
  ToRecordCode_Datatype a (ADT _ _ cis _)  c = ToRecordCode_Constructor a cis c
#else
  ToRecordCode_Datatype a (ADT _ _ cis)    c = ToRecordCode_Constructor a cis c
#endif
  ToRecordCode_Datatype a (Newtype _ _ ci) c = ToRecordCode_Constructor a '[ ci ] c

-- | Helper for 'RecordCodeOf', handling the constructor level. Only
-- single-constructor types are acceptable, and the constructor must
-- contain field labels.
--
-- As an exception, we accept an empty record, even though it does
-- not explicitly define any field labels.
--
type family
  ToRecordCode_Constructor (a :: Type) (cis :: [ConstructorInfo]) (c :: [[Type]]) :: RecordCode where
  ToRecordCode_Constructor a '[ 'Record _ fis  ] '[ ts  ] = ToRecordCode_Field fis ts
  ToRecordCode_Constructor a '[ 'Constructor _ ] '[ '[] ] = '[]
  ToRecordCode_Constructor a '[]                  _       =
    TypeError
      (    Text "The type `" :<>: ShowType a :<>: Text "' is not a record type."
      :$$: Text "It has no constructors."
      )
  ToRecordCode_Constructor a ( _ : _ : _ )        _       =
    TypeError
      (    Text "The type `" :<>: ShowType a :<>: Text "' is not a record type."
      :$$: Text "It has more than one constructor."
      )
  ToRecordCode_Constructor a '[ _ ]               _       =
    TypeError
      (    Text "The type `" :<>: ShowType a :<>: Text "' is not a record type."
      :$$: Text "It has no labelled fields."
      )

-- | Helper for 'RecordCodeOf', handling the field level. At this point,
-- we simply zip the list of field names and the list of types.
--
type family ToRecordCode_Field (fis :: [FieldInfo]) (c :: [Type]) :: RecordCode where
  ToRecordCode_Field '[]                    '[]        = '[]
  ToRecordCode_Field ( 'FieldInfo l : fis ) ( t : ts ) = '(l, t) : ToRecordCode_Field fis ts

-- * Relating the record code and the original code.

-- | The constraint @IsRecord a r@ states that the type 'a' is a record type
-- (i.e., has exactly one constructor and field labels) and that 'r' is the
-- record code associated with 'a'.
--
type IsRecord (a :: Type) (r :: RecordCode) =
  IsRecord' a r (GetSingleton (Code a))

-- | The constraint @IsRecord' a r xs@ states that 'a' is a record type
-- with record code 'r', and that the types contained in 'r' correspond
-- to the list 'xs'.
--
-- If the record code computation is correct, then the record code of a
-- type is strongly related to the original generics-sop code. Extracting
-- the types out of 'r' should correspond to 'xs'. Recombining the
-- labels from 'r' with 'xs' should yield 'r' exactly. These sanity
-- properties are captured by 'ValidRecordCode'.
--
type IsRecord' (a :: Type) (r :: RecordCode) (xs :: [Type]) =
  ( Generic a, Code a ~ '[ xs ]
  , RecordCodeOf a ~ r, ValidRecordCode r xs
  )

-- | Relates a recordcode 'r' and a list of types 'xs', stating that
-- 'xs' is indeed the list of types contained in 'r'.
--
type ValidRecordCode (r :: RecordCode) (xs :: [Type]) =
  ( ExtractTypesFromRecordCode r ~ xs
  , RecombineRecordCode (ExtractLabelsFromRecordCode r) xs ~ r
  )

-- | Extracts all the types from a record code.
type family ExtractTypesFromRecordCode (r :: RecordCode) :: [Type] where
  ExtractTypesFromRecordCode '[]             = '[]
  ExtractTypesFromRecordCode ( '(_, a) : r ) = a : ExtractTypesFromRecordCode r

-- | Extracts all the field labels from a record code.
type family ExtractLabelsFromRecordCode (r :: RecordCode) :: [FieldLabel] where
  ExtractLabelsFromRecordCode '[]             = '[]
  ExtractLabelsFromRecordCode ( '(l, _) : r ) = l : ExtractLabelsFromRecordCode r

-- | Given a list of labels and types, recombines them into a record code.
--
-- An important aspect of this function is that it is defined by induction
-- on the list of types, and forces the list of field labels to be at least
-- as long.
--
type family RecombineRecordCode (ls :: [FieldLabel]) (ts :: [Type]) :: RecordCode where
  RecombineRecordCode _ '[]       = '[]
  RecombineRecordCode ls (t : ts) = '(Head ls, t) : RecombineRecordCode (Tail ls) ts

--------------------------------------------------------------------------
-- Conversion between a type and its record representation.
--------------------------------------------------------------------------

-- | Convert a value into its record representation.
toRecord :: (IsRecord a _r) => a -> RecordRep a
toRecord = unsafeToRecord_NP . unZ . unSOP . from

-- | Convert an n-ary product into the corresponding record
-- representation. This is a no-op, and more efficiently
-- implented using 'unsafeToRecord_NP'. It is included here
-- to demonstrate that it actually is type-correct and also
-- to make it more obvious that it is indeed a no-op.
--
_toRecord_NP :: (ValidRecordCode r xs) => NP I xs -> Record r
_toRecord_NP Nil         = Nil
_toRecord_NP (I x :* xs) = P x :* _toRecord_NP xs

-- | Fast version of 'toRecord_NP'. Not actually unsafe as
-- long as the internal representations of 'NP' and 'Record'
-- are not changed.
--
unsafeToRecord_NP :: (ValidRecordCode r xs) => NP I xs -> Record r
unsafeToRecord_NP = unsafeCoerce

-- | Convert a record representation back into a value.
fromRecord :: forall a r . (IsRecord a r) => RecordRep a -> a
fromRecord = fromRecord'
  where
    fromRecord' :: forall xs . (IsRecord' a r xs) => RecordRep a -> a -- extra type signature should not be necessary, see GHC #21515
    fromRecord' = to . SOP . Z . unsafeFromRecord_NP

-- | Convert a record representation into an n-ary product. This is a no-op,
-- and more efficiently implemented using 'unsafeFromRecord_NP'.
--
-- It is also noteworthy that we let the resulting list drive the computation.
-- This is compatible with the definition of 'RecombineRecordCode' based on
-- the list of types.
--
_fromRecord_NP :: forall r xs . (ValidRecordCode r xs, SListI xs) => Record r -> NP I xs
_fromRecord_NP = case sList :: SList xs of
  SNil  -> const Nil
  SCons -> \ r -> case r of
    P x :* xs -> I x :* _fromRecord_NP xs

-- | Fast version of 'fromRecord_NP'. Not actually unsafe as
-- long as the internal representation of 'NP' and 'Record'
-- are not changed.
--
unsafeFromRecord_NP :: forall r xs . (ValidRecordCode r xs, SListI xs) => Record r -> NP I xs
unsafeFromRecord_NP = unsafeCoerce

--------------------------------------------------------------------------
-- Utilities
--------------------------------------------------------------------------

-- | Projection of the second component of a type-level pair,
-- wrapped in a newtype.
--
newtype P (p :: (a, Type)) = P (Snd p)
  deriving (GHC.Generic)

deriving instance Eq a => Eq (P '(l, a))
deriving instance Ord a => Ord (P '(l, a))
deriving instance Show a => Show (P '(l, a))

instance NFData a => NFData (P '(l, a)) where
  rnf (P x) = rnf x

-- | Type-level variant of 'snd'.
type family Snd (p :: (a, b)) :: b where
  Snd '(a, b) = b

-- | Type-level variant of 'head'.
type family Head (xs :: [k]) :: k where
  Head (x : xs) = x

-- | Type-level variant of 'tail'.
type family Tail (xs :: [k]) :: [k] where
  Tail (x : xs) = xs

-- | Partial type-level function that extracts the only element
-- from a singleton type-level list.
--
type family GetSingleton (xs :: [k]) :: k where
  GetSingleton '[ x ] = x
