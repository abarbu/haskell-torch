{-# LANGUAGE
    AllowAmbiguousTypes
  , FlexibleContexts
  , FlexibleInstances
  , MultiParamTypeClasses
  , ScopedTypeVariables
  , TypeApplications
  , TypeInType
  , TypeFamilies
  , TypeOperators
  , UndecidableInstances
#-}
{-# OPTIONS_GHC
  -fno-warn-unticked-promoted-constructors
#-}
module Generics.SOP.Record.SubTyping
  ( cast
  , IsSubTypeOf
  , IsElemOf
  , get
  , getField
  )
  where

import Data.Type.Equality
import Generics.SOP.NP
import GHC.Types

import Generics.SOP.Record

-- | Cast one record type to another if there is a subtype relationship
-- between them. Currently, only width subtyping is considered, which means
-- that we can forget and reorder fields.
--
cast :: (IsRecord a ra, IsRecord b rb, IsSubTypeOf ra rb) => a -> b
cast = fromRecord . castRecord . toRecord

-- | Extract a record field based on the symbolic name of a field.
-- Requires an explicit type application for the field name.
--
getField :: forall s a b ra . (IsRecord a ra, IsElemOf s b ra) => a -> b
getField = get @s . toRecord

-- | Class that checks whether one record code is convertible into another.
--
-- Conversion works if the first record contains at least the labels of the
-- second record, and if the types of the corresponding fields match exactly.
--
class IsSubTypeOf (r1 :: RecordCode) (r2 :: RecordCode) where
  -- | Perform a safe cast between two records.
  castRecord :: Record r1 -> Record r2

instance IsSubTypeOf r1 '[] where
  castRecord _ = Nil

instance (IsSubTypeOf r1 r2, IsElemOf s2 a2 r1) => IsSubTypeOf r1 ( '(s2, a2) : r2 ) where
  castRecord r = P (get @s2 r) :* castRecord r

-- | Class that checks whether a field of a particular type is contained
-- in a record.
--
class IsElemOf (s :: Symbol) (a :: Type) (r :: RecordCode) where
  -- | Perform an extraction of a given field. Field name has to be passed
  -- via type application.
  --
  get :: Record r -> a

-- | Helper class. Isn't strictly needed, but allows us to avoid
-- overlapping instances for the 'IsElemOf' class.
--
class IsElemOf' (b :: Bool)
  (s1 :: FieldLabel) (a1 :: Type)
  (s2 :: FieldLabel) (a2 :: Type)
  (r :: RecordCode)
  where
  get' :: Record ( '(s2, a2) : r ) -> a1

instance
  IsElemOf' (SameFieldLabel s1 s2) s1 a1 s2 a2 r =>
  IsElemOf s1 a1 ( '(s2, a2) : r )
  where
  get = get' @(SameFieldLabel s1 s2) @s1

instance (a1 ~ a2) => IsElemOf' True s a1 s a2 r where
  get' (P a :* _) = a

instance IsElemOf s1 a1 r => IsElemOf' False s1 a1 s2 a2 r where
  get' (_ :* r) = get @s1 r

-- | Decide the equality of two field labels.
--
-- Just a special case of polymorphic type equality.
--
type family
  SameFieldLabel (s1 :: FieldLabel) (s2 :: FieldLabel) :: Bool where
  SameFieldLabel s1 s2 = s1 == s2
