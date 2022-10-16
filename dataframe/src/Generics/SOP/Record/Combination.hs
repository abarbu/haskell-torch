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
module Generics.SOP.Record.Combination
  ( combination
  , IsCombinationOf
  , IsElemOf2
  , get2
  , getField2
  )
  where

import Data.Type.Equality
import Generics.SOP.NP
import GHC.Types

import Generics.SOP.Record


class IsElemOf2 (s :: Symbol) (a :: Type) (r1 :: RecordCode) (r2 :: RecordCode) where
      get2 :: Record r1 -> Record r2 -> a

class IsElemOfIf2 (b :: Bool)
                  (targetSymbol :: FieldLabel) (targetType :: Type)
                  (currentSymbol :: FieldLabel) (currentType :: Type)
                  (r1 :: RecordCode) (r2 :: RecordCode) where
      get2_1' :: Record ( '(currentSymbol, currentType) : r1 ) -> Record r2 -> targetType
      get2_2' :: Record r1 -> Record ( '(currentSymbol, currentType) : r2 ) -> targetType

-- Traverse
instance IsElemOfIf2 (targetSymbol == currentSymbol) targetSymbol targetType currentSymbol currentType r1 r2
      => IsElemOf2 targetSymbol targetType ( '(currentSymbol, currentType) : r1 ) r2 where
      get2 r1 r2 = get2_1' @(targetSymbol == currentSymbol) @targetSymbol @targetType r1 r2

instance IsElemOfIf2 (targetSymbol == currentSymbol) targetSymbol targetType currentSymbol currentType '[] r2
      => IsElemOf2 targetSymbol targetType '[] ( '(currentSymbol, currentType) : r2 ) where
      get2 r1 r2 = get2_2' @(targetSymbol == currentSymbol) @targetSymbol @targetType r1 r2

instance (targetType ~ currentType) => IsElemOfIf2 True s targetType s currentType r1 r2 where
      get2_1' (P a :* _) _ = a
      get2_2' _ (P a :* _) = a

instance IsElemOf2 targetSymbol targetType r1 r2 => IsElemOfIf2 False targetSymbol targetType currentSymbol currentType r1 r2 where
      get2_1' (_ :* r1) r2 = get2 @targetSymbol @targetType r1 r2
      get2_2' r1 (_ :* r2) = get2 @targetSymbol @targetType r1 r2

getField2 :: forall s a b o ra rb. (IsRecord a ra, IsRecord b rb, IsElemOf2 s o ra rb) => a -> b -> o
getField2 r1 r2 = get2 @s (toRecord r1) (toRecord r2)

class IsCombinationOf (r1 :: RecordCode) (r2 :: RecordCode) (r :: RecordCode) where
  combinationRecords :: Record r1 -> Record r2 -> Record r

instance IsCombinationOf r1 r2 '[] where
  combinationRecords _ _ = Nil

instance (IsCombinationOf r1 r2 r, IsElemOf2 s2 a2 r1 r2) => IsCombinationOf r1 r2 ( '(s2, a2) : r ) where
  combinationRecords r1 r2 = P (get2 @s2 r1 r2) :* combinationRecords r1 r2

combination :: (IsRecord a ra, IsRecord b rb, IsRecord c rc, IsCombinationOf ra rb rc) => a -> b -> c
combination r1 r2 = fromRecord $ combinationRecords (toRecord r1) (toRecord r2)

