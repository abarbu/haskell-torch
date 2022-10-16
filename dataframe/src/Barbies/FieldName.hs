{-# LANGUAGE AllowAmbiguousTypes, ConstraintKinds, ExtendedDefaultRules, FlexibleContexts, FlexibleInstances, GADTs, OverloadedStrings, MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds, QuasiQuotes, RankNTypes, ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies, DeriveAnyClass                  #-}
{-# LANGUAGE TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances, StandaloneDeriving, DeriveGeneric, QuantifiedConstraints                              #-}
{-# LANGUAGE CPP, DuplicateRecordFields, RecordWildCards, FunctionalDependencies, DeriveDataTypeable, DefaultSignatures #-}
module Barbies.FieldName where
import GHC.TypeLits(Symbol)
import GHC.Generics (Generic)
import qualified GHC.Generics as GHC
import Data.Proxy
import Control.Applicative
import Barbies
import Data.String(IsString(..))
import GHC.TypeLits(KnownSymbol(..),symbolVal)

class FieldNameB b where
  bfieldName' :: b p

instance FieldNameB GHC.U1 where
  bfieldName' = GHC.U1

instance (FieldNameB t) => FieldNameB (GHC.M1 GHC.C m t) where
  bfieldName' = GHC.M1 bfieldName'

instance (FieldNameB t) => FieldNameB (GHC.M1 GHC.D m t) where
  bfieldName' = GHC.M1 bfieldName'

instance (FieldNameB f, FieldNameB g) => FieldNameB (f GHC.:*: g) where
  bfieldName' = bfieldName' GHC.:*: bfieldName'

instance (m ~ 'GHC.MetaSel ('Just name) su ss ds, IsString a, KnownSymbol name) => FieldNameB (GHC.M1 GHC.S m (GHC.Rec1 (Const a))) where
  bfieldName' = GHC.M1 $ GHC.Rec1 $ Const $ fromString $ symbolVal (Proxy :: Proxy name)

instance (m ~ 'GHC.MetaSel ('Just name) su ss ds, IsString a, KnownSymbol name) => FieldNameB (GHC.M1 GHC.S m (GHC.Rec0 (Const a x))) where
  bfieldName' = GHC.M1 $ GHC.K1 $ Const $ fromString $ symbolVal (Proxy :: Proxy name)

bfieldName :: (Generic (b (Const a)), FieldNameB (GHC.Rep (b (Const a)))) => IsString a => b (Const a)
bfieldName = GHC.to bfieldName'
