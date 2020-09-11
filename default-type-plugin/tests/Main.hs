{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP, DataKinds, ScopedTypeVariables, TypeOperators, TypeApplications, TypeFamilies, DataKinds, PolyKinds, PartialTypeSignatures, AllowAmbiguousTypes, NoMonomorphismRestriction, FlexibleInstances #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=10 #-}
{-# OPTIONS_GHC -fplugin Data.DefaultType #-}

import Data.DefaultType
import GHC.TypeLits

data X = Y | Z

instance DefaultType X Z

data M (a :: X) = M

instance Show X where
  show Y = "Y"
  show Z = "Z"

instance (W (M a)) => Show (M a) where
  show a = s a

class W y where
  w :: y
  s :: y -> String

instance W (M Y) where
  w = M
  s _ = "y"

instance W (M Z) where
  w = M
  s _ = "z"

-- TODO Test that without this type signature we have an ambiguity. We have no
-- way of knowing that an M is involved, so we can't default anything
u :: (W (M x)) => M Y -> M x -> String
u a b = s a ++ s b

g = u w w

-- TODO Test that this remains as
-- q :: M a
-- and doesn't get defaulted, it's not ambiguous
q = M
-- This must be defaulted
o = print q
