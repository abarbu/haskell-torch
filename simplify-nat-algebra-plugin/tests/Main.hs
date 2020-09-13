{-# LANGUAGE FlexibleInstances, FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeOperators, PartialTypeSignatures #-}
{-# LANGUAGE RankNTypes, ScopedTypeVariables, TypeOperators, TypeApplications, KindSignatures #-}
{-# LANGUAGE DataKinds, GADTs, TypeOperators #-}
{-# OPTIONS -fplugin GHC.TypeLits.KnownNat.Solver -fplugin Plugin.SimplifyNat #-}
 
import Data.Proxy
import GHC.TypeLits
import GHC.TypeNats

w :: forall (x :: Nat). KnownNat x => Proxy x -> Proxy (x + 1)
w = undefined

e :: forall (x :: Nat). KnownNat x => Proxy x -> Proxy (x - 1)
e = undefined

f :: forall (x :: Nat). ((1 <=? x) ~ 'True, KnownNat x) => Proxy x -> Proxy x
f = undefined

-- When this type is inferred you should get
-- q :: (KnownNat x, (1 <=? (x + 1)) ~ 'True, (1 <=? ((x + 1) - 1)) ~ 'True) => Proxy x -> Proxy ((x + 1) - 1)
--  without the plugin, and
--    q :: (KnownNat x, (1 <=? x) ~ 'True) => Proxy x -> Proxy ((x + 1) - 1)
--  with the plugin
q :: _ => _
q i = f $ e $ w i

main = pure 0
