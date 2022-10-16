{-# LANGUAGE AllowAmbiguousTypes, ConstraintKinds, ExtendedDefaultRules, FlexibleContexts, FlexibleInstances, GADTs, OverloadedStrings, MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds, QuasiQuotes, RankNTypes, ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies                  #-}
{-# LANGUAGE TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances                                                   #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver -fplugin Plugin.DefaultType #-}

-- | This example shows you Haskell & PyTorch code right next to one
-- another. You get an idea of how the two are related and how to do so some of
-- he most basic operations.

module Torch.Tutorial.RL.Simple where
import           Control.Monad
import           Data.Default
import           Data.Kind
import           Data.Maybe
import           Data.Singletons
import           Data.String.InterpolateIO
import qualified Data.Vector                 as V'
import           Data.Vector.Storable        (Vector)
import qualified Data.Vector.Storable        as V
import           Foreign.C.Types
import           Pipes
import qualified Pipes.Prelude               as P
import           Torch
import qualified Data.Vector.Storable as VS
import qualified Simulator.Gym as G
