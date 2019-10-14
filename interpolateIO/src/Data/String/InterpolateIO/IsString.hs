{-# LANGUAGE TemplateHaskell #-}
module Data.String.InterpolateIO.IsString (c, fromStringIO) where

import           Data.String.ShowIO(fromStringIO)
import           Language.Haskell.TH.Quote (QuasiQuoter(..))

import qualified Data.String.InterpolateIO as I

-- |
-- Like `I.c`, but constructs a value of type
--
-- > IsString a => a
c :: QuasiQuoter
c = QuasiQuoter {
    quoteExp = \s -> [|fromStringIO =<< $(quoteExp I.c $ s)|]
  , quotePat = err "pattern"
  , quoteType = err "type"
  , quoteDec = err "declaration"
  }
  where
    err name = error ("Data.String.Interpolate.IsString.c: This QuasiQuoter can not be used as a " ++ name ++ "!")
