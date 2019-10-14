{-# LANGUAGE CPP #-}
module Data.String.InterpolateIO.Compat (
  readMaybe
, module Language.Haskell.TH
) where

import           Language.Haskell.TH
import           Text.Read

#if !MIN_VERSION_base(4,6,0)
import qualified Text.ParserCombinators.ReadP as P
#endif

#if !MIN_VERSION_base(4,6,0)
-- | Parse a string using the 'Read' instance.
-- Succeeds if there is exactly one valid result.
-- A 'Left' value indicates a parse error.
readEither :: Read a => String -> Either String a
readEither s =
  case [ x | (x,"") <- readPrec_to_S read' minPrec s ] of
    [x] -> Right x
    []  -> Left "Prelude.read: no parse"
    _   -> Left "Prelude.read: ambiguous parse"
 where
  read' =
    do x <- readPrec
       lift P.skipSpaces
       return x

-- | Parse a string using the 'Read' instance.
-- Succeeds if there is exactly one valid result.
readMaybe :: Read a => String -> Maybe a
readMaybe s = case readEither s of
                Left _  -> Nothing
                Right a -> Just a
#endif
