{-# LANGUAGE TemplateHaskell #-}
module Data.String.InterpolateIO (
-- * String interpolation done right for ShowIO
-- |
-- The examples in this module use `QuasiQuotes`.  Make sure to enable the
-- corresponding language extension.
--
-- >>> :set -XQuasiQuotes
-- >>> import Data.String.Interpolate
-- >>> import System.Environment(getEnv, setEnv)
  c, toStringShowStringIO
) where

import           Language.Haskell.TH.Quote (QuasiQuoter(..))
import           Language.Haskell.Meta.Parse (parseExp)

import           Data.String.InterpolateIO.Internal.Util
import           Data.String.InterpolateIO.Parse
import           Data.String.InterpolateIO.Compat (Q, Exp, appE)
import           Data.String.ShowIO

-- |
-- A `QuasiQuoter` for string interpolation.  Expression enclosed within
-- @#{...}@ are interpolated, the result has to be in the `Show` class.
--
-- It interpolates strings in IO
--
-- >>> setEnv "TESTVAR" "XYZ"
-- >>> putStrLn =<< [c|lang: #{getEnv "TESTVAR"}|]
-- lang: XYZ
--
-- or integers that are pure
--
-- >>> let age = 23
-- >>> putStrLn =<< [c|age: #{age}|]
-- age: 23
--
-- or arbitrary Haskell pure or IO expressions
--
-- >>> let profession = "\955-scientist"
-- >>> putStrLn =<< [c|profession: #{unwords [name, "the", profession]}|]
-- profession: Marvin the λ-scientist
c :: QuasiQuoter
c = QuasiQuoter {
    quoteExp = toExp . parseNodes . decodeNewlines
  , quotePat = err "pattern"
  , quoteType = err "type"
  , quoteDec = err "declaration"
  }
  where
    err name = error ("Data.String.Interpolate.i: This QuasiQuoter can not be used as a " ++ name ++ "!")

    toExp:: [Node] -> Q Exp
    toExp nodes = case nodes of
      [] -> [|pure ""|]
      (x:xs) -> f x `appE` toExp xs
      where
        -- f (Literal s) = [|(\z -> showStringIO s =<< z)|]
        f (Literal s) = [|(showStringIO s =<<)|]
        -- f (Expression e) = [|((=<<) . showStringIO . toStringIO) $(reifyExpression e)|]
        -- f (Expression e) = [|(\z_ -> toStringIO $(reifyExpression e) >>= \y_ -> (showStringIO y_ =<< z_))|]
        f (Expression e) = [|toStringShowStringIO $(reifyExpression e)|]

        reifyExpression :: String -> Q Exp
        reifyExpression s = case parseExp s of
          Left _ -> do
            fail "Parse error in expression!" :: Q Exp
          Right e -> return e

toStringShowStringIO :: ShowIO a => a -> IO String -> IO String
toStringShowStringIO e n = toStringIO e >>= \y -> (showStringIO y =<< n)

decodeNewlines :: String -> String
decodeNewlines = go
  where
    go xs = case xs of
      '\r' : '\n' : ys -> '\n' : go ys
      y : ys -> y : go ys
      [] -> []
