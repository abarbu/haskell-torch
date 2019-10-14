{-# LANGUAGE StandaloneDeriving #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
module Data.String.InterpolateIO.ParseSpec (main, spec) where

import           Test.Hspec

import           Data.String.InterpolateIO.Parse

deriving instance Eq Node
deriving instance Show Node

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "parseNodes" $ do
    it "parses string literals" $ do
      parseNodes "foo" `shouldBe` [Literal "foo"]

    it "parses embedded expressions" $ do
      parseNodes "foo #{bar} baz" `shouldBe` [Literal "foo ", Expression "bar", Literal " baz"]

    context "when given an unterminated expression" $ do
      it "parses it as a string literal" $ do
        parseNodes "foo #{bar" `shouldBe` [Literal "foo #{bar"]
