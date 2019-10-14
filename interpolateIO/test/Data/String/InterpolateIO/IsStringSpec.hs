{-# LANGUAGE QuasiQuotes #-}
module Data.String.InterpolateIO.IsStringSpec (main, spec) where

import           Test.Hspec

import qualified Data.Text as T
import           Data.String.InterpolateIO.IsString
import           System.IO.Unsafe

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "[c|...|]" $ do
    it "can be used to construct String literals" $ do
      (unsafePerformIO [c|foo #{23 :: Int} bar|]) `shouldBe` "foo 23 bar"
    it "can be used to construct Text literals" $ do
      (unsafePerformIO [c|foo #{23 :: Int} bar|]) `shouldBe` T.pack "foo 23 bar"
