{-# LANGUAGE QuasiQuotes #-}
module Data.String.InterpolateIOSpec (main, spec) where

import           Test.Hspec
import           Test.QuickCheck
import           System.IO.Unsafe

import           Data.String.InterpolateIO

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "[c|...|]" $ do
    it "interpolates an expression of type Int" $ do
      property $ \x y -> unsafePerformIO [c|foo #{x + y :: Int} bar|] `shouldBe` "foo " ++ show (x + y) ++ " bar"

    it "interpolates an expression of type String" $ do
      property $ \xs ys -> unsafePerformIO [c|foo #{xs ++ ys} bar|] `shouldBe` "foo " ++ xs ++ ys ++ " bar"

    it "accepts character escapes" $ do
      unsafePerformIO [c|foo \955 bar|] `shouldBe` "foo \955 bar"

    it "accepts character escapes in interpolated expressions" $ do
      unsafePerformIO [c|foo #{"\955" :: String} bar|] `shouldBe` "foo \955 bar"

    it "dose not strip backslashes (issue #1)" $ do
      unsafePerformIO [c|foo\\bar|] `shouldBe` "foo\\bar"

    it "allows to prevent interpolation by escaping the hash with a backslash" $ do
      unsafePerformIO [c|foo \#{23 :: Int} bar|] `shouldBe` "foo #{23 :: Int} bar"

    it "does not prevent interpolation on literal backslash" $ do
      unsafePerformIO [c|foo \\#{23 :: Int} bar|] `shouldBe` "foo \\23 bar"


