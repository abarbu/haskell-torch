module Data.String.InterpolateIO.UtilSpec (main, spec) where

import           Prelude ()
import           Prelude.Compat

import           Test.Hspec
import           Test.QuickCheck

import           Data.String.InterpolateIO.Util

main :: IO ()
main = hspec spec

emptyLine :: Gen String
emptyLine = (++ "\n") <$> listOf (elements " \t")

spec :: Spec
spec = do
  describe "unindent" $ do
    it "removes indentation" $ do
      let xs = "    foo\n  bar\n   baz  \n"
      unindent xs `shouldBe` "  foo\nbar\n baz  \n"

    it "removes the first line of the string if it is empty" $ do
      forAll emptyLine $ \xs -> do
        let ys = "  foo\nbar\n baz\n"
        unindent (xs ++ ys) `shouldBe` ys

    it "does not affect additional empty lines at the beginning" $ do
      unindent "  \n  \nfoo" `shouldBe` "  \nfoo"

    it "empties the last line if it only consists of spaces" $ do
      let xs = "foo\n  "
      unindent xs `shouldBe` "foo\n"

    it "does not affect other whitespace lines at the end" $ do
      unindent "foo\n  \n  " `shouldBe` "foo\n  \n"

    it "disregards empty lines when calculating indentation" $ do
      let xs = "  foo\n\n \n  bar\n"
      unindent xs `shouldBe` "foo\n\n\nbar\n"

    it "correctly handles strings that do not end with a newline" $ do
      let xs = "foo"
      unindent xs `shouldBe` xs

    it "does not affect lines consisting of whitespace (apart from unindenting)" $ do
      unindent " foo\n  \n bar" `shouldBe` "foo\n \nbar"

    it "is total" $ do
      property $ \xs -> length (unindent xs) `shouldSatisfy` (>= 0)

    context "when all lines are empty" $ do
      it "does not unindent at all" $ do
        forAll emptyLine $ \x -> (forAll $ listOf emptyLine) $ \xs -> do
          let ys = concat xs
          unindent (x ++ ys) `shouldBe` ys
