{-# LANGUAGE CPP #-}
module Data.String.InterpolateIO.Internal.UtilSpec where

import           Test.Hspec
import           Test.QuickCheck
import           Test.QuickCheck.Instances ()

import qualified Data.Text as T
import qualified Data.Text.Lazy as LT
import qualified Data.ByteString.Char8 as B
import qualified Data.ByteString.Lazy.Char8 as LB

import           Data.String.InterpolateIO.Internal.Util

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "toString" $ do
    it "behaves like `show`" $ do
      property $ \n -> toString (n :: Int) `shouldBe` show n

    context "when used with String" $ do
      it "behaves like `id`" $ do
        property $ \s -> toString s `shouldBe` s

    context "when used with Text" $ do
      it "behaves like `unpack`" $ do
        property $ \s -> toString s `shouldBe` T.unpack s

    context "when used with lazy Text" $ do
      it "behaves like `unpack`" $ do
        property $ \s -> toString s `shouldBe` LT.unpack s

    context "when used with ByteString" $ do
      it "behaves like `unpack`" $ do
        property $ \s -> toString s `shouldBe` B.unpack s

    context "when used with lazy ByteString" $ do
      it "behaves like `unpack`" $ do
        property $ \s -> do
#if __GLASGOW_HASKELL__ < 706
          pendingWith "Does not work with GHC < 7.6"
#endif
          toString s `shouldBe` LB.unpack s

  describe "unescape" $ do
    it "unescapes single-character escape codes" $ do
      unescape "\\n" `shouldBe` "\n"

    it "unescapes ASCII control code abbreviations" $ do
      unescape "\\BEL" `shouldBe` "\BEL"

    it "unescapes decimal character literals" $ do
      unescape "\\955" `shouldBe` "\955"

    it "unescapes hexadecimal character literals" $ do
      unescape "\\xbeef" `shouldBe` "\xbeef"

    it "unescapes octal character literals" $ do
      unescape "\\o1234" `shouldBe` "\o1234"

    context "with control escape sequences" $ do
      it "unescapes null character" $ do
        unescape "\\^@" `shouldBe` "\^@"

      it "unescapes control codes" $ do
        unescape "\\^A" `shouldBe` "\^A"

      it "unescapes escape" $ do
        unescape "\\^[" `shouldBe` "\^["

      it "unescapes file separator" $ do
        unescape "\\^\\ x" `shouldBe` "\^\ x"

      it "unescapes group separator" $ do
        unescape "\\^]" `shouldBe` "\^]"

      it "unescapes record separator" $ do
        unescape "\\^^" `shouldBe` "\^^"

      it "unescapes unit separator" $ do
        unescape "\\^_" `shouldBe` "\^_"
