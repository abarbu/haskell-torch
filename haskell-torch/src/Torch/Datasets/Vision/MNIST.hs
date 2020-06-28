{-# LANGUAGE AllowAmbiguousTypes, ConstraintKinds, DataKinds, FlexibleContexts, FlexibleInstances, GADTs, MultiParamTypeClasses   #-}
{-# LANGUAGE NoStarIsType, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, RankNTypes, ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications, TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances, IncoherentInstances       #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -Wall -fno-warn-name-shadowing -Wno-unticked-promoted-constructors -Wno-unused-matches -Wno-type-defaults #-}

-- | MNIST <http://yann.lecun.com/exdb/mnist/> Dataset

module Torch.Datasets.Vision.MNIST where
import           Control.Monad            (liftM, when)
import           Control.Monad.Except
import           Data.Binary.Get          (Get, getByteString, getWord32be, runGet)
import qualified Data.ByteString          as BS
import qualified Data.ByteString.Lazy     as BSL (readFile)
import           Data.Singletons
import           Data.Singletons.Prelude  as SP
import           Data.Singletons.TypeLits
import           Data.Text                (Text)
import qualified Data.Text                as T
import           Data.Word                (Word32)
import           Pipes
import           Torch.Datasets.Common
import           Torch.Inplace
import           Torch.Misc
import           Torch.Tensor
import           Torch.Types

type MNIST dataPurpose = Dataset () dataPurpose Int (Tensor TFloat KCpu '[784]) (Tensor TLong KCpu '[1])

mnist :: Path -> IO (MNIST Train, MNIST Test)
mnist path =
  pure (remoteDatasetMNIST @'Train @60000 path "mnist"
         "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
         "a25bea736e30d166cdddb491f175f624"
         "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
         "6bbc9ace898e44ae57da46a324031adb"
       ,remoteDatasetMNIST @'Test  @10000 path "mnist"
         "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
         "27ae3e4e09519cfbb04c329615203637"
         "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
         "2646ac647ad5339dbf082846283269ea")

remoteDatasetMNIST :: forall (dataPurpose :: DataPurpose) (nrExamples :: Nat).
                     (SingI dataPurpose, SingI nrExamples, KnownNat nrExamples
                     ,(Narrow '[nrExamples] 0 0 1) ~ '[1]
                     ,(Narrow '[784*nrExamples] 0 0 784) ~ '[784])
                   => Text -> Text -> Text -> Text -> Text -> Text
                   -> MNIST dataPurpose
remoteDatasetMNIST root directory labelurl labelmd5 imageurl imagemd5 =
  Dataset
  { checkDataset = canFail checkAll
  , fetchDataset = canFail $ fetchAll False >> (access @dataPurpose @nrExamples)
  , forceFetchDataset = canFail $ fetchAll True >> (access @dataPurpose @nrExamples)
  , accessDataset = canFail $ access @dataPurpose @nrExamples
  , metadataDataset = pure (Right ())
  }
  where dirpath = root </> directory
        checkAll = do
          checkMD5 (dirpath </> takeBaseName labelurl) labelmd5
          checkMD5 (dirpath </> takeBaseName imageurl) imagemd5
        fetchAll force = do
          fetchOne force labelurl labelmd5
          fetchOne force imageurl imagemd5
        access :: forall dataPurpose (nrExamples :: Nat).
                 (SingI dataPurpose, SingI nrExamples, KnownNat nrExamples
                 ,(Narrow '[nrExamples] 0 0 1) ~ '[1]
                 ,(Narrow '[784*nrExamples] 0 0 784) ~ '[784])
               => CanFail (DataStream dataPurpose Int (Tensor TFloat KCpu '[784]) (Tensor TLong KCpu '[1]))
        access = do
          labs <- liftIO $ readMNISTLabels @nrExamples (dirpath </> takeBaseName labelurl)
          imgs <- liftIO $ readMNISTSamples @nrExamples (dirpath </> takeBaseName imageurl)
          s <- liftIO $ toScalar (1 / 255)
          imgs' <- liftIO $ toType imgs >>= \x -> (mul_ x s >> pure x)
          labs' <- liftIO $ toType labs
          pure $ mapM_ (\n ->
                   yield (DataSample (fromIntegral n)
                           (narrowFromToByLength (dimension_ @0) (size_ @784) imgs' (fromIntegral (n * 784)))
                           (narrowFromToByLength (dimension_ @0) (size_ @1) labs' (fromIntegral n))))
            [0..demoteN @nrExamples - 1]
        fetchOne force url md5 = do
          liftIO $ createDirectoryIfMissing' (root </> directory)
          checkMD5 (dirpath </> takeBaseName url) md5 `retryAfter`
            (do downloadUrl url (dirpath </> takeFileName url)
                extractGzip (dirpath </> takeFileName url))

-- | Check's the file's endianess, throwing an error if it's not as expected.
checkEndian :: Get ()
checkEndian = do
    magic <- getWord32be
    when (magic `notElem` ([2049, 2051] :: [Word32])) $
        fail "Expected big endian, but image file is little endian."

-- | Reads an MNIST file and returns a list of samples.
readMNISTSamples :: forall sz. (SingI sz, SingI (784 * sz))
                 => Text -> IO (Tensor 'TByte 'KCpu '[784 * sz])
readMNISTSamples path = do
    raw <- BSL.readFile (T.unpack path)
    fromByteStringNoCopy $ runGet getMNIST raw
  where
    getMNIST :: Get BS.ByteString
    getMNIST = do
        checkEndian
        cnt  <- liftM fromIntegral getWord32be
        rows <- liftM fromIntegral getWord32be
        cols <- liftM fromIntegral getWord32be
        getByteString $ fromIntegral $ cnt * rows * cols

-- | Reads a list of MNIST labels from a file and returns them.
readMNISTLabels :: forall sz. SingI sz
                  => Text -> IO (Tensor 'TByte 'KCpu '[sz])
readMNISTLabels path = do
    raw <- BSL.readFile (T.unpack path)
    fromByteStringNoCopy $ runGet getLabels raw
  where getLabels :: Get BS.ByteString
        getLabels = do
            checkEndian
            cnt <- liftM fromIntegral getWord32be
            getByteString cnt
