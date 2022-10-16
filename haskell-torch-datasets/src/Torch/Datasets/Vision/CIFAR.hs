{-# LANGUAGE AllowAmbiguousTypes, ConstraintKinds, DataKinds, FlexibleContexts, FlexibleInstances, GADTs, KindSignatures              #-}
{-# LANGUAGE OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, RankNTypes, ScopedTypeVariables, TypeApplications #-}
{-# LANGUAGE TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances                                                  #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fdefer-type-errors #-}

-- CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html> Dataset

module Torch.Datasets.Vision.CIFAR where
import           Control.Monad.Except
import           Data.Map.Strict       (Map)
import qualified Data.Map.Strict       as M
import           Data.Singletons
import           Data.Text             (Text)
import qualified Data.Text             as T
import qualified Data.Text.IO          as T
import qualified Data.Text.Read        as T
import qualified Foreign.ImageMagick   as I
import           Pipes
import           Torch.Datasets.Common
import           Torch.Images
import           Torch.Misc
import           Torch.Tensor
import           Torch.Types

type CIFAR dataPurpose = Dataset (Map Int Text) dataPurpose Int (Image 3 32 32) (Tensor TLong KCpu '[])

cifar10 :: Path -> IO (CIFAR Train ,CIFAR Test)
cifar10 path =
  pure (remoteDatasetCIFAR @'Train path filename directory url md5
       ,remoteDatasetCIFAR @'Test  path filename directory url md5)
  where url = "https://pjreddie.com/media/files/cifar.tgz"
        filename = "cifar.tgz"
        md5 = "a00ceaeb02303e3ff0d0011b38b465fa"
        directory = "cifar"

remoteDatasetCIFAR :: forall (dataPurpose :: DataPurpose). (SingI dataPurpose)
                   => Text -> Text -> Text -> Text -> Text
                   -> CIFAR dataPurpose
remoteDatasetCIFAR root filename directory url md5 =
  Dataset
  { checkDataset = canFail checkAll
  , fetchDataset = canFail $ fetchAll False >> access @dataPurpose
  , forceFetchDataset = canFail $ fetchAll True >> access @dataPurpose
  , accessDataset = canFail (access @dataPurpose)
  , metadataDataset = canFail (fst <$> metadata)
  }
  where path = root </> filename
        dirpath x = root </> directory </> x
        checkAll = checkMD5 path md5
        fetchAll force = do
          liftIO $ createDirectoryIfMissing' root
          checkMD5 path md5 `retryAfter` downloadUrl url path
          e <- liftIO $ doesDirectoryExist (root </> directory)
          when (force || not e) $ extractTar path root
        metadata = do
          c <- liftIO $ T.readFile (T.unpack $ dirpath "labels.txt")
          pure (M.fromList $ zipWith (\l n -> (n,l)) (T.lines c) [0..]
               ,M.fromList $ zipWith (\l n -> (l,n)) (T.lines c) [0..])
        pipeCifar :: forall dataPurpose. (SingI dataPurpose)
                  => M.Map Text Int
                  -> Pipe Text (DataSample dataPurpose Int (Image 3 32 32) (Tensor TLong KCpu '[])) IO ()
        pipeCifar labtonr = forever $ do
          fname <- await
          let [nrtext, label] = T.splitOn "_" $ dropExtension $ takeBaseName $ fname
          let (Right (nr, "")) = T.decimal nrtext
          case M.lookup label labtonr of
            Nothing -> error $ "CIFAR: Don't know what the label of this image should be! " ++ show fname
            Just labelnr ->
              yield $ DataSample @dataPurpose nr (readImageFromFile fname)
                                                 (toScalar $ fromIntegral labelnr)
        access :: forall dataPurpose. (SingI dataPurpose)
               => CanFail (DataStream dataPurpose Int (Image 3 32 32) (Tensor TLong KCpu '[]))
        access = do
          liftIO $ I.initialize
          (_,labtonr) <- metadata
          case demote @dataPurpose of
            Test ->
              pure $ pipeListDirectory (dirpath "test")
                 >-> pipeCifar @dataPurpose labtonr
            Train ->
              pure $ pipeListDirectory (dirpath "train")
                 >-> pipeCifar @dataPurpose labtonr
