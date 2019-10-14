{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, DeriveAnyClass, DeriveGeneric, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies, GADTs, KindSignatures, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds     #-}
{-# LANGUAGE QuasiQuotes, RankNTypes, RecordWildCards, ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies           #-}
{-# LANGUAGE TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances                                                  #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=10 #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}
{-# OPTIONS_GHC -fplugin-opt GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Torch.Tutorial.Intro.T10_GAN where
import           Control.Monad
import           Control.Monad.Extra
import           Data.Default
import           Data.IORef
import           Data.String.InterpolateIO
import           System.Directory          as D
import           System.IO.Unsafe
import           Torch
import           Torch.Tensor              as T

-- This is an example where we intentionally left it in a more intermediate
-- stage for you. Note that discriminator and generator have wildcards for their
-- constraints and their weights. This makes Haskell-Torch very flexible, you
-- can easily change network layouts.

imageTransformsTrain x = transformSampleObject
  (  reshape @'[1,28,28]
  >=> normalizeGreyImageTensor (unsafePerformIO $ full 0.5)
                              (unsafePerformIO $ full 0.5)
  >=> reshape @'[784]
  >=> pure . typed @TFloat) x

type LatentSize = 64
type HiddenSize = 256
type ImageSize = 784
type BatchSz = 100

discriminator :: _ => _ -> Tensor ty ki '[batch, ImageSize] -> IO (Tensor ty ki '[batch, 1])
discriminator (w1,w2,w3) =
       linear (inF_ @ImageSize)  (outF_ @HiddenSize) w1
    >=> leakyRelu 0.2
    >=> linear (inF_ @HiddenSize) (outF_ @HiddenSize) w2
    >=> leakyRelu 0.2
    >=> linear (inF_ @HiddenSize) (outF_ @1) w3
    >=> sigmoid

generator :: _ => _ -> Tensor ty ki '[batch, LatentSize] -> IO (Tensor ty ki '[batch, ImageSize])
generator (w1,w2,w3) =
       linear (inF_ @LatentSize)  (outF_ @HiddenSize) w1
    >=> relu
    >=> linear (inF_ @HiddenSize) (outF_ @HiddenSize) w2
    >=> relu
    >=> linear (inF_ @HiddenSize) (outF_ @ImageSize) w3
    >=> T.tanh

denorm x =
  clamp 0 1 =<< ((x .+@ 1) ../@ pure 2)

ex = do
  let epochs = 200 :: Int
  --
  whenM (D.doesDirectoryExist "generated-samples") $
    D.removeDirectoryRecursive "generated-samples"
  D.createDirectory "generated-samples"
  dNet <- gradP
  dParams <- toParameters dNet
  dOptimizer <- newIORef (adam (def { adamLearningRate = 0.0002 }) dParams)
  gNet <- gradP
  gParams <- toParameters gNet
  gOptimizer <- newIORef (adam (def { adamLearningRate = 0.0002 }) gParams)
  let allParams = dParams ++ gParams
  --
  (tr, _) <- mnist "datasets/image/"
  (Right trs) <- fetchDataset tr
  let trainStream = batchTensors (batchSize_ @100)
                  $ shuffle 10000
                  $ transformStream imageTransformsTrain trs
  --
  let criterion y ypred = binaryCrossEntropyLoss y def def ypred
  --
  withGrad
    $ mapM_
    (\epoch ->
        forEachDataN
          (\d n -> do
              images <- reshape =<< dataObject d
              --
              realLabels <- noGrad =<< sized (size_ @'[BatchSz, 1]) <$> ones
              fakeLabels <- noGrad =<< sized (size_ @'[BatchSz, 1]) <$> zeros
              --
              -- In case you haven't seen GANs before, the logic here is:
              --  Train the discriminator with real images to predict they're real
              --    and the discriminator with fake images from the generator to predict they're fake
              --  Train the generator to produce images that make the discriminator think they're real
              --
              -- =============== train discriminator ===============
              -- Compute BCE_Loss using real images where
              --  BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
              -- Second term of the loss is always zero since real_labels == 1
              realScore <- discriminator dNet images
              dLossReal <- criterion realLabels realScore
              -- Compute BCELoss using fake images
              -- First term of the loss is always zero since fake_labels == 0
              z <- sized (size_ @'[BatchSz, LatentSize]) <$> randn
              fakeImages <- generator gNet z
              fakeScore <- discriminator dNet fakeImages
              dLossFake <- criterion fakeLabels fakeScore
              -- Backprop and optimize
              dLoss <- dLossReal .+ dLossFake
              zeroGradients_ allParams
              backward1 dLoss False False
              step_ dOptimizer
              --
              -- =============== train generator ===================
              -- Compute loss with fake images
              z <- sized (size_ @'[BatchSz, LatentSize]) <$> randn
              fakeImages <- generator gNet z
              -- We train G to maximize log(D(G(z))) instead of minimizing log(1-D(G(z)))
              -- See the last paragraph of section 3 https://arxiv.org/pdf/1406.2661.pdf
              gLoss <- criterion realLabels =<< discriminator dNet fakeImages
              --
              zeroGradients_ allParams
              backward1 gLoss False False
              step_ gOptimizer
              --
              -- The real MNIST images
              when (epoch == 0 && n == 0) $ do
                writeGreyTensorToFile "generated-samples/real-images.png"
                  =<< makeGreyGrid (size_ @8) (padding_ @0) 0
                  =<< denorm
                  =<< reshape @'[BatchSz, 1, 28, 28] images
                pure ()
              -- Images produced by the GAN
              when (n == 0) $ do
                writeGreyTensorToFile ("generated-samples/fake-images-"<>show' epoch<>"@"<>show' n<>".jpg")
                  =<< makeGreyGrid (size_ @8) (padding_ @2) 0
                  =<< denorm
                  =<< reshape @'[BatchSz, 1, 28, 28] fakeImages
                pure ()
              --
              when (n `rem` 100  == 0) $
                putStrLn =<< [c|Epoch #{epoch+1}/#{epochs} dLoss #{dLoss} gLoss #{gLoss} D(x):#{mean realScore} G(x):#{mean fakeScore}|]
              putStrLn =<< [c|Step #{n+1} #{epoch+1}/#{epochs} dLoss #{dLoss} gLoss #{gLoss} D(x):#{mean realScore} G(x):#{mean fakeScore}|]
              pure ())
            trainStream)
    [0..epochs-1]
  pure ()
