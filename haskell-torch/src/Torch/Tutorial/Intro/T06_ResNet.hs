{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, DeriveAnyClass, DeriveGeneric, FlexibleContexts, FlexibleInstances  #-}
{-# LANGUAGE FunctionalDependencies, GADTs, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, QuasiQuotes         #-}
{-# LANGUAGE RankNTypes, RecordWildCards, ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies #-}
{-# LANGUAGE TypeInType, TypeOperators, UndecidableInstances                                                                           #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=10 #-}
{-# OPTIONS_GHC -fplugin-opt GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Tutorial.Intro.T06_ResNet where
import           Control.Monad
import           Data.Default
import           Data.IORef
import           Data.String.InterpolateIO
import           GHC.Generics
import           Torch

imageTransformsTrain x = transformSampleObject
  (constantPad @4 @4 0 0 0
  >=> randomHorizontalFlip_ 0.5
  >=> randomCrop @32 @32
  >=> rgbImageToTensor
  >=> pure . typed @TFloat) x

imageTransformsTest x = transformSampleObject
  (rgbImageToTensor >=> pure . typed @TFloat) x

data SmallResNetBlock ki inChans outChans inSz outSz  =
  SmallResNetBlock { bw1  :: ConvParam 'TFloat ki outChans '[inChans, inSz, inSz],
                     bw2  :: ConvParam 'TFloat ki outChans '[outChans, outSz, outSz],
                     bbn1 :: BatchNormState 'TFloat ki '[outChans],
                     bbn2 :: BatchNormState 'TFloat ki '[outChans]
                   }
  deriving(Generic,ParameterNames,Stored,Initialize,ToTensors,ToParameters)

data SmallResNet ki =
  SmallResNet  { sw1   :: ConvParam 'TFloat ki 16 '[3, 3, 3],
                 sbn1  :: BatchNormState 'TFloat ki '[16],
                 sr1   :: SmallResNetBlock ki 16 16 3 3,
                 sr2   :: SmallResNetBlock ki 16 16 3 3,
                 sr3   :: SmallResNetBlock ki 16 32 3 3,
                 sr3w  :: ConvParam 'TFloat ki 32 '[16, 3, 3],
                 sr3bn :: BatchNormState 'TFloat ki '[32],
                 sr4   :: SmallResNetBlock ki 32 32 3 3,
                 sr5   :: SmallResNetBlock ki 32 64 3 3,
                 sr5w  :: ConvParam 'TFloat ki 64 '[32, 3, 3],
                 sr5bn :: BatchNormState 'TFloat ki '[64],
                 sr6   :: SmallResNetBlock ki 64 64 3 3,
                 swl   :: LinearParam 'TFloat ki 64 10
               }
  deriving(Generic,ParameterNames,Stored,Initialize,ToTensors,ToParameters)

residualBlock :: forall inChannels outChannels stride inSz outSz.
                _ =>
                  SmallResNetBlock KCpu inChannels outChannels 3 3
                -> DataPurpose -> (Tensor TFloat KCpu '[100, inChannels, inSz, inSz] -> IO (Tensor TFloat KCpu '[100, outChannels, outSz, outSz]))
                -> Tensor TFloat KCpu '[100, inChannels, inSz, inSz] -> IO (Tensor TFloat KCpu '[100, outChannels, outSz, outSz])
residualBlock SmallResNetBlock{..} isTraining downsample input =
    (conv2d (inChannels_ @inChannels) (outChannels_ @outChannels) (kernel_ @'(3,3))
         (stride_ @'(stride,stride)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) bw1
  >=> batchNorm2d_ bbn1 Nothing def def isTraining
  >=> relu_
  >=> conv2d (inChannels_ @outChannels) (outChannels_ @outChannels) (kernel_ @'(3,3))
            (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) bw2
  >=> batchNorm2d_ bbn2 Nothing def def isTraining
  >=> (\val -> do
         residual <- downsample input
         val .+= residual)
  >=> relu_) input

forward :: SmallResNet KCpu -> DataPurpose -> Tensor 'TFloat 'KCpu '[100, 3, 32, 32] -> IO (Tensor 'TFloat 'KCpu '[100, 10])
forward SmallResNet{..} isTraining =
      conv2d InChannels (outChannels_ @16) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) sw1
  >=> batchNorm2d_ sbn1 Nothing def def isTraining
  >=> relu_
  >=> residualBlock @16 @16 @1 @32 @32 sr1 isTraining pure
  >=> residualBlock @16 @16 @1 @32 @32 sr2 isTraining pure
  >=> residualBlock @16 @32 @2 @32 @16 sr3 isTraining
           (\x -> do
               r <- conv2d (inChannels_ @16) (outChannels_ @32) (kernel_ @'(3,3))
                   (stride_ @'(2,2)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) sr3w x
               batchNorm2d_ sr3bn Nothing def def isTraining r)
  >=> residualBlock @32 @32 @1 @16 @16 sr4 isTraining pure
  >=> residualBlock @32 @64 @2 @16 @8 sr5 isTraining
           (\x -> do
               r <- conv2d InChannels (outChannels_ @64) (kernel_ @'(3,3))
                   (stride_ @'(2,2)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) sr5w x
               batchNorm2d_ sr5bn Nothing def def isTraining r)
  >=> residualBlock @64 @64 @1 @8 @8 sr6 isTraining pure
  >=> avgPool2d (kernel_ @'(8,8)) (stride_ @'(1,1)) (padding_ @'(0,0)) (ceilMode_ @False) True
  >=> view @'[100, 64]
  >=> linear (inFeatures_ @64) (outFeatures_ @10) swl

ex = do
  let epochs = 10
  -- Datasets get downloaded and then streamed using Pipes
  (train,test) <- cifar10 "datasets/image/"
  -- Unpack the training set
  (Right trainStream') <- fetchDataset train
  (Right testStream') <- fetchDataset train
  let trainStream = batchTensors (batchSize_ @100)
                  $ shuffle 1000
                  $ transformStream imageTransformsTrain trainStream'
  let testStream  = batchTensors (batchSize_ @100)
                  $ transformStream imageTransformsTest trainStream'
  let criterion y ypred = crossEntropyLoss y def def def ypred
  net <- gradP @(SmallResNet KCpu)
  params <- toParameters net
  let initialLearningRate = 0.001
  optimizer <- newIORef (adam' (def { adamLearningRate = initialLearningRate }) params)
  sw <- summaryWriter "/tmp/qlog" "tester"
  -- I don't suggest updating the learning rate for Adam using this
  -- schedule. But it's what the original tutorials did, so we want to show you
  -- that this is possible.
  currentLR <- newIORef initialLearningRate
  -- Training
  withGrad
    $ mapM_ (\(epoch::Int) -> do
            forEachDataN
              (\d n -> do
                  zeroGradients_ params
                  loss <- do
                    images <- toDevice =<< dataObject d
                    labels <- toDevice =<< dataLabel d
                    pred   <- forward net (dataPurpose d) images
                    criterion labels pred
                  backward1 loss False False
                  step_ optimizer
                  addScalar sw "loss" =<< fromScalar loss
                  when (n `rem` 100  == 0) $ do
                    putStrLn =<< [c|Epoch #{epoch+1}/#{epochs} loss #{loss}|]
                    addImageRGBGrid sw "imageBatch" (size_ @8) (padding_ @4) 0 =<< dataObject d
                    addScalar sw "average lr" =<< getLearningRate optimizer
                  nextStep sw
                  pure ()) trainStream
            -- Decay the learning rate
            when (epoch+1 `mod` 20 == 0) $ do
              modifyIORef' currentLR (/3)
              lr <- readIORef currentLR
              setLearningRate_ optimizer lr
              pure ()
            )
    [0..epochs-1]
  pure ()
  -- Testing
  y <- withoutGrad
    $ foldData
      (\(nr, correct) d -> do
          images <- toDevice =<< dataObject d
          labels <- toDevice =<< dataLabel d
          pred <- forward net (dataPurpose d) images
          (_, predictionIndices) <- Torch.maxDim @1 pred
          nrCorrect <- fromIntegral <$> (fromScalar =<< Torch.sum =<< toType @TInt =<< (predictionIndices .== labels))
          pure (nr + 100, correct + nrCorrect))
      (0, 0)
      testStream
  putStrLn =<< [c|Test set accuracy #{((100 * (fromIntegral (snd y) / fromIntegral (fst y))) :: Double)}% on #{((fromIntegral (fst y))::Double)} images|]
  -- Save our model
  writeModelToFile net "resnet.ht"
  pure ()
