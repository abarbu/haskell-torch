{-# LANGUAGE AllowAmbiguousTypes, DataKinds, DeriveAnyClass, DeriveGeneric, ExtendedDefaultRules, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings, PolyKinds, QuasiQuotes, RecordWildCards, ScopedTypeVariables, TemplateHaskell, TypeApplications       #-}
{-# LANGUAGE TypeFamilies, TypeOperators                                                                                              #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver -fplugin Plugin.SimplifyNat -fconstraint-solver-iterations=10000 #-}

module Torch.Tutorial.Intro.T05_CNN where
import           Control.Monad
import           Data.Default
import           Data.IORef
import           Data.String.InterpolateIO
import           GHC.Generics
import           GHC.TypeNats
import           Torch
import           Torch.Datasets

data SmallCNN = SmallCNN {
    w1  :: ConvParam   TFloat KCpu 16 '[1, 5, 5]
  , w2  :: AffineParam TFloat KCpu '[16]
  , w3  :: ConvParam   TFloat KCpu 32 '[16, 5, 5]
  , w4  :: AffineParam TFloat KCpu '[32]
  , w5  :: LinearParam TFloat KCpu 1568 10
  , bn1 :: BatchNormState 'TFloat 'KCpu '[16]
  , bn2 :: BatchNormState 'TFloat 'KCpu '[32]
  }
  deriving(Generic,ParameterNames,Stored,Initialize,ToTensors,ToParameters)

forward :: SmallCNN -> DataPurpose -> Tensor 'TFloat 'KCpu '[100, 1, 28, 28] -> IO (Tensor 'TFloat 'KCpu '[100, 10])
forward SmallCNN{..} isTraining =
            conv2d InChannels (outChannels_ @16) (kernel_ @'(5,5)) (stride_ @'(1,1)) (padding_ @'(2,2)) (dilation_ @'(1,1)) (groups_ @1) w1
        >=> batchNorm2d_ bn1 (Just w2) def def isTraining
        >=> relu
        >=> maxPool2d (kernel_ @'(2,2)) (stride_ @'(2,2)) (padding_ @'(0,0)) (dilation_ @'(1,1)) (ceilMode_ @False)
        >=> pure . fst
        >=> conv2d InChannels (outChannels_ @32) (kernel_ @'(5,5)) (stride_ @'(1,1)) (padding_ @'(2,2)) (dilation_ @'(1,1)) (groups_ @1) w3
        >=> batchNorm2d_ bn2 (Just w4) def def isTraining
        >=> relu
        >=> maxPool2d (kernel_ @'(2,2)) (stride_ @'(2,2)) (padding_ @'(0,0)) (dilation_ @'(1,1)) (ceilMode_ @False)
        >=> pure . fst
        >=> view @'[100, 1568]
        >=> linear (inFeatures_ @1568) (outFeatures_ @10) w5

ex = do
  let epochs = 2
  let learningRate = 0.001
  --
  (tr, te) <- mnist "datasets/image/"
  (Right tes) <- fetchDataset te
  let testStream = batchTensors (batchSize_ @100) tes
  (Right trs) <- fetchDataset tr
  let trainStream = batchTensors (batchSize_ @100)
                  $ shuffle 1000 trs
  --
  net <- gradP @SmallCNN
  params <- toParameters net
  optimizer <- newIORef (adam (def { adamLearningRate = learningRate }) params)
  --
  let criterion y ypred = crossEntropyLoss y def def def ypred
  --
  withGrad
    $ mapM_ (\epoch ->
            forEachDataN
              (\d n -> do
                  loss <- do
                    o <- view =<< toCpu =<< dataObject d
                    l <- view =<< toCpu =<< dataLabel d
                    pred <- (forward net (dataPurpose d) . sized (size_ @'[100, 1, 28, 28])) =<< view (sized (size_ @'[100,784]) o)
                    criterion (sized (size_ @'[100]) l) pred
                  zeroGradients_ params
                  backward1 loss False False
                  step_ optimizer
                  when (n `rem` 100  == 0) $ putStrLn =<< [c|Epoch #{epoch+1}/#{epochs} loss #{loss}|]
                  putStrLn =<< [c|Step #{epoch+1}/#{epochs} #{n+1} loss #{loss}|]
                  pure ())
              trainStream)
    [0..epochs-1]
  --
  y <- withoutGrad
    $ foldData
      (\(nr, correct) d -> do
          o <- copy @TFloat @KCpu =<< dataObject d
          l <- sized (size_ @'[100]) <$> (view =<< toCpu =<< dataLabel d)
          pred <- (forward net (dataPurpose d) . sized (size_ @'[100, 1, 28, 28])) =<< view (sized (size_ @'[100,784]) o)
          (_, is) <- Torch.maxDim @1 pred
          s <- fromIntegral <$> (fromScalar =<< Torch.sum =<< toType @TInt =<< (is .== l))
          pure (nr + 100, correct + s))
      (0, 0)
      testStream
  putStrLn =<< [c|Test set accuracy #{100 * (fromIntegral (snd y) / fromIntegral (fst y))}% on #{fromIntegral (fst y)} images|]
  pure ()
