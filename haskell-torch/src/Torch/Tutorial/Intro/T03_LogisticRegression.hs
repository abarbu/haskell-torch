{-# LANGUAGE AllowAmbiguousTypes, DataKinds, ExtendedDefaultRules, FlexibleContexts, FlexibleInstances, OverloadedStrings, PolyKinds #-}
{-# LANGUAGE QuasiQuotes, ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies                                       #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
module Torch.Tutorial.Intro.T03_LogisticRegression where
import           Control.Monad
import           Data.Default
import           Data.IORef
import           Data.String.InterpolateIO
import           Torch

ex = do
  let epochs = 5
  let learningRate = 0.001
  (tr, te) <- mnist "datasets/image/"
  (Right tes) <- fetchDataset te
  let testStream = batchTensors (batchSize_ @100) tes
  (Right trs) <- fetchDataset tr
  let trainStream = batchTensors (batchSize_ @100) (shuffle 1000 trs)
  w <- gradP
  let model = linear (inFeatures_ @784) (outFeatures_ @10) w
  let criterion y ypred = crossEntropyLoss y def def def ypred
  params <- toParameters w
  --
  optimizer <- newIORef (sgd (def { sgdLearningRate = learningRate }) params)
  withGrad
    $ mapM_ (\epoch -> do
            forEachDataN
              (\d n -> do
                  zeroGradients_ params
                  loss <- do
                    o <- view =<< dataObject d
                    l <- view =<< dataLabel d
                    pred <- model (sized (size_ @'[100,784]) o)
                    criterion (sized (size_ @'[100]) l) pred
                  backward1 loss False False
                  step_ optimizer
                  when (n `rem` 100  == 0) $ putStrLn =<< [c|Epoch #{epoch+1}/#{epochs} loss #{loss}|]
                  pure ())
              trainStream)
    [0..epochs-1]
  x <- withoutGrad
    $ foldData
      (\(nr, correct :: Int) d -> do
          o <- copy @TFloat @KCpu =<< dataObject d
          l <- sized (size_ @'[100]) <$> (view =<< dataLabel d)
          pred <- model (sized (size_ @'[100,784]) o)
          (_, is) <- Torch.maxDim @1 pred
          s <- fromIntegral <$> (fromScalar =<< Torch.sum =<< toType @TInt =<< (is .== l))
          pure $ (nr + 100, correct + s))
      (0, 0)
      trainStream
  putStrLn =<< [c|Training set accuracy #{100 * (fromIntegral (snd x) / fromIntegral (fst x))}% #{fromIntegral (fst x)} images|]
  y <- withoutGrad
    $ foldData
      (\(nr, correct) d -> do
          o <- copy @TFloat @KCpu =<< dataObject d
          l <- sized (size_ @'[100]) <$> (view =<< dataLabel d)
          pred <- model (sized (size_ @'[100,784]) o)
          (_, is) <- Torch.maxDim @1 pred
          s <- fromIntegral <$> (fromScalar =<< Torch.sum =<< toType @TInt =<< (is .== l))
          pure (nr + 100, correct + s))
      (0, 0)
      testStream
  putStrLn =<< [c|Test set accuracy #{100 * (fromIntegral (snd y) / fromIntegral (fst y))}% on #{fromIntegral (fst y)} images|]
  pure ()
