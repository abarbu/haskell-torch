{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, DeriveAnyClass, DeriveGeneric, FlexibleContexts, FlexibleInstances  #-}
{-# LANGUAGE FunctionalDependencies, GADTs, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, QuasiQuotes         #-}
{-# LANGUAGE RankNTypes, RecordWildCards, ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies #-}
{-# LANGUAGE TypeInType, TypeOperators, UndecidableInstances                                                                           #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=10 -fdefer-typed-holes #-}
{-# OPTIONS_GHC -fplugin-opt GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Tutorial.Intro.T08_BiRNN where
import           Control.Monad
import           Data.Default
import           Data.IORef
import           Data.String.InterpolateIO
import           GHC.Generics
import qualified GHC.TypeLits              as TL
import           Torch
import           Torch.Datasets

data Model = Model { w1 :: LSTMParams TFloat KBest 28 128 2 'True 'True,
                     w2 :: LinearParam TFloat KBest (2 TL.* 128) 10 }
  deriving(Generic,ParameterNames,Stored,Initialize,ToTensors,ToParameters)

forward :: _ -> DataPurpose -> Tensor TFloat KBest '[100, 28, 28] -> IO (Tensor TFloat KBest '[100, 10])
forward (Model{..},state1) isTraining = do
           lstmBatchFirst (inF_ @28) (hiddenF_ @128) (nrLayers_ @2) (isBidirectional_ @True) 0 isTraining w1 state1
        >=> (\(a,s) -> select @1 a (-1))
        >=> linear (inF_ @(2 TL.* 128)) (outFeatures_ @10) w2

ex = do
  let epochs = 2 :: Int
  let learningRate = 0.001
  --
  (tr, te) <- mnist "datasets/image/"
  (Right tes) <- fetchDataset te
  let testStream = batchTensors (batchSize_ @100) tes
  (Right trs) <- fetchDataset tr
  let trainStream = batchTensors (batchSize_ @100)
                  $ shuffle 1000 trs
  --
  net <- gradP
  params <- toParameters net
  optimizer <- newIORef (adam (def { adamLearningRate = learningRate }) params)
  --
  let criterion y ypred = crossEntropyLoss y def def def ypred
  --
  withGrad
    $ mapM_ (\epoch ->
            forEachDataN
              (\d n -> do
                  initialState <- gradP
                  zeroGradients_ params
                  loss <- do
                    o <- view =<< toDevice =<< dataObject d
                    l <- view =<< toDevice =<< dataLabel d
                    pred <- (forward (net,initialState) (dataPurpose d) . sized (size_ @'[100, 28, 28])) =<< view (sized (size_ @'[100,784]) o)
                    criterion (sized (size_ @'[100]) l) pred
                  backward1 loss False False
                  step_ optimizer
                  when (n `rem` 100  == 0) $ putStrLn =<< [c|Epoch #{epoch+1}/#{epochs} loss #{loss}|]
                  putStrLn =<< [c|Step1 #{epoch+1}/#{epochs} #{n+1} loss #{loss}|]
                  pure ())
              trainStream)
    [0..epochs-1]
  --
  y <- withoutGrad
    $ foldData
      (\(nr, correct) d -> do
          initialState <- gradP
          images <- view =<< toDevice =<< dataObject d
          labels <- view =<< toDevice =<< dataLabel d
          pred <- forward (net,initialState) (dataPurpose d) images
          (_, predictionIndices) <- Torch.maxDim @1 pred
          nrCorrect <- fromIntegral <$> (fromScalar =<< Torch.sum =<< toType @TInt =<< (predictionIndices .== sized (size_ @'[100]) labels))
          pure (nr + 100, correct + nrCorrect))
      (0, 0)
      testStream
  putStrLn =<< [c|Test set accuracy #{((100 * (fromIntegral (snd y) / fromIntegral (fst y))) :: Double)}% on #{((fromIntegral (fst y))::Double)} images|]
  --
  writeModelToFile net "lstm.ht"
  pure ()
