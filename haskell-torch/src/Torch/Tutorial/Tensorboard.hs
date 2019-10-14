{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, EmptyCase, FlexibleContexts, FlexibleInstances              #-}
{-# LANGUAGE FunctionalDependencies, GADTs, KindSignatures, MultiParamTypeClasses, OverloadedLabels, OverloadedStrings         #-}
{-# LANGUAGE PartialTypeSignatures, PolyKinds, QuasiQuotes, RankNTypes, ScopedTypeVariables, TemplateHaskell, TypeApplications #-}
{-# LANGUAGE TypeFamilies, TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances                             #-}
{-# options_ghc -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Torch.Tutorial.Tensorboard where
import           Control.Monad
import           Control.Monad.Logger
import           Data.Default
import           Data.ProtoLens.Labels ()
import           Torch                 as T

ex = do
  sw <- summaryWriter "/tmp/qlog" "tester"
  (tr, te) <- mnist "datasets/image/"
  (Right test') <- fetchDataset te
  let test = transformObjectStream (T.view @'[1,28,28]
                                     >=> greyTensorToImage
                                     >=> randomHorizontalFlip_ 0.5
                                     >=> randomVerticalFlip_ 0.5
                                     >=> randomContrastJitter_ 100
                                     >=> randomBrightnessJitter_ 100
                                     >=> greyImageToTensor @28 @28 @TFloat
                                   ) test'
  writeEvent sw (EventMessage LevelInfo "Woof")
  t0 <- randn @TFloat @KCpu @'[100]
  x <- typed @TFloat <$> stored @KCpu <$> sized (size_ @'[10,3]) <$> randn
  y <- sized (size_ @'[10,2]) <$> randn
  w1 <- noGradP
  w2 <- noGradP
  let model = linear (inFeatures_ @3) (outFeatures_ @10) w1
            >=> relu
            >=> linear (inFeatures_ @10) (outFeatures_ @2) w2
  let criterion = mseLoss y def
  (loss, trace) <- withTracing [AnyTensor x, AnyTensor y] $ do
    pred <- model x
    criterion pred
  printTrace trace
  rawTrace trace
  addGraph sw "grph" trace
  forEachDataUntil
    (\step _ -> pure (step == 10))
    (\_ _ stream -> pure stream)
    (\step epoch value -> do
        addScalar sw "bloop/blimp" 3.0
        addHistogram sw "bloop/h1" t0
        t <- dataObject value ..*@ pure 255
        addImageGrey sw "bloop/mnist" t
        t0 .*=@ 1.2
        nextStep sw)
    test
  pure ()
