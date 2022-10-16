{-# LANGUAGE AllowAmbiguousTypes, ConstraintKinds, DataKinds, ExtendedDefaultRules, FlexibleContexts, FlexibleInstances, GADTs #-}
{-# LANGUAGE OverloadedLists, OverloadedStrings, PolyKinds, QuasiQuotes, RankNTypes, ScopedTypeVariables, TemplateHaskell      #-}
{-# LANGUAGE TypeApplications, TypeFamilies, TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances           #-}
module Torch.Tutorial.Intro.T02_LinearRegression where
import           Data.Default
import           Data.String.InterpolateIO
import           Graphics.Matplotlib       (o1, o2, (%), (@@))
import qualified Graphics.Matplotlib       as M
import           Pipes
import qualified Pipes.Prelude             as P
import           Torch
import           Torch.Datasets

ex = do
  -- hyperparameters
  let epochs = 60
  let learningRate = 0.001
  --
  let train =
        ((yield $
        let x = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168
                ,9.779, 6.182, 7.59, 2.167, 7.042
                ,10.791, 5.313, 7.997, 3.1]
            y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573
                ,3.366, 2.596, 2.53, 1.221, 2.827
                ,3.465, 1.65, 2.904, 1.3]
        in DataSample @Train () (fromVector x) (fromVector y))
         :: Producer (DataSample 'Train ()
                     (Tensor TDouble KCpu '[15,1])
                     (Tensor TDouble KCpu '[15,1]))
             IO ())
  w <- gradP
  let model = linear (inFeatures_ @1) (outFeatures_ @1) w
  let criterion y = mseLoss y def
  params <- toParameters w
  mapM_ (\epoch -> do
            zeroGradients_ params
            loss <- lossForEachData (\d -> do
                                       o <- dataObject d
                                       l <- dataLabel d
                                       pred <- model o
                                       criterion l pred) train
            backward1 loss False False
            _ <- step_ (sgd (def { sgdLearningRate = learningRate }) params)
            putStrLn =<< [c|Epoch #{epoch}/#{epochs} loss #{loss}|])
    [0..epochs-1]
  (Just e) <- P.head train
  xs <- dataObject e >>= toVector
  ys <- dataLabel e >>= toVector
  ysPred <- dataObject e >>= model >>= toVector
  M.onscreen $ M.plot xs ysPred @@ [o1 "go-", o2 "linewidth" 2]
             % M.plot xs ys     @@ [o1 "ro"]
  pure ()
