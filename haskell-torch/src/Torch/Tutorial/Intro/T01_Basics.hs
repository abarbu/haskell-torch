{-# LANGUAGE AllowAmbiguousTypes, ConstraintKinds, ExtendedDefaultRules, FlexibleContexts, FlexibleInstances, GADTs, OverloadedStrings #-}
{-# LANGUAGE PolyKinds, QuasiQuotes, RankNTypes, ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies                  #-}
{-# LANGUAGE TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances                                                   #-}

-- | This example shows you Haskell & PyTorch code right next to one
-- another. You get an idea of how the two are related and how to do so some of
-- he most basic operations.

module Torch.Tutorial.Intro.T01_Basics where
import           Control.Monad
import           Data.Default
import           Data.Kind
import           Data.Maybe
import           Data.Singletons
import           Data.String.InterpolateIO
import qualified Data.Vector                 as V'
import           Data.Vector.Storable        (Vector)
import qualified Data.Vector.Storable        as V
import           Foreign.C.Types
import           Pipes
import qualified Pipes.Prelude               as P
import           Torch
import qualified Torch.C.Variable            as C
import           Torch.Datasets.Vision.CIFAR

-- | Basic autograd
ex1 = do
  setSeed 0
  s <- stored @KCpu <$> toScalar (float 1)
  -- Create tensors.
  x <- stored @KCpu <$> (needGrad =<< toScalar (float 1))
  w <- needGrad =<< toScalar (float 2)
  b <- needGrad =<< toScalar (float 3)
  putStrLn =<< [c|X: #{x}
W: #{w}
B: #{b}|]
  -- Compute primal
  y <- pure w ..* pure x ..+ pure b
  -- Compute gradient
  backward1 y False False
  putStrLn =<< [c|dX: #{gradient x} expected 2
dW: #{gradient w} expected 1
dB: #{gradient b} expected 1|]
  debuggingPrintADGraph y

-- | Basic autograd with SGD
ex2 = do
  unsafeEnableGrad
  setSeed 0
  -- Create tensors of shape (10, 3) and (10, 2).
  x <- typed @TFloat <$> stored @KCpu <$> sized (size_ @'[10,3]) <$> randn
  y <- sized (size_ @'[10,2]) <$> randn
  -- Weights and biases for a linear layer.
  w <- gradP
  --
  let model = linear (inFeatures_ @3) (outFeatures_ @2) w
  pred <- model x
  let criterion = mseLoss y def
  loss <- criterion pred
  --
  backward1 loss False False
  putStrLn =<< [c|weights & biases:\n#{w}
Loss: #{loss}|]
  -- 1 step of gradient descent
  params <- toParameters w
  step_ (sgd (def { sgdLearningRate = 0.01 }) params)
  --
  pred <- model x
  loss <- criterion pred
  putStrLn =<< [c|Loss after 1 SGD step #{loss}|]
  pure ()

-- | Loading data from a Storable Vector
ex3 = do
  let v = V.fromList [1,2,3,4]
  -- The resulting tensor is always on the CPU, does not have gradients enabled
  -- and is marked as a leaf. Its type depends on the type of the Vector.  Only
  -- Vectors with Foreign.C types are allowed (so CDouble instead of Double,
  -- etc.).
  --
  -- For the result to be useful you must somehow constrain the types.  Here we
  -- do so locally using type application but if some downstream consumer of t
  -- constrained its shape we would not need to do so.
  --
  -- This is one of the few interfaces between runtime values and types. It will
  -- error out if the size of the vector is not exactly equal to size of the
  -- tensor.
  t <- fromVector @'TDouble @'[4] v
  -- Alternatively we can use the functions found under the Constraints heading
  -- in Torch.Tensor. These have no runtime component, they just allow you to
  -- constrain the types of tensors easily.
  t' <- typed @'TDouble <$> sized (size_ @'[4]) <$> fromVector v
  -- Or we can say that the new tensor should inherit its properties, aside from
  -- AD status like if gradients are required, from another tensor.
  t'' <- like t <$> fromVector v
  -- A few other ways to create vectors exist, see the "Tensor creation" section
  -- in Tensor.Torch, for example we can make the vector of all 1s that's just
  -- like t.
  t''' <- like t <$> ones
  -- We can also convert tenstors back into vectors.
  v' <- toVector t
  writeModelToFile t "/tmp/woof"
  tl <- like t <$> readModelFromFile "/tmp/woof"
  out t
  out tl
  out =<< t .== tl
  print v'
  print $ v == v'

-- | Input datasets
ex4 = do
  -- Datasets get downloaded and then streamed using Pipes
  (train,test) <- cifar10 "datasets/image/"
  -- Unpack the training set
  (Right trainStream) <- liftM (transformObjectStream rgbImageToTensor) <$> fetchDataset train
  -- Data is loaded on demand, here we read the first data point
  (Just d) <- P.head trainStream
  image <- typed @TByte <$> dataObject d
  label <- dataLabel d
  print $ dataProperties d
  print $ size image
  out label
  -- All datasets can define any custom metadata that they want. CIFAR gives you
  -- a map between indices and text labels so you can interpret the classes.
  metadata <- metadataDataset train
  print metadata
  -- Lets iterate one by one over the first 10 data points shuffling with a
  -- horizon of 1000
  forEachData
    (\d -> do
        print "One data point at a time"
        -- Training code goes here
        putStrLn =<< [c|n: #{dataProperties d} lab: #{dataLabel d}|])
    (shuffle 1000 trainStream >-> P.take 10)
  -- Same if we batch by 64. True means give us a partial batch at the end if our
  -- data isn't divisble by 64.
  forEachData
    (\ds -> do
        print "Batches of 64"
        print $ V'.length ds
        mapM_ (\d ->
                          -- Training code goes here
                          putStrLn =<< [c|n: #{dataProperties d} lab: #{dataLabel d}|]) ds)
     (batch 64 True (shuffle 1000 trainStream) >-> P.take 3)
  pure ()

-- ex5 does not exist. We don't need anything like custom loaders, you just
-- create pipes. Look at how the datasets are constructed in Torch.Datasets

-- ex6 does not exist. Have a look at Torch.Models.Vision.AlexNet how to load
-- pretrained models.

-- ex7 does not exist. TODO We do not yet have integrated checkpointing
-- support. You can save and load a model but we cannot yet do this with
-- optimizers and cannot do it all for you in one go.

-- Viewing the trace of a computation
-------------------------------------------------------------------------------

ex8 = do
  unsafeEnableGrad
  setSeed 0
  -- Create tensors of shape (10, 3) and (10, 2).
  x <- typed @TFloat <$> stored @KCpu <$> sized (size_ @'[7,5]) <$> randn
  y <- sized (size_ @'[7,2]) <$> randn
  -- Weights and biases for a fully connected layer
  w <- noGradP
  let model = linear (inFeatures_ @5) (outFeatures_ @2) w
  let criterion = mseLoss y def
  params <- toParameters w
  (loss, trace) <- withTracing [AnyTensor x, AnyTensor y] $ do
    pred <- model x
    criterion pred
  putStrLn =<< [c|weights & biases:\n#{w}
Loss: #{loss}|]
  printTrace trace
  printTraceONNX trace [AnyTensor x, AnyTensor y] False 11
  trace' <- parseTrace trace
  summarizeTrace trace'
  showTraceGraph trace False
  -- 1 step of gradient descent
  p <- toParameters w
  step_ (sgd (def { sgdLearningRate = 0.01 }) p)
  --
  pred <- model x
  loss <- criterion pred
  putStrLn =<< [c|Loss after 1 SGD step #{loss}|]
  pure ()
