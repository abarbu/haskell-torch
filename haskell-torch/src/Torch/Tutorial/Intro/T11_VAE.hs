{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, DeriveAnyClass, DeriveGeneric, FlexibleContexts, FlexibleInstances  #-}
{-# LANGUAGE FunctionalDependencies, GADTs, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, QuasiQuotes         #-}
{-# LANGUAGE RankNTypes, RecordWildCards, ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies #-}
{-# LANGUAGE TypeInType, TypeOperators, UndecidableInstances                                                                           #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=10 -fdefer-typed-holes #-}
{-# OPTIONS_GHC -fplugin-opt GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Tutorial.Intro.T11_VAE where
import           Control.Monad
import           Control.Monad.Extra
import           Data.Default
import           Data.IORef
import           Data.String.InterpolateIO
import           System.Directory          as D
import           Torch
import           Torch.Tensor              as T

type ImageSize = 784
type HDim = 400
type ZDim = 20
type BatchSz = 128

encode :: _ => _
       -> Tensor TFloat ki '[batch, ImageSize]
       -> IO (Tensor TFloat ki '[batch, ZDim], Tensor TFloat ki '[batch, ZDim])
encode (w1,w2,w3,_,_) x = do
  h <- linear (inF_ @ImageSize) (outF_ @HDim) w1 x
  a <- linear (inF_ @HDim) (outF_ @ZDim) w2 h
  b <- linear (inF_ @HDim) (outF_ @ZDim) w3 h
  pure (a,b)

decode :: _
       => _
       -> Tensor TFloat ki '[batch, ZDim]
       -> IO (Tensor TFloat ki '[batch, ImageSize])
decode (_,_,_,w4,w5) x = do
  h <- relu =<< linear (inF_ @ZDim) (outF_ @HDim) w4 x
  sigmoid =<< linear (inF_ @HDim) (outF_ @ImageSize) w5 h

reparameterize :: _
               => Tensor TFloat ki '[batch, ZDim]
               -> Tensor TFloat ki '[batch, ZDim]
               -> IO (Tensor TFloat ki '[batch, ZDim])
reparameterize mu logVar = do
  std <- T.exp =<< logVar ./@ 2
  eps <- randn
  pure mu ..+ (like std eps .* std)

forward ws x = do
  (mu, logVar) <- encode ws x
  z <- reparameterize mu logVar
  x' <- decode ws z
  pure (x', mu, logVar)

ex = do
  let epochs = 200 :: Int
  --
  whenM (D.doesDirectoryExist "generated-images") $
    D.removeDirectoryRecursive "generated-images"
  D.createDirectory "generated-images"
  net <- gradP
  params <- toParameters net
  optimizer <- newIORef (adam (def { adamLearningRate = 1e-3 }) params)
  --
  (tr, _) <- mnist "datasets/image/"
  (Right trs) <- fetchDataset tr
  let trainStream = batchTensors (batchSize_ @BatchSz)
                  $ shuffle 5000 trs
  --
  withGrad
    $ mapM_
    (\epoch ->
        forEachDataN
          (\d n -> do
              images <- reshape =<< dataObject d
              (images', mu, logVar) <- forward net images
              -- Compute reconstruction loss and kl divergence
              -- For KL divergence, see Appendix B in VAE paper
              --   or http://yunjey47.tistory.com/43
              reconstructionLoss <- binaryCrossEntropyLoss images def (SizeAverage False) images'
              -- TODO This could be a lot cleaner
              klDivergence <-
                ((-0.5) @*. ) =<< T.sum =<< (1 @+. logVar ..- (T.pow mu =<< toScalar 2) ..- T.exp logVar)
              -- Backprop and optimize
              loss <- reconstructionLoss .+ klDivergence
              zeroGradients_ params
              backward1 loss False False
              step_ optimizer
              when (n `rem` 100  == 0) $
                putStrLn =<< [c|Epoch #{epoch+1}/#{epochs} Reconstruction loss #{reconstructionLoss} KL divergence #{klDivergence}|]
              putStrLn =<< [c|Step #{n+1} #{epoch+1}/#{epochs} Reconstruction loss #{reconstructionLoss} KL divergence #{klDivergence}|]
              when (n `rem` 100  == 0) $ withoutGrad $ do
                -- Sample an image from tha latent space
                z <- sized (size_ @'[BatchSz, ZDim]) <$> randn
                sampledImages <- decode net z
                writeGreyTensorToFile ("generated-images/sampled-"<>show' epoch<>"@"<>show' n<>".jpg")
                  =<< makeGreyGrid (size_ @8) (padding_ @2) 0
                  =<< reshape @'[BatchSz, 1, 28, 28] sampledImages
                -- Reconstruct images from the dataset
                (images', _, _) <- forward net images
                is <- reshape @'[BatchSz, 1, 28, 28] images
                is' <- reshape @'[BatchSz, 1, 28, 28] images'
                writeGreyTensorToFile ("generated-images/reconstructed-"<>show' epoch<>"@"<>show' n<>".jpg")
                  =<< makeGreyGrid (size_ @8) (padding_ @2) 0
                  =<< cat2 @0 is is'
                pure ()
              pure ())
            trainStream)
    [0..epochs-1]
  pure ()
