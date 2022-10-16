{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, DeriveAnyClass, DeriveGeneric, FlexibleContexts, FlexibleInstances  #-}
{-# LANGUAGE FunctionalDependencies, GADTs, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, QuasiQuotes         #-}
{-# LANGUAGE RankNTypes, RecordWildCards, ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies #-}
{-# LANGUAGE TypeInType, TypeOperators, UndecidableInstances                                                                           #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=100 #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver -fplugin Plugin.SimplifyNat #-}

-- https://pytorch.org/assets/images/resnet.png

module Torch.Models.Vision.ResNet where
import           Control.Monad
import           Data.Singletons
import           Data.Singletons.Prelude.Ord
import qualified Data.Text                   as T
import           Generics.Eot                as GE
import           GHC.TypeLits                as TL hiding (type (+), type (-))
import           Torch
import           Torch.Datasets.Augmentation
import           Torch.Datasets.Common
import           Data.Default
import           Lens.Micro

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

residualBlock :: forall inChannels outChannels stride inSz outSz batch.
                _ =>
                  SmallResNetBlock KCpu inChannels outChannels 3 3
                -> DataPurpose
                -> (Tensor TFloat KCpu '[batch, inChannels, inSz, inSz] -> IO (Tensor TFloat KCpu '[batch, outChannels, outSz, outSz]))
                -> Tensor TFloat KCpu '[batch, inChannels, inSz, inSz] -> IO (Tensor TFloat KCpu '[batch, outChannels, outSz, outSz])
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

data ResNet18 ki =
  ResNet18  { r18_pre  :: (ConvParam 'TFloat ki 64 '[3, 7, 7], BatchNormState 'TFloat ki '[64]),
              r18_l1   :: (SmallResNetBlock ki 64 64 3 3, SmallResNetBlock ki 64 64 3 3),
              r18_l2   :: (SmallResNetBlock ki 64 128 3 3, SmallResNetBlock ki 128 128 3 3),
              r18_l2cn :: (ConvParam 'TFloat ki 128 '[64, 3, 3], BatchNormState 'TFloat ki '[128]),
              r18_l3   :: (SmallResNetBlock ki 128 256 3 3, SmallResNetBlock ki 256 256 3 3),
              r18_l3cn :: (ConvParam 'TFloat ki 256 '[128, 3, 3], BatchNormState 'TFloat ki '[256]),
              r18_l4   :: (SmallResNetBlock ki 256 512 3 3, SmallResNetBlock ki 512 512 3 3),
              r18_l4cn :: (ConvParam 'TFloat ki 512 '[256, 3, 3], BatchNormState 'TFloat ki '[512]),
              r18_l    :: LinearParam 'TFloat ki 512 1000
            }
  deriving(Generic,ParameterNames,Stored,Initialize,ToTensors,ToParameters)

resnet18Features :: forall batch. ((1 <=? batch) ~ 'True, _)
  => ResNet18 KCpu
  -> DataPurpose
  -> Tensor 'TFloat 'KCpu '[batch, 3, 224, 224]
  -> IO (Tensor 'TFloat 'KCpu '[batch, 1000])
resnet18Features ResNet18{..} isTraining =
      conv2d InChannels (outChannels_ @64) (kernel_ @'(7,7)) (stride_ @'(2,2)) (padding_ @'(3,3)) (dilation_ @'(1,1)) (groups_ @1) (r18_pre ^. _1)
  >=> batchNorm2d_ (r18_pre ^. _2) Nothing def def isTraining
  >=> relu_
  >=> maxPool2d (kernel_ @'(3,3)) (stride_ @'(2,2)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (ceilMode_ @False)
  >=> (pure . fst)
  --
  >=> residualBlock @64 @64 @1 (fst r18_l1) isTraining pure
  >=> residualBlock @64 @64 @1 (snd r18_l1) isTraining pure
  --
  >=> residualBlock @64 @128 @2 (fst r18_l2) isTraining
           (\x -> do
               r <- conv2d (inChannels_ @64) (outChannels_ @128) (kernel_ @'(3,3))
                   (stride_ @'(2,2)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) (fst r18_l2cn) x
               batchNorm2d_ (snd r18_l2cn) Nothing def def isTraining r)
  >=> residualBlock @128 @128 @1 (snd r18_l2) isTraining pure
  --
  >=> residualBlock @128 @256 @2 (fst r18_l3) isTraining
           (\x -> do
               r <- conv2d (inChannels_ @128) (outChannels_ @256) (kernel_ @'(3,3))
                   (stride_ @'(2,2)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) (fst r18_l3cn) x
               batchNorm2d_ (snd r18_l3cn) Nothing def def isTraining r)
  >=> residualBlock @256 @256 @1 (snd r18_l3) isTraining pure
  --
  >=> residualBlock @256 @512 @2 (fst r18_l4) isTraining
           (\x -> do
               r <- conv2d (inChannels_ @256) (outChannels_ @512) (kernel_ @'(3,3))
                   (stride_ @'(2,2)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) (fst r18_l4cn) x
               batchNorm2d_ (snd r18_l4cn) Nothing def def isTraining r)
  >=> residualBlock @512 @512 @1 (snd r18_l4) isTraining pure
  --
  >=> adaptiveAvgPool2d (outFeatures_ @'[1,1])
  >=> view @'[batch, 512]
  >=> linear (inFeatures_ @512) (outFeatures_ @1000) r18_l
