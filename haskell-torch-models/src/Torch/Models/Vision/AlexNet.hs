{-# LANGUAGE AllowAmbiguousTypes, ConstraintKinds, DataKinds, DeriveAnyClass, DeriveGeneric, ExtendedDefaultRules, FlexibleContexts  #-}
{-# LANGUAGE FlexibleInstances, GADTs, KindSignatures, MultiParamTypeClasses, OverloadedStrings, PolyKinds, RankNTypes               #-}
{-# LANGUAGE ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies, TypeInType, TypeOperators #-}
{-# LANGUAGE UndecidableInstances                                                                                                    #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=50 #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver #-}

-- | AlexNet using the pretrained weights from torchvision. Don't forget to use
-- @standardRGBNormalization@ when running this on any images.

module Torch.Models.Vision.AlexNet where
import           Control.Monad
import           Data.Singletons
import           Data.Singletons.Prelude.Ord
import           Generics.Eot
import           GHC.TypeLits                as TL
import           Torch
import           Torch.Datasets

data AlexNet ki = AlexNet
                  (ConvParam TFloat ki 64 '[3, 11, 11]
                  ,ConvParam TFloat ki 192 '[64, 5, 5]
                  ,ConvParam TFloat ki 384 '[192, 3, 3]
                  ,ConvParam TFloat ki 256 '[384, 3, 3]
                  ,ConvParam TFloat ki 256 '[256, 3, 3])
                  (LinearParam TFloat ki 9216 4096
                  ,LinearParam TFloat ki 4096 4096,
                   LinearParam TFloat ki 4096 1000)
  deriving (Generic,ParameterNames,Stored)

pathToAlexNet = getCachedFileOrDownload "https://www.mediafire.com/file/4p6acunb5ykle5a/alexnet.pt/file" "ba9248ae47a1887ed6623cc568a9755e" "alexnet.pt" "models"

loadAlexNet :: IO (AlexNet KCpu)
loadAlexNet = do
  fname <- pathToAlexNet
  m <- readStoredModel fname
  a <- loadWithNames m ((("features.0.weight",   "features.0.bias"),
                        ("features.3.weight",   "features.3.bias"),
                        ("features.6.weight",   "features.6.bias"),
                        ("features.8.weight",   "features.8.bias"),
                        ("features.10.weight",  "features.10.bias")),
                       (("classifier.1.weight", "classifier.1.bias"),
                        ("classifier.4.weight", "classifier.4.bias"),
                        ("classifier.6.weight", "classifier.6.bias")))
  pure a

alexNetFeatures (w1, w2, w3, w4, w5) =
     conv2d (inChannels_ @3) (outChannels_ @64) (kernel_ @'(11,11)) (stride_ @'(4,4)) (padding_ @'(2,2)) (dilation_ @'(1,1)) (groups_ @1) w1
  >=> relu_
  >=> maxPool2d (kernel_ @'(3,3)) (stride_ @'(2,2)) (padding_ @'(0,0)) (dilation_ @'(1,1)) (ceilMode_ @False)
  >=> pure . fst
  >=> conv2d (inChannels_ @64) (outChannels_ @192) (kernel_ @'(5,5)) (stride_ @'(1,1)) (padding_ @'(2,2)) (dilation_ @'(1,1)) (groups_ @1) w2
  >=> relu_
  >=> maxPool2d (kernel_ @'(3,3)) (stride_ @'(2,2)) (padding_ @'(0,0)) (dilation_ @'(1,1)) (ceilMode_ @False)
  >=> pure . fst
  >=> conv2d (inChannels_ @192) (outChannels_ @384) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w3
  >=> relu_
  >=> conv2d (inChannels_ @384) (outChannels_ @256) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w4
  >=> relu_
  >=> conv2d (inChannels_ @256) (outChannels_ @256) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w5
  >=> relu_
  >=> maxPool2d (kernel_ @'(3,3)) (stride_ @'(2,2)) (padding_ @'(0,0)) (dilation_ @'(1,1)) (ceilMode_ @False)
  >=> pure . fst

alexNetClassifier (w1, w2, w3) dataPurpose =
     dropout 0.5 dataPurpose
  >=> linear (inFeatures_ @(256 TL.* 6 TL.* 6)) (outFeatures_ @4096) w1
  >=> relu_
  >=> dropout 0.5 dataPurpose
  >=> linear (inFeatures_ @4096) (outFeatures_ @4096) w2
  >=> relu_
  >=> linear (inFeatures_ @4096) (outFeatures_ @1000) w3

alexNetForward (AlexNet w1s w2s) dataPurpose =
     alexNetFeatures w1s
  >=> adaptiveAvgPool2d (outFeatures_ @'[6,6])
  >=> flatten
  >=> alexNetClassifier w2s dataPurpose
