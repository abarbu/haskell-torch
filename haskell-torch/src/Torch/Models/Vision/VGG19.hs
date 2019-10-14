{-# LANGUAGE AllowAmbiguousTypes, ConstraintKinds, DataKinds, DeriveAnyClass, DeriveGeneric, ExistentialQuantification               #-}
{-# LANGUAGE ExtendedDefaultRules, FlexibleContexts, FlexibleInstances, GADTs, KindSignatures, MultiParamTypeClasses                 #-}
{-# LANGUAGE OverloadedStrings, PartialTypeSignatures, PolyKinds, RankNTypes, ScopedTypeVariables, TemplateHaskell, TypeApplications #-}
{-# LANGUAGE TypeFamilies, TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances                                   #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=100 -Wno-partial-type-signatures #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver #-}

-- | VGG19 using the pretrained weights from torchvision. Don't forget to use
-- @standardRGBNormalization@ when running this on any images.

module Torch.Models.Vision.VGG19 where
import           Control.Monad
import           Data.Singletons
import           Data.Singletons.Prelude.Ord
import qualified Data.Text                   as T
import           Generics.Eot                as GE
import           GHC.TypeLits                as TL hiding (type (+), type (-))
import           Torch

data VGG19Net ki = VGG19Net
                   ((ConvParam TFloat ki 64 '[3, 3, 3],
                     ConvParam TFloat ki 64 '[64, 3, 3]),
                    (ConvParam TFloat ki 128 '[64, 3, 3],
                     ConvParam TFloat ki 128 '[128, 3, 3]),
                    (ConvParam TFloat ki 256 '[128, 3, 3],
                     ConvParam TFloat ki 256 '[256, 3, 3],
                     ConvParam TFloat ki 256 '[256, 3, 3],
                     ConvParam TFloat ki 256 '[256, 3, 3]),
                    (ConvParam TFloat ki 512 '[256, 3, 3],
                     ConvParam TFloat ki 512 '[512, 3, 3],
                     ConvParam TFloat ki 512 '[512, 3, 3],
                     ConvParam TFloat ki 512 '[512, 3, 3]),
                    (ConvParam TFloat ki 512 '[512, 3, 3],
                     ConvParam TFloat ki 512 '[512, 3, 3],
                     ConvParam TFloat ki 512 '[512, 3, 3],
                     ConvParam TFloat ki 512 '[512, 3, 3]))
                   (LinearParam TFloat ki 25088 4096,
                    LinearParam TFloat ki 4096 4096,
                    LinearParam TFloat ki 4096 1000)
                 deriving (Generic,ParameterNames,Stored,ToTensors,ToParameters)

pathToVGG19 = getCachedFileOrDownload "https://www.mediafire.com/file/4qc4tfnqi1tdbyk/vgg19.pt/file" "50a6bf59bb777695a2dce1080b6dd805" "vgg19.pt" "models"

loadVGG19 :: IO (VGG19Net KCpu)
loadVGG19 = do
  fname <- pathToVGG19
  m <- readStoredModel fname
  a <- loadWithNames m (((("features.0.weight", "features.0.bias"),
                         ("features.2.weight", "features.2.bias")),
                        (("features.5.weight", "features.5.bias"),
                         ("features.7.weight", "features.7.bias")),
                        (("features.10.weight", "features.10.bias"),
                         ("features.12.weight", "features.12.bias"),
                         ("features.14.weight", "features.14.bias"),
                         ("features.16.weight", "features.16.bias")),
                        (("features.19.weight", "features.19.bias"),
                         ("features.21.weight", "features.21.bias"),
                         ("features.23.weight", "features.23.bias"),
                         ("features.25.weight", "features.25.bias")),
                        (("features.28.weight", "features.28.bias"),
                         ("features.30.weight", "features.30.bias"),
                         ("features.32.weight", "features.32.bias"),
                         ("features.34.weight", "features.34.bias"))),
                       (("classifier.0.weight", "classifier.0.bias"),
                        ("classifier.3.weight", "classifier.3.bias"),
                        ("classifier.6.weight", "classifier.6.bias")))
  pure a

vgg19Features :: _ => _ -> Tensor TFloat ki '[1,3,224,244] -> IO (Tensor TFloat ki '[1, 512, 7, 7])
vgg19Features ((w1, w1'), (w2, w2'), (w3, w3', w3'', w3'''), (w4, w4', w4'', w4'''), (w5, w5', w5'', w5''')) = do
     conv2d (inChannels_ @3) (outChannels_ @64) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w1
  >=> relu_
  >=> conv2d (inChannels_ @64) (outChannels_ @64) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w1'
  >=> relu_
  >=> maxPool2d (kernel_ @'(2,2)) (stride_ @'(2,2)) (padding_ @'(0,0)) (dilation_ @'(1,1)) (ceilMode_ @False)
  >=> pure . fst
  --
  >=> conv2d (inChannels_ @64) (outChannels_ @128) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w2
  >=> relu_
  >=> conv2d (inChannels_ @128) (outChannels_ @128) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w2'
  >=> relu_
  >=> maxPool2d (kernel_ @'(2,2)) (stride_ @'(2,2)) (padding_ @'(0,0)) (dilation_ @'(1,1)) (ceilMode_ @False)
  >=> pure . fst
  --
  >=> conv2d (inChannels_ @128) (outChannels_ @256) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w3
  >=> relu_
  >=> conv2d (inChannels_ @256) (outChannels_ @256) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w3'
  >=> relu_
  >=> conv2d (inChannels_ @256) (outChannels_ @256) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w3''
  >=> relu_
  >=> conv2d (inChannels_ @256) (outChannels_ @256) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w3'''
  >=> relu_
  >=> maxPool2d (kernel_ @'(2,2)) (stride_ @'(2,2)) (padding_ @'(0,0)) (dilation_ @'(1,1)) (ceilMode_ @False)
  >=> pure . fst
  --
  >=> conv2d (inChannels_ @256) (outChannels_ @512) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w4
  >=> relu_
  >=> conv2d (inChannels_ @512) (outChannels_ @512) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w4'
  >=> relu_
  >=> conv2d (inChannels_ @512) (outChannels_ @512) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w4''
  >=> relu_
  >=> conv2d (inChannels_ @512) (outChannels_ @512) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w4'''
  >=> relu_
  >=> maxPool2d (kernel_ @'(2,2)) (stride_ @'(2,2)) (padding_ @'(0,0)) (dilation_ @'(1,1)) (ceilMode_ @False)
  >=> pure . fst
  --
  >=> conv2d (inChannels_ @512) (outChannels_ @512) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w5
  >=> relu_
  >=> conv2d (inChannels_ @512) (outChannels_ @512) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w5'
  >=> relu_
  >=> conv2d (inChannels_ @512) (outChannels_ @512) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w5''
  >=> relu_
  >=> conv2d (inChannels_ @512) (outChannels_ @512) (kernel_ @'(3,3)) (stride_ @'(1,1)) (padding_ @'(1,1)) (dilation_ @'(1,1)) (groups_ @1) w5'''
  >=> relu_
  >=> maxPool2d (kernel_ @'(2,2)) (stride_ @'(2,2)) (padding_ @'(0,0)) (dilation_ @'(1,1)) (ceilMode_ @False)
  >=> pure . fst

vgg19Classifier (w1, w2, w3) dataPurpose = do
     linear (inFeatures_ @(512 TL.* 7 TL.* 7)) (outFeatures_ @4096) w1
  >=> relu_
  >=> dropout 0.5 dataPurpose
  >=> linear (inFeatures_ @4096) (outFeatures_ @4096) w2
  >=> relu_
  >=> dropout 0.5 dataPurpose
  >=> linear (inFeatures_ @4096) (outFeatures_ @1000) w3

vgg19Forward (VGG19Net w1s w2s) dataPurpose = do
     vgg19Features w1s
  >=> adaptiveAvgPool2d (outFeatures_ @'[7,7])
  >=> flatten
  >=> vgg19Classifier w2s dataPurpose
