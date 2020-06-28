{-# LANGUAGE AllowAmbiguousTypes, ConstraintKinds, DataKinds, FlexibleContexts, FlexibleInstances, GADTs, KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, RankNTypes    #-}
{-# LANGUAGE ScopedTypeVariables, TypeApplications, TypeFamilies, TypeFamilyDependencies, TypeInType, TypeOperators      #-}
{-# LANGUAGE UndecidableInstances                                                                                        #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures -fconstraint-solver-iterations=10 #-}

-- | Fast image manipulations with ImageMagick on statically-sized images.  Some of
-- these operations convert images to tensors and back, others apply image-like
-- transformations to images or to tensors. If you encounter torchvision Transforms
-- this is where you want to be.
--
-- It's important that operations in this file take the images and tensors they
-- work on as thier the last arguments. This allows us to chain them together.

module Torch.Images where
import           Control.Monad
import           Data.Coerce
import           Data.Singletons
import           Data.Singletons.Prelude  as SP
import           Data.Singletons.TypeLits
import           Data.Text                (Text)
import           Data.Vector.Storable     (Vector)
import qualified Data.Vector.Storable     as V
import           Data.Word
import           Foreign.C.Types
import           Foreign.ForeignPtr
import qualified Foreign.ImageMagick      as I
import           GHC.Int
import qualified GHC.TypeLits             as TL
import           System.Random            (randomRIO)
import qualified Torch.C.Types                as C
import qualified Torch.C.Tensor           as C
import           Torch.Inplace
import           Torch.Misc
import           Torch.Tensor
import           Torch.Types

-- | Images are statically typed. They wrap an ImageMagick Wand that contains an
-- image and its properties.
data Image (depth :: Nat) (szh :: Nat) (szw :: Nat) where
  Image :: (SingI depth, SingI szh, SingI szw)
        => ForeignPtr I.CMagickWand -> Image depth szh szw

-- | Internal
imageStorageForTensorTy :: forall (ty :: TensorType). (SingI ty) => I.StorageType
imageStorageForTensorTy = case (sing :: Sing ty) of
                              STBool   -> I.CharPixel
                              STByte   -> I.CharPixel
                              STChar   -> I.CharPixel
                              STShort  -> I.ShortPixel
                              STInt    -> I.LongPixel
                              STLong   -> I.LongLongPixel
                              STHalf   -> error "Half-sized floats can't be converted to images" -- TODO Enforce this in the type
                              STFloat  -> I.FloatPixel
                              STDouble -> I.DoublePixel

-- * Reading and writing images to files and buffers

-- | You specify what size and image depth you want. Whatever exists on disk
-- will get converted to this size and depth with some sensible defaults.
readImageFromFile :: forall depth szh szw. (SingI szw, SingI szh, SingI depth)
                  => Text -> IO (Image depth szh szw)
readImageFromFile fname = do
  I.initialize
  w <- I.newWand
  I.readImage w fname
  I.resizeBilinear w (demoteN @szh) (demoteN @szw)
  I.convertToColorspace w (case demote @depth of
                              1 -> I.GRAYColorspace
                              3 -> I.RGBColorspace
                              4 -> I.CMYKColorspace
                              -- TODO Move this to the type level
                              _ -> error $ "Not sure what color space to use for a depth " <> show (demote @depth) <> " image")
  -- NB This is only as a backup in case we don't understand some corner case in
  -- ImageMagick. It should never fail, but costs us nothing to verify anyway.
  de <- I.imageDepth w
  unless (de == demoteN @depth) $ error "Image depth mismatch"
  iw <- I.imageWidth w
  unless (iw == demoteN @szw) $ error "Image width mismatch"
  ih <- I.imageHeight w
  unless (ih == demoteN @szh) $ error "Image height mismatch"
  pure $ Image w

-- | You must specify the depth and size of the on-disk image. Use
-- @fileImageProperties@ along with dependent types if you are not sure of what
-- these should be.
readImageFromFileWithSize :: forall depth szh szw. (SingI szw, SingI szh, SingI depth)
                  => Text -> IO (Image depth szh szw)
readImageFromFileWithSize fname = do
  I.initialize
  w <- I.newWand
  I.readImage w fname
  de <- I.imageDepth w
  unless (de == demoteN @depth) $ error "Image depth mismatch"
  iw <- I.imageWidth w
  unless (iw == demoteN @szw) $ error "Image width mismatch"
  ih <- I.imageHeight w
  unless (ih == demoteN @szh) $ error "Image height mismatch"
  pure $ Image w

-- | Return the dimensions of an image in a file. (depth, height, width)
fileImageProperties :: Text -> IO (Int32, Int32, Int32)
fileImageProperties fname = do
  I.initialize
  w <- I.newWand
  I.readImage w fname
  iw <- I.imageWidth w
  ih <- I.imageHeight w
  de <- I.imageDepth w
  pure (coerce de, coerce ih, coerce iw)

-- | Encode an image as a JPEG file into a vector, you can write this to disk
imageToJPEG :: Image depth szh szw -> IO (Vector Word8)
imageToJPEG (Image w) = I.writeImageToBuffer w I.F_JPEG

-- | Encode an image as a PNG file into a vector, you can write this to disk
imageToPNG :: Image depth szh szw -> IO (Vector Word8)
imageToPNG (Image w) = I.writeImageToBuffer w I.F_PNG

-- | Encode an image as a PDF file into a vector, you can write this to disk
imageToPDF :: Image depth szh szw -> IO (Vector Word8)
imageToPDF (Image w) = I.writeImageToBuffer w I.F_PDF

-- | Encode an image as a PNM file into a vector, you can write this to disk
imageToPNM :: Image depth szh szw -> IO (Vector Word8)
imageToPNM (Image w) = I.writeImageToBuffer w I.F_PNM

-- | Write an image to a file
writeImageToFile :: Text -> Image depth szh szw -> IO Bool
writeImageToFile fname (Image w) = I.writeImage w fname

-- * Convert between tensors and images

-- | Convert an RGB, 3-plane, tensor to an image. We take care of the right
-- format conversions internally. Floating point tensors should be in the range
-- [0,1], others should be [0,255].
rgbTensorToImage :: forall szh szw ty.
                   (KnownNat szw, KnownNat szh, SingI ty)
                 => Tensor ty KCpu '[3,szh,szw] -> IO (Image 3 szh szw)
rgbTensorToImage ten = do
  I.initialize
  w <- I.newWand
  ten' <- reshape ten
  ten'' <- t $ sized (size_ @'[3,szh TL.* szw]) ten'
  withDataPtr ten''
    (\ptr len -> I.createImageFromPtr w I.RGB (imageStorageForTensorTy @ty) (demoteN @szw) (demoteN @szh) (coerce ptr))
  pure $ Image w

-- | Convert a greyscale, 1-plane, tensor to an image. We take care of the right
-- format conversions internally. Floating point tensors should be in the range
-- [0,1], others should be [0,255].
greyTensorToImage :: forall szh szw ty.
                    (KnownNat szw, KnownNat szh, SingI ty)
                  => Tensor ty KCpu '[1,szh,szw] -> IO (Image 1 szh szw)
greyTensorToImage ten = do
  I.initialize
  w <- I.newWand
  ten' <- reshape ten
  ten'' <- t $ sized (size_ @'[1,szh TL.* szw]) ten'
  withDataPtr ten''
    (\ptr len -> I.createImageFromPtr w I.I (imageStorageForTensorTy @ty) (demoteN @szw) (demoteN @szh) (coerce ptr))
  pure $ Image w

-- | Convert an RGB, 3-plane, image to a tensor.
rgbImageToTensor :: forall szh szw ty.
                   (SingI ty, SingI (szh TL.* szw)
                   ,Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
                   ,V.Storable (TensorTyToHs ty), V.Storable (TensorTyToHsC ty))
                 => Image 3 szh szw -> IO (Tensor ty KCpu '[3,szh,szw])
rgbImageToTensor (Image w) = do
  ten <- empty
  withDataPtr (sized (size_ @'[szh TL.* szw,3]) ten)
              (\ptr len ->
                 I.writeImagePixelsToPtr w I.RGB (imageStorageForTensorTy @ty) (coerce ptr)
                 -- pure ()
              )
  reshape =<< t ten

-- | Convert an greyscale, 1-plane, image to a tensor.
greyImageToTensor :: forall szh szw ty.
                   (SingI ty, SingI (szh TL.* szw)
                   ,Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
                   ,V.Storable (TensorTyToHs ty), V.Storable (TensorTyToHsC ty))
                 => Image 1 szh szw -> IO (Tensor ty KCpu '[1,szh,szw])
greyImageToTensor (Image w) = do
  ten <- empty
  withDataPtr (sized (size_ @'[szh TL.* szw,1]) ten)
              (\ptr len -> I.writeImagePixelsToPtr w I.I (imageStorageForTensorTy @ty) (coerce ptr))
  reshape =<< t ten

-- * Operations on tensors that are images

-- TODO Using the trick for StoredModel parameters, write a generic version of
-- normalize. Should go elsewhere though to not clutter up this file.
normalizeGreyImageTensor :: forall ty ki szh szw.
                           Tensor ty ki '[1]
                         -> Tensor ty ki '[1]
                         -> Tensor ty ki '[1,szh,szw]
                         -> IO (Tensor ty ki '[1,szh,szw])
normalizeGreyImageTensor (Tensor mean _) (Tensor std _) (Tensor t _) = do
  t' <- C.clone__tm t (fromIntegral $ fromEnum  C.MemoryFormatPreserve)
  s <- toCScalar @ty @ki 1
  C.sub_mts t' mean s
  C.div__mt t' std
  pure $ Tensor t' Nothing

-- TODO Test this normalizeRGBImageTensor
-- TODO This should be marked as inplace
normalizeRGBImageTensor :: forall ty ki szh szw. (Num (TensorTyToHs ty), KnownNat szh, KnownNat szw)
                        => Tensor ty ki '[3]
                        -> Tensor ty ki '[3]
                        -> Tensor ty ki '[3, szh, szw]
                        -> IO (Tensor ty ki '[3, szh, szw])
normalizeRGBImageTensor mean std t = do
  undefined -- FIXME
  -- t' <- clone t
  -- sub_ t' =<< expand' @'[3,szh,szw] mean
  -- div_ t' =<< expand' @'[3,szh,szw] std
  -- pure t'

    -- • Couldn't match type ‘Case_6989586621680089195
    --                          '[3]
    --                          '[3, szh, szw]
    --                          3
    --                          '[]
    --                          szw
    --                          '[szh, 3]
    --                          (DefaultEq 3 szw || DefaultEq szw 1)’
    --                  with ‘'True’

-- normalizeRGBImageTensor :: forall ty ki szc szh szw.
--                           Tensor ty ki '[szc]
--                         -> Tensor ty ki '[szc]
--                         -> Tensor ty ki '[szc,szh,szw]
--                         -> IO (Tensor ty ki '[szc,szh,szw])
-- normalizeRGBImageTensor (Tensor t _) (Tensor mean _) (Tensor std _) = do
--   t' <- C.clone t
--   s <- toCScalar @ty @ki 1
--   C.sub_ t' mean s
--   C.div_ t' std
--   pure $ Tensor t' Nothing

-- | All standard torchvision models take as input images normalized with the
-- following function. You must rememebr to use this and to process all of your
-- datastreams with it!
-- TODO This should be marked as inplace
standardRGBNormalization :: (SingI ty, KnownNat szh, KnownNat szw, Num (TensorTyToHs ty),
                            V.Storable (TensorTyToHs ty), V.Storable (TensorTyToHsC ty),
                            Fractional (TensorTyToHsC ty)) =>
                           Tensor ty 'KCpu '[3, szh, szw] -> IO (Tensor ty 'KCpu '[3, szh, szw])
standardRGBNormalization t = do
  mean <- fromVector (V.fromList [0.485, 0.456, 0.406])
  std <- fromVector (V.fromList [0.229, 0.224, 0.225])
  normalizeRGBImageTensor mean std t

-- | Since torchvision models take as input images normalized with
-- @standardRGBNormalization@, if we want to view images we have to undo the
-- normalization. That's what this function does.
-- https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
-- TODO This should be inplace
standardRGBDeNormalization :: (SingI ty, KnownNat szh, KnownNat szw, Num (TensorTyToHs ty),
                            V.Storable (TensorTyToHs ty), V.Storable (TensorTyToHsC ty),
                            Fractional (TensorTyToHsC ty)) =>
                           Tensor ty 'KCpu '[3, szh, szw] -> IO (Tensor ty 'KCpu '[3, szh, szw])
standardRGBDeNormalization t = do
  mean <- fromVector (V.fromList [-2.12, -2.04, -1.80])
  std <- fromVector (V.fromList [4.36, 4.46, 4.44])
  normalizeRGBImageTensor mean std t

-- * Reading and writing tensors to files and buffers

-- | Encode an RGB tensor as JPEG and write it to a buffer of bytes
rgbTensorToVector :: (KnownNat szw, KnownNat szh, SingI ty)
                  => Tensor ty 'KCpu '[3, szh, szw] -> IO (Vector Word8)
rgbTensorToVector ten = rgbTensorToImage ten >>= imageToJPEG

-- | Encode a greyscale tensor as JPEG and write it to a buffer of bytes
greyTensorToVector :: (KnownNat szw, KnownNat szh, SingI ty)
                   => Tensor ty 'KCpu '[1, szh, szw] -> IO (Vector Word8)
greyTensorToVector ten = greyTensorToImage ten >>= imageToJPEG

-- | Read an RGB tensor from a file
readRGBTensorFromFile :: forall szh szw ty.
  (KnownNat szh, KnownNat szw, SingI ty, Num (TensorTyToHs ty),
    Num (TensorTyToHsC ty), V.Storable (TensorTyToHs ty),
    V.Storable (TensorTyToHsC ty)) =>
  Text -> IO (Tensor ty 'KCpu '[3, szh, szw])
readRGBTensorFromFile fname = readImageFromFile fname >>= rgbImageToTensor

-- | Read a greyscale tensor from a file
readGreyTensorFromFile :: forall szh szw ty.
  (KnownNat szh, KnownNat szw, SingI ty, Num (TensorTyToHs ty),
   Num (TensorTyToHsC ty), V.Storable (TensorTyToHs ty),
   V.Storable (TensorTyToHsC ty)) =>
  Text -> IO (Tensor ty 'KCpu '[1, szh, szw])
readGreyTensorFromFile fname = readImageFromFile fname >>= greyImageToTensor

-- | Write an RGB tensor to a file
writeRGBTensorToFile ::
  (KnownNat szw, KnownNat szh, SingI ty) =>
  Text -> Tensor ty 'KCpu '[3, szh, szw] -> IO Bool
writeRGBTensorToFile fname ten = rgbTensorToImage ten >>= writeImageToFile fname

-- | Write a greyscale tensor to a file
writeGreyTensorToFile ::
  (KnownNat szw, KnownNat szh, SingI ty) =>
  Text -> Tensor ty 'KCpu '[1, szh, szw] -> IO Bool
writeGreyTensorToFile fname ten = greyTensorToImage ten >>= writeImageToFile fname

-- * Operations on groups of image tensors

-- | Turn a tensor containing many images into a tensor that contains a single grid of those images
makeGreyGrid :: forall (imagesPerRow :: Nat) (padding :: Nat) (ty :: TensorType) (ki :: TensorKind) (nr :: Nat) (szh :: Nat) (szw :: Nat) (rows :: Nat).
              _
            => Size imagesPerRow
            -> Padding padding
            -> TensorTyToHs ty
            -> Tensor ty 'KCpu '[nr, 1, szh, szw]
            -> IO (Tensor ty 'KCpu '[1
                                   ,padding + (GridRows imagesPerRow nr) TL.* (szh+padding)
                                   ,padding + imagesPerRow TL.* (padding+szw)])
makeGreyGrid _ _ padValue images = do
  grid <- full padValue
  --
  let nmaps = demoteN @nr
  let xmaps = Prelude.min (demoteN @imagesPerRow) nmaps
  let ymaps = ceiling $ fromIntegral nmaps / fromIntegral xmaps
  let height = demoteN @szh
  let width = demoteN @szw
  let padding = demoteN @padding
  let y = 0
  let x = 0
  let k = 0
  mapM_ (\y -> do
           gridViewH <- narrowFromToByLength (dimension_ @1) (size_ @szh) grid (padding + y * (height + padding))
           mapM_ (\x -> when (y + x*ymaps < nmaps) $ do
                           gridView <- narrowFromToByLength (dimension_ @2) (size_ @szw) gridViewH (padding + x * (width + padding))
                           copy_ gridView =<< select @0 images (y + x*ymaps)
                           pure ())
             [0..xmaps-1])
    [0..ymaps-1]
  pure grid

-- | Turn a tensor containing many images into a tensor that contains a single grid of those images
makeRGBGrid :: forall (imagesPerRow :: Nat) (padding :: Nat) (ty :: TensorType) (ki :: TensorKind) (nr :: Nat) (szh :: Nat) (szw :: Nat) (rows :: Nat).
              _
            => Size imagesPerRow
            -> Padding padding
            -> TensorTyToHs ty
            -> Tensor ty 'KCpu '[nr, 3, szh, szw]
            -> IO (Tensor ty 'KCpu '[3
                                   ,padding + (GridRows imagesPerRow nr) TL.* (szh+padding)
                                   ,padding + imagesPerRow TL.* (padding+szw)])
makeRGBGrid _ _ padValue images = do
  grid <- full padValue
  --
  let nmaps = demoteN @nr
  let xmaps = Prelude.min (demoteN @imagesPerRow) nmaps
  let ymaps = ceiling $ fromIntegral nmaps / fromIntegral xmaps
  let height = demoteN @szh
  let width = demoteN @szw
  let padding = demoteN @padding
  let y = 0
  let x = 0
  let k = 0
  mapM_ (\y -> do
           gridViewH <- narrowFromToByLength (dimension_ @1) (size_ @szh) grid (padding + y * (height + padding))
           mapM_ (\x -> when (y + x*ymaps < nmaps) $ do
                           gridView <- narrowFromToByLength (dimension_ @2) (size_ @szw) gridViewH (padding + x * (width + padding))
                           copy_ gridView =<< select @0 images (y + x*ymaps)
                           pure ())
             [0..xmaps-1])
    [0..ymaps-1]
  pure grid

-- * Convert images

-- | Convert an image to greyscale
convertToGreyscale_ :: Image depth szh szw -> IO (Image 1 szh szw)
convertToGreyscale_ (Image w) = I.convertToColorspace w I.GRAYColorspace >> pure (Image w)

-- | Convert an image to RGB
convertToRGB_ :: Image depth szh szw -> IO (Image 3 szh szw)
convertToRGB_ (Image w) = I.convertToColorspace w I.RGBColorspace >> pure (Image w)

-- | Convert an image to HSL, Hue Saturation Lightness
convertToHSL_ :: Image depth szh szw -> IO (Image 3 szh szw)
convertToHSL_ (Image w) = I.convertToColorspace w I.HSLColorspace >> pure (Image w)

-- | Convert an image to XYZ, a CIE perceptually-calibrated color space so that
-- linear distances between colors correspond roughly to what humans would
-- estimate them to be.
convertToXYZ_ :: Image depth szh szw -> IO (Image 3 szh szw)
convertToXYZ_ (Image w) = I.convertToColorspace w I.XYZColorspace >> pure (Image w)

-- | Crop the canter of an image, you provide the output size.
centerCropImage_ :: forall szh' szw' depth szh szw.
                   ((szh' <= szh) ~ True, (szw' <= szw) ~ True, SingI szw', SingI szh')
                 => Image depth szh szw -> IO (Image depth szh' szw')
centerCropImage_ (Image wand) = do
  let h = demoteN @szh
  let w = demoteN @szw
  let h' = demoteN @szh'
  let w' = demoteN @szw'
  I.crop wand h' w' ((h - h') `Prelude.div` 2) ((w - w') `Prelude.div` 2)
  pure $ Image wand

-- * Augmentation

-- | Clone an image, you get a copy with separate storage
cloneImage :: Image depth szh szw -> IO (Image depth szh szw)
cloneImage (Image i) = Image <$> I.cloneWand i

-- | Crop an image, you provide the output size as a type parameter and the x,y
-- shift as a value.
cropImage_ :: forall szh' szw' depth szh szw.
       ((szh' <= szh) ~ True, (szw' <= szw) ~ True, SingI szw', SingI szh')
     => Int -> Int -> Image depth szh szw -> IO (Image depth szh' szw')
cropImage_ x y (Image w) = do
  I.crop w (demoteN @szw') (demoteN @szh') (fromIntegral x) (fromIntegral y) >> pure ()
  pure $ Image w

-- | Crop into five pieces, four corners and a center. You provide the output
-- size as a type parameter.
fiveCrop :: forall croph cropw depth szh szw.
           ((croph <= szh) ~ True, (cropw <= szw) ~ True
           ,SingI croph, SingI cropw
           ,SingI szh, SingI szw)
         => Image depth szh szw -> IO (Image depth croph cropw
                                    ,Image depth croph cropw
                                    ,Image depth croph cropw
                                    ,Image depth croph cropw
                                    ,Image depth croph cropw)
fiveCrop i = do
  let w = demoteN @szh
  let h = demoteN @szw
  let croph = demoteN @croph
  let cropw = demoteN @cropw
  i1 <- cloneImage i
  i1' <- cropImage_ @croph @cropw 0 0 i1
  i2 <- cloneImage i
  i2' <- cropImage_ @croph @cropw (w - cropw) 0 i2
  i3 <- cloneImage i
  i3' <- cropImage_ @croph @cropw 0 (h - croph) i3
  i4 <- cloneImage i
  i4' <- cropImage_ @croph @cropw (w - cropw) (h - croph) i4
  i5 <- cloneImage i
  i5' <- centerCropImage_ @croph @cropw i5
  pure (i1', i2', i3', i4', i5')

-- | Crop an image into ten pieces, two five crops after flipping the image. about an axis.
tenCrop verticalFlip i = do
  (i1,i2,i3,i4,i5) <- fiveCrop i
  i' <- cloneImage i
  if verticalFlip then
    vflip_ i' else
    hflip_ i'
  (i1',i2',i3',i4',i5') <- fiveCrop i'
  pure (i1,i2,i3,i4,i5,i1',i2',i3',i4',i5')

-- | Flip an image horizontally.
hflip_ :: Image depth szh szw -> IO (Image depth szh szw)
hflip_ (Image w) = do
  I.flop w
  pure $ Image w

-- | Flip an image vertically.
vflip_ :: Image depth szh szw -> IO (Image depth szh szw)
vflip_ (Image w) = do
  I.flop w
  pure $ Image w

-- | Pad an image with a constant, you provide the pad size as type argument.
constantPad :: forall padh padw depth szh szw.
              (SingI padh, SingI padw, SingI (szh + padh), SingI (szw + padw))
            => Double -> Double -> Double -> Image depth szh szw
            -> IO (Image depth (szh + padh) (szw + padw))
constantPad r g b (Image w) = do
  I.pad w (demoteN @padh) (demoteN @padw) (coerce r) (coerce g) (coerce b)
  pure $ Image w

-- * Augmentation with random parameters

-- | Jitter the brightens of an entire image by a given amount
randomBrightnessJitter_ :: Double -> Image depth szh szw -> IO (Image depth szh szw)
randomBrightnessJitter_ brightness i =
  randomBrightnessPerturb_ (Prelude.max 0 (1 - brightness)) (1 + brightness) i

-- | Randomly change the brightens of an entire image within a range
randomBrightnessPerturb_ :: Double -> Double -> Image depth szh szw -> IO (Image depth szh szw)
randomBrightnessPerturb_ brightnessMin brightnessMax i@(Image w)  = do
  factor <- randomRIO (brightnessMin, brightnessMax)
  I.modulate w (coerce factor) 0 0
  pure i

-- | Jitter the contrast of an entire image by a given amount
randomContrastJitter_ :: Double -> Image depth szh szw -> IO (Image depth szh szw)
randomContrastJitter_ contrast i =
  randomContrastPerturb_ (Prelude.max 0 (1 - contrast)) (1 + contrast) i

-- | Randomly change the contrast of an entire image within a range
randomContrastPerturb_ :: Double -> Double -> Image depth szh szw -> IO (Image depth szh szw)
randomContrastPerturb_ contrastMin contrastMax i@(Image w)  = do
  factor <- randomRIO (contrastMin, contrastMax)
  I.modulate w (coerce factor) 0 0
  pure i

-- | Jitter the saturation of an entire image by a given amount
randomSaturationJitter_ :: Double -> Image depth szh szw -> IO (Image depth szh szw)
randomSaturationJitter_ saturation i =
  randomSaturationPerturb_ (Prelude.max 0 (1 - saturation)) (1 + saturation) i

-- | Randomly change the saturation of an entire image within a range
randomSaturationPerturb_ :: Double -> Double -> Image depth szh szw -> IO (Image depth szh szw)
randomSaturationPerturb_ saturationMin saturationMax i@(Image w)  = do
  factor <- randomRIO (saturationMin, saturationMax)
  I.modulate w (coerce factor) 0 0
  pure i

-- | Jitter the hue of an entire image by a given amount
-- Maximum hue jitter_ is -0.5 to 0.5
randomHueJitter_ :: Image depth szh szw -> Double -> IO (Image depth szh szw)
randomHueJitter_ i hue =
  randomHuePerturb_ i (Prelude.max 0 (1 - hue)) (1 + hue)

-- | Randomly change the hue of an entire image within a range
-- Maximum hue perturb_ation is -0.5 to 0.5
randomHuePerturb_ :: Image depth szh szw -> Double -> Double -> IO (Image depth szh szw)
randomHuePerturb_ i@(Image w) hueMin hueMax = do
  factor <- randomRIO (Prelude.max hueMin (-0.5), Prelude.min hueMax 0.5)
  I.modulate w (coerce factor) 0 0
  pure i

-- | Apply a random affine transformation within the provided bounds.
--
-- TODO Doesn't support random scales
randomAffine :: forall scalex scaley depth szh szw.
               (SingI (szh TL.* scalex), SingI (szw TL.* scaley))
             => (Double, Double) -> (Double, Double) -> ((Double, Double), (Double, Double)) -> Image depth szh szw
             -> IO (Image depth (szh TL.* scalex) (szw TL.* scaley))
randomAffine (transx, transy) (rotateMin, rotateMax) ((shearxMin, shearxMax), (shearyMin, shearyMax)) (Image w) = do
  let ih = demoteN @szw
  let iw = demoteN @szh
  transy' <- coerce <$> randomRIO (-ih*transy, ih*transy)
  transx' <- coerce <$> randomRIO (-iw*transx, iw*transx)
  rotate' <- coerce <$> randomRIO (rotateMin, rotateMax)
  sheary' <- coerce <$> randomRIO (shearyMin, shearyMax)
  shearx' <- coerce <$> randomRIO (shearxMin, shearxMax)
  I.affine w transx' transy' rotate' rotate' shearx' sheary'
  pure $ Image w

-- | Randomly crop an image, you provide the output size as a type argument.
randomCrop :: forall szh' szw' depth szh szw.
             ((szh' <= szh) ~ True, (szw' <= szw) ~ True
             ,SingI szw', SingI szh', SingI szw, SingI szh)
           => Image depth szh szw -> IO (Image depth szh' szw')
randomCrop i = do
  let ih = demoteN @szw
  let iw = demoteN @szh
  let ih' = demoteN @szw'
  let iw' = demoteN @szh'
  y <- randomRIO (0, ih - ih')
  x <- randomRIO (0, iw - iw')
  cropImage_ @szh' @szw' x y i

-- | Randomly remove the colors from the image with the given probability. This
-- keeps the depth planes at 3, so it's a desatured RGB image.
randomToGreyscale_ :: Double -> Image 3 szh szw -> IO (Image 3 szh szw)
randomToGreyscale_ p i@(Image w) = do
  b :: Double <- randomRIO (0, 1)
  when (b <= p) $ do
    I.convertToColorspace w I.GRAYColorspace
    I.convertToColorspace w I.RGBColorspace
    pure ()
  pure i

-- | Randomly flip the image horizontally with the given probability.
randomHorizontalFlip_ :: Double -> Image depth szh szw -> IO (Image depth szh szw)
randomHorizontalFlip_ p i = do
  b :: Double <- randomRIO (0, 1)
  when (b <= p) $ do
    hflip_ i
    pure ()
  pure i

-- | Randomly flip the image vertically with the given probability.
randomVerticalFlip_ :: Double -> Image depth szh szw -> IO (Image depth szh szw)
randomVerticalFlip_ p i = do
  b :: Double <- randomRIO (0, 1)
  when (b <= p) $ do
    vflip_ i
    pure ()
  pure i
