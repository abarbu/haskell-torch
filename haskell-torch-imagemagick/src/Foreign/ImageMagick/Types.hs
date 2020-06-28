{-# LANGUAGE OverloadedStrings, QuasiQuotes, ScopedTypeVariables, TemplateHaskell #-}

module Foreign.ImageMagick.Types where
import qualified Data.ByteString                  as BS
import qualified Data.Map                         as Map
import           Data.Text                        (Text)
import qualified Data.Text.Encoding               as T
import           Foreign.C.String
import           Foreign.C.Types
import qualified Language.C.Inline                as C
import qualified Language.C.Inline.Context        as C
import qualified Language.C.Types                 as C
import qualified Language.Haskell.TH              as TH

C.verbatim "#define MAGICKCORE_QUANTUM_DEPTH 16"
C.include "<ImageMagick-7/MagickWand/MagickWand.h>"

data CMagickWand
data CDrawingWand
data CPixelWand

imageMagickCtx :: C.Context
imageMagickCtx = C.baseCtx <> C.funCtx <> C.vecCtx <> C.fptrCtx <> ctx
  where ctx = mempty
          { C.ctxTypesTable = imageMagickTypesTable }

imageMagickTypesTable :: Map.Map C.TypeSpecifier TH.TypeQ
imageMagickTypesTable = Map.fromList
  [ (C.TypeName "bool", [t| C.CBool |])
  , (C.TypeName "MagickWand", [t| CMagickWand |])
  , (C.TypeName "DrawingWand", [t| CDrawingWand |])
  , (C.TypeName "PixelWand", [t| CPixelWand |])
  ]


-- | From the ImageMagick docs:
-- This string reflects the expected ordering of the pixel array. It can be any
-- combination or order of R = red, G = green, B = blue, A = alpha (0 is
-- transparent), O = alpha (0 is opaque), C = cyan, Y = yellow, M = magenta, K =
-- black, I = intensity (for grayscale), P = pad.
data PixelOrder = I     -- ^ grayscale
                | RGB
                | RGBP  -- ^ p is padding
                | RGBA
                | BGR
                | BGRP
                | BGRA  -- ^ p is padding
                deriving (Eq,Show)

toPixelOrderString :: PixelOrder -> Text
toPixelOrderString I    = "I"
toPixelOrderString RGB  = "RGB"
toPixelOrderString RGBP = "RGBP"
toPixelOrderString RGBA = "RGBA"
toPixelOrderString BGR  = "BGR"
toPixelOrderString BGRP = "BGRP"
toPixelOrderString BGRA = "BGRA"

data ImageFormat = F_PNG
                 | F_JPEG
                 | F_PNM
                 | F_PPM
                 | F_PGM
                 | F_PBM
                 | F_PS
                 | F_PDF
                 | F_TTF
                 deriving (Eq)

toFormatString :: ImageFormat -> Text
toFormatString F_PNG  = "PNG"
toFormatString F_JPEG = "JPEG"
toFormatString F_PNM  = "PNM"
toFormatString F_PPM  = "PPM"
toFormatString F_PGM  = "PGM"
toFormatString F_PBM  = "PBM"
toFormatString F_PS   = "PS"
toFormatString F_PDF  = "PDF"
toFormatString F_TTF  =  "TTF"

-- TODO Internal
withPixelOrder :: PixelOrder -> (CString -> IO a) -> IO a
withPixelOrder po f =
  BS.useAsCString (T.encodeUtf8 (toPixelOrderString po)) f

-- TODO Internal
withFormat :: ImageFormat -> (CString -> IO a) -> IO a
withFormat format f =
  BS.useAsCString (T.encodeUtf8 (toFormatString format)) f

data StorageType = UndefinedPixel
                 | CharPixel
                 | DoublePixel
                 | FloatPixel
                 | LongPixel
                 | LongLongPixel
                 | QuantumPixel
                 | ShortPixel
                 deriving (Show, Eq)

instance Enum StorageType where
  toEnum x | x == fromIntegral [C.pure|int { (int)UndefinedPixel }|]  = UndefinedPixel
           | x == fromIntegral [C.pure|int { (int)CharPixel }|]       = CharPixel
           | x == fromIntegral [C.pure|int { (int)DoublePixel }|]     = DoublePixel
           | x == fromIntegral [C.pure|int { (int)FloatPixel }|]      = FloatPixel
           | x == fromIntegral [C.pure|int { (int)LongPixel }|]       = LongPixel
           | x == fromIntegral [C.pure|int { (int)LongLongPixel }|]   = LongLongPixel
           | x == fromIntegral [C.pure|int { (int)QuantumPixel }|]    = QuantumPixel
           | x == fromIntegral [C.pure|int { (int)ShortPixel }|]      = ShortPixel
           | otherwise = error "Cannot convert AttributeKind to enum"
  fromEnum UndefinedPixel = fromIntegral [C.pure|int { (int)UndefinedPixel }|]
  fromEnum CharPixel      = fromIntegral [C.pure|int { (int)CharPixel }|]
  fromEnum DoublePixel    = fromIntegral [C.pure|int { (int)DoublePixel }|]
  fromEnum FloatPixel     = fromIntegral [C.pure|int { (int)FloatPixel }|]
  fromEnum LongPixel      = fromIntegral [C.pure|int { (int)LongPixel }|]
  fromEnum LongLongPixel  = fromIntegral [C.pure|int { (int)LongLongPixel }|]
  fromEnum QuantumPixel   = fromIntegral [C.pure|int { (int)QuantumPixel }|]
  fromEnum ShortPixel     = fromIntegral [C.pure|int { (int)ShortPixel }|]

data ColorSpace = UndefinedColorspace
                | CMYColorspace           -- ^ negated linear RGB colorspace
                | CMYKColorspace          -- ^ CMY with Black separation
                | GRAYColorspace          -- ^ Single Channel greyscale (non-linear) image
                | HCLColorspace
                | HCLpColorspace
                | HSBColorspace
                | HSIColorspace
                | HSLColorspace
                | HSVColorspace           -- ^ alias for HSB
                | HWBColorspace
                | LabColorspace
                | LCHColorspace           -- ^ alias for LCHuv
                | LCHabColorspace         -- ^ Cylindrical (Polar) Lab
                | LCHuvColorspace         -- ^ Cylindrical (Polar) Luv
                | LogColorspace
                | LMSColorspace
                | LuvColorspace
                | OHTAColorspace
                | Rec601YCbCrColorspace
                | Rec709YCbCrColorspace
                | RGBColorspace           -- ^ Linear RGB colorspace
                | ScRGBColorspace         -- ^ ???
                | SRGBColorspace          -- ^ Default: non-linear sRGB colorspace
                | TransparentColorspace
                | LOWERxyYColorspace
                | XYZColorspace           -- ^ IEEE Color Reference colorspace
                | YCbCrColorspace
                | YCCColorspace
                | YDbDrColorspace
                | YIQColorspace
                | YPbPrColorspace
                | YUVColorspace
                | LinearGRAYColorspace     -- ^ Single Channel greyscale (linear) image
                 deriving (Show, Eq)

instance Enum ColorSpace where
  toEnum x | x == fromIntegral [C.pure|int { (int)UndefinedColorspace }|]    = UndefinedColorspace
           | x == fromIntegral [C.pure|int { (int)CMYColorspace }|]          =  CMYColorspace
           | x == fromIntegral [C.pure|int { (int)CMYKColorspace }|]         =  CMYKColorspace
           | x == fromIntegral [C.pure|int { (int)GRAYColorspace }|]         =  GRAYColorspace
           | x == fromIntegral [C.pure|int { (int)HCLColorspace }|]          =  HCLColorspace
           | x == fromIntegral [C.pure|int { (int)HCLpColorspace }|]         =  HCLpColorspace
           | x == fromIntegral [C.pure|int { (int)HSBColorspace }|]          =  HSBColorspace
           | x == fromIntegral [C.pure|int { (int)HSIColorspace }|]          =  HSIColorspace
           | x == fromIntegral [C.pure|int { (int)HSLColorspace }|]          =  HSLColorspace
           | x == fromIntegral [C.pure|int { (int)HSVColorspace }|]          =  HSVColorspace
           | x == fromIntegral [C.pure|int { (int)HWBColorspace }|]          =  HWBColorspace
           | x == fromIntegral [C.pure|int { (int)LabColorspace }|]          =  LabColorspace
           | x == fromIntegral [C.pure|int { (int)LCHColorspace }|]          =  LCHColorspace
           | x == fromIntegral [C.pure|int { (int)LCHabColorspace }|]        =  LCHabColorspace
           | x == fromIntegral [C.pure|int { (int)LCHuvColorspace }|]        =  LCHuvColorspace
           | x == fromIntegral [C.pure|int { (int)LogColorspace }|]          =  LogColorspace
           | x == fromIntegral [C.pure|int { (int)LMSColorspace }|]          =  LMSColorspace
           | x == fromIntegral [C.pure|int { (int)LuvColorspace }|]          =  LuvColorspace
           | x == fromIntegral [C.pure|int { (int)OHTAColorspace }|]         =  OHTAColorspace
           | x == fromIntegral [C.pure|int { (int)Rec601YCbCrColorspace }|]  =  Rec601YCbCrColorspace
           | x == fromIntegral [C.pure|int { (int)Rec709YCbCrColorspace }|]  =  Rec709YCbCrColorspace
           | x == fromIntegral [C.pure|int { (int)RGBColorspace }|]          =  RGBColorspace
           | x == fromIntegral [C.pure|int { (int)scRGBColorspace }|]        =  ScRGBColorspace
           | x == fromIntegral [C.pure|int { (int)sRGBColorspace }|]         =  SRGBColorspace
           | x == fromIntegral [C.pure|int { (int)TransparentColorspace }|]  =  TransparentColorspace
           | x == fromIntegral [C.pure|int { (int)xyYColorspace }|]          =  LOWERxyYColorspace
           | x == fromIntegral [C.pure|int { (int)XYZColorspace }|]          =  XYZColorspace
           | x == fromIntegral [C.pure|int { (int)YCbCrColorspace }|]        =  YCbCrColorspace
           | x == fromIntegral [C.pure|int { (int)YCCColorspace }|]          =  YCCColorspace
           | x == fromIntegral [C.pure|int { (int)YDbDrColorspace }|]        =  YDbDrColorspace
           | x == fromIntegral [C.pure|int { (int)YIQColorspace }|]          =  YIQColorspace
           | x == fromIntegral [C.pure|int { (int)YPbPrColorspace }|]        =  YPbPrColorspace
           | x == fromIntegral [C.pure|int { (int)YUVColorspace }|]          =  YUVColorspace
           | x == fromIntegral [C.pure|int { (int)LinearGRAYColorspace }|]   =  LinearGRAYColorspace
           | otherwise = error "Cannot convert AttributeKind to enum"
  fromEnum UndefinedColorspace   = fromIntegral [C.pure|int { (int)UndefinedColorspace }|]
  fromEnum CMYColorspace         = fromIntegral [C.pure|int { (int)CMYColorspace }|]
  fromEnum CMYKColorspace        = fromIntegral [C.pure|int { (int)CMYKColorspace }|]
  fromEnum GRAYColorspace        = fromIntegral [C.pure|int { (int)GRAYColorspace }|]
  fromEnum HCLColorspace         = fromIntegral [C.pure|int { (int)HCLColorspace }|]
  fromEnum HCLpColorspace        = fromIntegral [C.pure|int { (int)HCLpColorspace }|]
  fromEnum HSBColorspace         = fromIntegral [C.pure|int { (int)HSBColorspace }|]
  fromEnum HSIColorspace         = fromIntegral [C.pure|int { (int)HSIColorspace }|]
  fromEnum HSLColorspace         = fromIntegral [C.pure|int { (int)HSLColorspace }|]
  fromEnum HSVColorspace         = fromIntegral [C.pure|int { (int)HSVColorspace }|]
  fromEnum HWBColorspace         = fromIntegral [C.pure|int { (int)HWBColorspace }|]
  fromEnum LabColorspace         = fromIntegral [C.pure|int { (int)LabColorspace }|]
  fromEnum LCHColorspace         = fromIntegral [C.pure|int { (int)LCHColorspace }|]
  fromEnum LCHabColorspace       = fromIntegral [C.pure|int { (int)LCHabColorspace }|]
  fromEnum LCHuvColorspace       = fromIntegral [C.pure|int { (int)LCHuvColorspace }|]
  fromEnum LogColorspace         = fromIntegral [C.pure|int { (int)LogColorspace }|]
  fromEnum LMSColorspace         = fromIntegral [C.pure|int { (int)LMSColorspace }|]
  fromEnum LuvColorspace         = fromIntegral [C.pure|int { (int)LuvColorspace }|]
  fromEnum OHTAColorspace        = fromIntegral [C.pure|int { (int)OHTAColorspace }|]
  fromEnum Rec601YCbCrColorspace = fromIntegral [C.pure|int { (int)Rec601YCbCrColorspace }|]
  fromEnum Rec709YCbCrColorspace = fromIntegral [C.pure|int { (int)Rec709YCbCrColorspace }|]
  fromEnum RGBColorspace         = fromIntegral [C.pure|int { (int)RGBColorspace }|]
  fromEnum ScRGBColorspace       = fromIntegral [C.pure|int { (int)scRGBColorspace }|]
  fromEnum SRGBColorspace        = fromIntegral [C.pure|int { (int)sRGBColorspace }|]
  fromEnum TransparentColorspace = fromIntegral [C.pure|int { (int)TransparentColorspace }|]
  fromEnum LOWERxyYColorspace    = fromIntegral [C.pure|int { (int)xyYColorspace }|]
  fromEnum XYZColorspace         = fromIntegral [C.pure|int { (int)XYZColorspace }|]
  fromEnum YCbCrColorspace       = fromIntegral [C.pure|int { (int)YCbCrColorspace }|]
  fromEnum YCCColorspace         = fromIntegral [C.pure|int { (int)YCCColorspace }|]
  fromEnum YDbDrColorspace       = fromIntegral [C.pure|int { (int)YDbDrColorspace }|]
  fromEnum YIQColorspace         = fromIntegral [C.pure|int { (int)YIQColorspace }|]
  fromEnum YPbPrColorspace       = fromIntegral [C.pure|int { (int)YPbPrColorspace }|]
  fromEnum YUVColorspace         = fromIntegral [C.pure|int { (int)YUVColorspace }|]
  fromEnum LinearGRAYColorspace  = fromIntegral [C.pure|int { (int)LinearGRAYColorspace }|]
