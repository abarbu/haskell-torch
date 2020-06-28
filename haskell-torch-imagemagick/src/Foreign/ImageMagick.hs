{-# LANGUAGE OverloadedStrings, QuasiQuotes, ScopedTypeVariables, TemplateHaskell, FlexibleContexts #-}

-- | Minimal matio bindings. These are just enough to read the few types of mat
-- files that tend to contain datasets.

module Foreign.ImageMagick(module Foreign.ImageMagick, module Foreign.ImageMagick.Types) where
import qualified Data.ByteString                  as BS
import qualified Data.ByteString.Unsafe           as BS
import           Data.Coerce
import qualified Data.Map                         as Map
import           Data.Monoid                      (mempty, (<>))
import           Data.Text                        (Text)
import qualified Data.Text                        as T
import qualified Data.Text.Encoding               as T
import qualified Data.Text.Foreign                as T
import qualified Data.Vector                      as V'
import           Data.Vector.Storable             (Vector)
import qualified Data.Vector.Storable             as V
import           Data.Word
import           Foreign.C.String
import           Foreign.C.Types
import           Foreign.ForeignPtr
import           Foreign.ForeignPtr.Unsafe
import           Foreign.ImageMagick.Types
import           Foreign.Marshal.Alloc
import           Foreign.Marshal.Alloc
import           Foreign.Marshal.Array
import           Foreign.Marshal.Utils
import           Foreign.Ptr
import           Foreign.Storable
import qualified Language.C.Inline                as C
import qualified Language.C.Inline.Context        as C
import qualified Language.C.Inline.Cpp            as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Types                 as C

C.context imageMagickCtx

C.verbatim "#define MAGICKCORE_QUANTUM_DEPTH 16"
C.include "<MagickWand/MagickWand.h>"

C.verbatim "void delete_magickwand(MagickWand *w) { DestroyMagickWand(w); }"
foreign import ccall "&delete_magickwand" deleteMagickWand :: FunPtr (Ptr CMagickWand -> IO ())

C.verbatim "void magick_relinquish_memory(void *p) { MagickRelinquishMemory(p); }"
foreign import ccall "&magick_relinquish_memory" magickRelinquishMemory :: FunPtr (Ptr () -> IO ())

-- | You must call this before using any functions from this library!
-- Calling it multiple times is fine
initialize :: IO ()
initialize = [C.exp|void { MagickWandGenesis() }|]

-- | This is an internal function to handle errors, you don't need it.
handleWandErrors :: ForeignPtr CMagickWand -> IO CInt -> IO Bool
handleWandErrors w f = do
  -- TODO
  -- description=MagickGetException(wand,&severity); \
  -- (void) fprintf(stderr,"%s %s %lu %s\n",GetMagickModule(),description); \
  -- description=(char *) MagickRelinquishMemory(description); \
  i <- f
  if i == [C.pure|int { MagickFalse } |] then do
    [C.block|int {
        ExceptionType ty;
        char *description = MagickGetException($fptr-ptr:(MagickWand *w),&ty);
        printf("Imagemagick exception! %s\n", description);
    }|]
    pure False else
    pure True

newWand :: IO (ForeignPtr CMagickWand)
newWand = [C.exp|MagickWand *{ NewMagickWand() }|] >>= newForeignPtr deleteMagickWand

cloneWand :: ForeignPtr CMagickWand -> IO (ForeignPtr CMagickWand)
cloneWand w = [C.exp|MagickWand *{ CloneMagickWand($fptr-ptr:(MagickWand *w)) }|]
  >>= newForeignPtr deleteMagickWand

-- | Read an image into the Wand.
readImage :: ForeignPtr CMagickWand -> Text -> IO Bool
readImage w path =
  withCString (T.unpack path)
    (\path ->
       handleWandErrors w
       [C.exp|int { MagickReadImage($fptr-ptr:(MagickWand *w), $(char *path)) }|])

-- | Read an image from the Wand.
writeImage :: ForeignPtr CMagickWand -> Text -> IO Bool
writeImage w path =
  withCString (T.unpack path)
    (\path ->
       handleWandErrors w
       [C.exp|int { MagickWriteImage($fptr-ptr:(MagickWand *w), $(char *path)) }|])

-- |
writeImageToBuffer :: ForeignPtr CMagickWand -> ImageFormat -> IO (V.Vector Word8)
writeImageToBuffer w format =
  alloca (\(lenPtr::Ptr CSize) -> do
             -- TODO Check return
             withFormat format
               (\formatPtr ->
                  [C.exp|void {MagickSetImageFormat($fptr-ptr:(MagickWand *w), $(char *formatPtr))}|])
             -- TODO Check return
             ptr <- [C.exp|void* { MagickGetImageBlob($fptr-ptr:(MagickWand *w), $(size_t *lenPtr)) }|]
             len <- peek lenPtr
             fptr <- newForeignPtr magickRelinquishMemory ptr
             pure $ V.unsafeFromForeignPtr0 (coerce fptr) (fromIntegral len))

imageWidth :: ForeignPtr CMagickWand -> IO CInt
imageWidth w = [C.exp|int { MagickGetImageWidth($fptr-ptr:(MagickWand *w)) }|]

imageHeight :: ForeignPtr CMagickWand -> IO CInt
imageHeight w = [C.exp|int { MagickGetImageHeight($fptr-ptr:(MagickWand *w)) }|]

imageDepth :: ForeignPtr CMagickWand -> IO CInt
imageDepth w = [C.block|int {
                   ColorspaceType ty = MagickGetImageColorspace($fptr-ptr:(MagickWand *w));
                   switch(ty) {
                      case UndefinedColorspace:
                           return -1;
                      case GRAYColorspace:
                      case LinearGRAYColorspace:
                      case LogColorspace:
                      case TransparentColorspace:
                           return 1;
                      case CMYColorspace:
                      case HCLColorspace:
                      case HCLpColorspace:
                      case HSBColorspace:
                      case HSIColorspace:
                      case HSLColorspace:
                      case HSVColorspace:
                      case HWBColorspace:
                      case LabColorspace:
                      case LCHColorspace:
                      case LCHabColorspace:
                      case LCHuvColorspace:
                      case LMSColorspace:
                      case LuvColorspace:
                      case Rec601YCbCrColorspace:
                      case Rec709YCbCrColorspace:
                      case RGBColorspace:
                      case scRGBColorspace:
                      case sRGBColorspace:
                      case xyYColorspace:
                      case XYZColorspace:
                      case YCbCrColorspace:
                      case YCCColorspace:
                      case YDbDrColorspace:
                      case YIQColorspace:
                      case YPbPrColorspace:
                      case YUVColorspace:
                           return 3;
                      case CMYKColorspace:
                      case OHTAColorspace:
                           return 4;
                     default:
                           return -1;
                   }}|]

perspective w topleft topright botright botleft =
  [C.block|int {
      double args[] = {$(double tlx),$(double tly),$(double tlx'),$(double tly'),
                       $(double trx),$(double try),$(double trx'),$(double try'),
                       $(double brx),$(double bry),$(double brx'),$(double bry'),
                       $(double blx),$(double bly),$(double blx'),$(double bly')};
      return MagickDistortImage($fptr-ptr:(MagickWand *w), PerspectiveDistortion, 4*4, args, 0); }|]
  where ((tlx,tly),(tlx',tly')) = topleft
        ((trx,try),(trx',try')) = topright
        ((brx,bry),(brx',bry')) = botright
        ((blx,bly),(blx',bly')) = botleft

crop w width height x y =
  [C.exp|int { MagickCropImage($fptr-ptr:(MagickWand *w), $(int width), $(int height), $(int x), $(int y)) }|]

-- MagickFlipImage() creates a vertical mirror image by reflecting the pixels around the central x-axis.
flip w = [C.exp|int { MagickFlipImage($fptr-ptr:(MagickWand *w)) }|]

-- MagickFlopImage() creates a horizontal mirror image by reflecting the pixels around the central y-axis.
flop w = [C.exp|int { MagickFlopImage($fptr-ptr:(MagickWand *w)) }|]

brightnessContrast w b c = [C.exp|int { MagickBrightnessContrastImage($fptr-ptr:(MagickWand *w), $(double b), $(double c)) }|]

modulate w brightness saturation hue = [C.exp|int { MagickModulateImage($fptr-ptr:(MagickWand *w), $(double brightness), $(double saturation), $(double hue)) }|]

getColorspace :: ForeignPtr CMagickWand -> IO ColorSpace
getColorspace w = toEnum . fromIntegral <$> [C.exp|int { MagickGetImageColorspace($fptr-ptr:(MagickWand *w)) }|]

convertToColorspace w col =
  [C.exp|int { MagickTransformImageColorspace($fptr-ptr:(MagickWand *w), $(int c)) }|]
  where c = fromIntegral $ fromEnum col

rotate w degrees r g b = [C.block|int {
                             PixelWand *pwand = NewPixelWand();
                             PixelSetRed(pwand, $(double r));
                             PixelSetGreen(pwand, $(double g));
                             PixelSetBlue(pwand, $(double b));
                             int out = MagickRotateImage($fptr-ptr:(MagickWand *w), pwand, $(double degrees));
                             DestroyPixelWand(pwand);
                             return out; }|]

pad w width height r g b = [C.block|int {
                             PixelWand *pwand = NewPixelWand();
                             PixelSetRed(pwand, $(double r));
                             PixelSetGreen(pwand, $(double g));
                             PixelSetBlue(pwand, $(double b));
                             int out = MagickBorderImage($fptr-ptr:(MagickWand *w), pwand, $(int width), $(int height), OverCompositeOp);
                             DestroyPixelWand(pwand);
                             return out; }|]

resizeBilinear w height width = [C.exp|int { MagickInterpolativeResizeImage($fptr-ptr:(MagickWand *w), $(int height), $(int width), BilinearInterpolatePixel) }|]
resizeNearest w height width = [C.exp|int { MagickInterpolativeResizeImage($fptr-ptr:(MagickWand *w), $(int height), $(int width), NearestInterpolatePixel) }|]


affine w transx transy rotatex rotatey shearx sheary = [C.block|int {
                             DrawingWand *dwand = NewDrawingWand();
                             AffineMatrix m;
                             m.sx = $(double shearx);
                             m.sy = $(double sheary);
                             m.rx = $(double rotatex);
                             m.ry = $(double rotatey);
                             m.tx = $(double transx);
                             m.ty = $(double transy);
                             DrawAffine(dwand, &m);
                             int out = MagickAffineTransformImage($fptr-ptr:(MagickWand *w), dwand);
                             DestroyDrawingWand(dwand);
                             return out; }|]

-- | Write the image from the wand into the buffer
writeImagePixelsToPtr :: ForeignPtr CMagickWand -> PixelOrder -> StorageType -> Ptr () -> IO Bool
writeImagePixelsToPtr w po storage ptr =
  withPixelOrder po
  (\po ->
      handleWandErrors w
      [C.block|int {
          return MagickExportImagePixels($fptr-ptr:(MagickWand *w),
                                  0, 0,
                                  MagickGetImageWidth($fptr-ptr:(MagickWand *w)),
                                  MagickGetImageHeight($fptr-ptr:(MagickWand *w)),
                                  $(char *po),  $(int storage'),
                                  $(void *ptr)); }|])
  where storage' = fromIntegral $ fromEnum storage

-- | Create a new image in the wand
createImageFromPtr :: ForeignPtr CMagickWand -> PixelOrder -> StorageType -> CInt -> CInt -> Ptr () -> IO Bool
createImageFromPtr w po storage height width ptr =
  withPixelOrder po
  (\po ->
      handleWandErrors w
      [C.block|int {
          PixelWand *pwand = NewPixelWand();
          PixelSetRed(pwand, 0);
          PixelSetGreen(pwand, 0);
          PixelSetBlue(pwand, 0);
          MagickNewImage($fptr-ptr:(MagickWand *w), $(int height), $(int width), pwand);
          DestroyPixelWand(pwand);
          return MagickImportImagePixels($fptr-ptr:(MagickWand *w),
                                         0, 0,
                                         $(int height), $(int width),
                                         $(char *po), $(int storage'),
                                         $(void *ptr));
       }|])
  where storage' = fromIntegral $ fromEnum storage

-- | Read the buffer into the wand
readImagePixelsFromPtr :: ForeignPtr CMagickWand -> PixelOrder -> StorageType -> Ptr () -> IO Bool
readImagePixelsFromPtr w po storage ptr =
  withPixelOrder po
  (\po ->
      handleWandErrors w
      [C.exp|int {
          MagickImportImagePixels($fptr-ptr:(MagickWand *w),
                                  0, 0,
                                  MagickGetImageWidth($fptr-ptr:(MagickWand *w)),
                                  MagickGetImageHeight($fptr-ptr:(MagickWand *w)),
                                  $(char *po), $(int storage'),
                                  $(void *ptr)) }|])
  where storage' = fromIntegral $ fromEnum storage

-- | Write the image from the wand into the buffer
writeImagePixelsTo :: ForeignPtr CMagickWand -> PixelOrder -> StorageType -> Ptr () -> IO Bool
writeImagePixelsTo w po storage ptr =
  withPixelOrder po
  (\po ->
      handleWandErrors w
      [C.exp|int {
          MagickExportImagePixels($fptr-ptr:(MagickWand *w),
                                  0, 0,
                                  MagickGetImageWidth($fptr-ptr:(MagickWand *w)),
                                  MagickGetImageHeight($fptr-ptr:(MagickWand *w)),
                                  $(char *po), $(int storage'),
                                  $(void *ptr)) }|])
  where storage' = fromIntegral $ fromEnum storage

-- * Pixel properties

withPixelWand f = do
  ptr <- [C.exp|PixelWand *{ NewPixelWand() }|]
  r <- f ptr
  [C.exp|void { DestroyPixelWand($(PixelWand *ptr)) }|]
  pure r

setR pwand r = [C.exp|void { PixelSetRed($(PixelWand *pwand), $(double r)); }|]
setG pwand g = [C.exp|void { PixelSetGreen($(PixelWand *pwand), $(double g)); }|]
setB pwand b = [C.exp|void { PixelSetBlue($(PixelWand *pwand), $(double b)); }|]
setA pwand a = [C.exp|void { PixelSetAlpha($(PixelWand *pwand), $(double a)); }|]

setColor pwand colorStr =
  withCString colorStr (\cptr -> [C.exp|int { PixelSetColor($(PixelWand *pwand), $(char *cptr)) }|])

-- * Drawing on images

withDrawingWand f = do
  ptr <- [C.exp|DrawingWand *{ NewDrawingWand() }|]
  r <- f ptr
  [C.exp|void { DestroyDrawingWand($(DrawingWand *ptr)) }|]
  pure r

setStrokeColor dwand pwand =
  [C.exp|void { DrawSetStrokeColor($(DrawingWand *dwand), $(PixelWand *pwand)) }|]

setBorderColor dwand pwand =
  [C.exp|void { DrawSetBorderColor($(DrawingWand *dwand), $(PixelWand *pwand)) }|]

setFillColor dwand pwand =
  [C.exp|void { DrawSetFillColor($(DrawingWand *dwand), $(PixelWand *pwand)) }|]

setTextUnderColor dwand pwand =
  [C.exp|void { DrawSetTextUnderColor($(DrawingWand *dwand), $(PixelWand *pwand)) }|]

setFillOpacity dwand alpha =
  [C.exp|void { DrawSetFillOpacity($(DrawingWand *dwand), $(double alpha)) }|]

setOpacity dwand alpha =
  [C.exp|void { DrawSetOpacity($(DrawingWand *dwand), $(double alpha)) }|]

setStrokeWidth dwand alpha =
  [C.exp|void { DrawSetStrokeWidth($(DrawingWand *dwand), $(double alpha)) }|]

setStrokeAntialias dwand b =
  [C.exp|void { DrawSetStrokeAntialias($(DrawingWand *dwand), $(int b)) }|]

drawLine dwand sx sy ex ey =
  [C.exp|void { DrawLine($(DrawingWand *dwand), $(double sx), $(double sy), $(double ex), $(double ey)) }|]

drawRectangle dwand x1 y1 x2 y2 =
  [C.exp|void { DrawRectangle($(DrawingWand *dwand), $(double x1), $(double y1), $(double x2), $(double y2)) }|]

drawPoint dwand x y =
  [C.exp|void { DrawPoint($(DrawingWand *dwand), $(double x), $(double y)) }|]

drawRoundRectangle dwand x1 y1 x2 y2 rx ry =
  [C.exp|void { DrawRoundRectangle($(DrawingWand *dwand), $(double x1), $(double y1), $(double x2), $(double y2), $(double rx), $(double ry)) }|]

drawArc dwand sx sy ex ey sd ed =
  [C.exp|void { DrawArc($(DrawingWand *dwand), $(double sx), $(double sy), $(double ex), $(double ey), $(double sd), $(double ed)) }|]

drawCircle dwand ox oy px py =
  [C.exp|void { DrawCircle($(DrawingWand *dwand), $(double ox), $(double oy), $(double px), $(double py)) }|]

drawAnnotation dwand x y colorStr =
  withCString colorStr (\cptr -> [C.exp|void { DrawAnnotation($(DrawingWand *dwand), $(double x), $(double y), $(char *cptr)) }|])

setFontSize dwand pointsz =
  [C.exp|void { DrawSetFontSize($(DrawingWand *dwand), $(double pointsz)) }|]
