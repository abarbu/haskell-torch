{-# LANGUAGE OverloadedStrings, QuasiQuotes, ScopedTypeVariables, TemplateHaskell #-}

-- | Minimal matio bindings. These are just enough to read the few types of mat
-- files that tend to contain datasets. More complete bindings will come one
-- day if they matter.

module Foreign.Matio where
import           Data.Coerce
import qualified Data.Map                         as Map
import           Data.Monoid                      (mempty, (<>))
import           Data.Text                        (Text)
import qualified Data.Text                        as T
import qualified Data.Text.Foreign                as T
import qualified Data.Vector                      as V'
import           Data.Vector.Storable             (Vector)
import qualified Data.Vector.Storable             as V
import           Foreign.C.String
import           Foreign.C.Types
import           Foreign.ForeignPtr
import           Foreign.ForeignPtr.Unsafe
import           Foreign.Marshal.Alloc
import           Foreign.Marshal.Array
import           Foreign.Matio.Types
import           Foreign.Ptr
import           Foreign.Storable
import qualified Language.C.Inline                as C
import qualified Language.C.Inline.Context        as C
import qualified Language.C.Inline.Cpp            as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Types                 as C

C.context matioCtx

C.include "<matio.h>"

C.verbatim "void delete_variable(matvar_t* v) { Mat_VarFree(v); }"
foreign import ccall "&delete_variable" deleteVariable :: FunPtr (Ptr CMatVar -> IO ())

C.verbatim "void delete_matfile(mat_t* mf) { Mat_Close(mf); }"
foreign import ccall "&delete_matfile" deleteMatfile :: FunPtr (Ptr CMat -> IO ())

openReadOnly :: Text -> IO (ForeignPtr CMat)
openReadOnly path =
  withCString (T.unpack path)
  (\path -> [C.exp|mat_t* { Mat_Open($(char *path), MAT_ACC_RDONLY) }|]
           >>= newForeignPtr deleteMatfile)

listVariables :: ForeignPtr CMat -> IO [Text]
listVariables m = do
  alloca (\np -> do
             vs <- [C.exp|char **{ Mat_GetDir($fptr-ptr:(mat_t *m), $(size_t *np)) }|]
             n <- fromIntegral <$> peek np
             r <- mapM (\i -> peekElemOff vs i >>= peekCString) [0..n-1]
             pure $ map T.pack $ r)

readRawVariable :: ForeignPtr CMat -> Text -> IO (Maybe (ForeignPtr CMatVar))
readRawVariable m name = do
  withCString (T.unpack name)
    (\name -> do
      mv <- [C.exp|matvar_t *{ Mat_VarRead($fptr-ptr:(mat_t *m), $(char *name)) }|]
      if mv == nullPtr then
        pure Nothing else
        Just <$> newForeignPtr deleteVariable mv)

readRawVariableInfo :: ForeignPtr CMat -> Text -> IO (Maybe (ForeignPtr CMatVar))
readRawVariableInfo m name = do
  withCString (T.unpack name)
    (\name -> do
      mv <- [C.exp|matvar_t *{ Mat_VarReadInfo($fptr-ptr:(mat_t *m), $(char *name)) }|]
      if mv == nullPtr then
        pure Nothing else
        Just <$> newForeignPtr deleteVariable mv)

readRawVariableData :: ForeignPtr CMat -> ForeignPtr CMatVar -> Ptr () -> CSize -> CSize -> CSize -> IO Bool
readRawVariableData m mv ptr start stride nrelements = do
  r <- [C.exp|int { Mat_VarReadDataLinear($fptr-ptr:(mat_t *m), $fptr-ptr:(matvar_t *mv), $(void *ptr),
                                         $(size_t start), $(size_t stride), $(size_t nrelements)) }|]
  pure $ r == 0

readVariable :: ForeignPtr CMat -> Text -> IO (Maybe MatValue)
readVariable m name = do
  r <- readRawVariable m name
  case r of
    Nothing -> pure Nothing
    Just v  -> Just <$> variableValue v

variableRank :: ForeignPtr CMatVar -> IO CInt
variableRank mv = [C.exp|int{$fptr-ptr:(matvar_t *mv)->rank}|]

variableDimensions :: ForeignPtr CMatVar -> IO (Vector CSize)
variableDimensions mv = do
  rank <- fromIntegral <$> [C.exp|int{$fptr-ptr:(matvar_t *mv)->rank}|]
  lenptr <- [C.exp|size_t *{$fptr-ptr:(matvar_t *mv)->dims}|]
  arr <- mallocArray rank
  copyArray arr lenptr rank
  arr' <- newForeignPtr finalizerFree arr
  pure $ V.unsafeFromForeignPtr0 arr' rank

variableValue :: ForeignPtr CMatVar -> IO MatValue
variableValue mv = do
  (ct :: MatClass) <- toEnum . fromIntegral <$> [C.exp|int { (int)$fptr-ptr:(matvar_t *mv)->class_type }|]
  (dt :: MatType) <- toEnum . fromIntegral <$> [C.exp|int { (int)$fptr-ptr:(matvar_t *mv)->data_type }|]
  rank <- fromIntegral <$> [C.exp|int{$fptr-ptr:(matvar_t *mv)->rank}|]
  lenptr <- [C.exp|size_t *{$fptr-ptr:(matvar_t *mv)->dims}|]
  lenptr' <- newForeignPtr_ lenptr
  let sz = V.unsafeFromForeignPtr0 lenptr' rank
  let len = fromIntegral $ V.product sz
  case (ct, dt) of
   (MatClassUInt8, MatTypeUInt8) -> do
     ptr <- [C.exp|unsigned char *{$fptr-ptr:(matvar_t *mv)->data}|]
     ptr' <- newForeignPtr finalizerFree =<< mallocArray len
     withForeignPtr ptr' (\ptr' -> copyArray ptr' ptr len)
     pure $ MatUInt8Array sz $ V.unsafeFromForeignPtr0 ptr' len
   (MatClassCell, MatTypeCell) -> do
     l <- mapM (\i -> do
             let i' = fromIntegral i
             mc <- [C.exp|matvar_t *{Mat_VarGetCell($fptr-ptr:(matvar_t *mv), $(int i'))}|] >>= newForeignPtr_
             variableValue mc) [0..len-1]
     pure $ MatCellArray sz (V'.fromList l)
   (MatClassChar, MatTypeUTF8) -> do
     ptr <- [C.exp|char *{$fptr-ptr:(matvar_t *mv)->data}|]
     MatText <$> T.peekCStringLen (ptr, len)
   _ -> error $ "Don't know how to convert these types from matlab " ++ show (ct, dt)

debugPrintVariable :: ForeignPtr CMatVar -> Bool -> IO ()
debugPrintVariable mvar printContents =
  let b = if printContents then 1 else 0
  in [C.exp|void { Mat_VarPrint($fptr-ptr:(matvar_t *mvar), $(int b)) }|]
