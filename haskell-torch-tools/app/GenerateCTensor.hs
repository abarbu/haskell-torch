{-# LANGUAGE FlexibleContexts, OverloadedLists, OverloadedStrings, QuasiQuotes, ScopedTypeVariables #-}

-- | This is run by the haskell-torch toplevel makefile in order to generate
-- raw bindings to the C tensor API.

module GenerateCTensor where
import           Control.Monad
import           Control.Monad.Extra
import qualified Data.Aeson               as A
import           Data.List
import qualified Data.Map                 as M
import           Data.String
import           Data.Text                (Text)
import qualified Data.Text                as T
import qualified Data.Text.IO             as T
import qualified Data.Text.Lazy           as TL
import           System.Console.Docopt
import           System.Directory
import           Data.Maybe
import           System.Environment       (getArgs)
import qualified Text.Mustache            as M
import           Text.Mustache.Compile.TH (mustache)

getArgOrExit :: Arguments -> Option -> IO String
getArgOrExit = getArgOrExitWith patterns

patterns :: Docopt
patterns = [docopt|
generate-ctensor

generate-ctensor processes torch/csrc/autograd/generated/VariableType.h
and fills out CTensor.hs. Note that this file is generated, it does not
exist in the upstream repo as is.

Usage:
  generate-ctensor <dest-file> <variable-type-header> [--verbose]
  generate-ctensor -h | --help
  generate-ctensor -v | --version

Options:
  -h --help             Show this screen
  --verbose             No compiler messages
  -v --version          Show version
|]

emit :: Bool -> Text -> Text -> Text -> [(Text, Text)] -> Text
emit generatePure retTy nameC nameHs args =
  fillTemplate' [mustache|
-- {{retTy}} {{nameC}} {{nameHs}}
-- {{inlineHs}}
{{&nameHs}} :: {{&argTysHs}}{{&monadHs}}{{&retTyHs}}
{{&nameHs}} {{&argNames}} = {{&unsafePerformIO}} {{&preHsFn}}
  [C.block|{{&retTyC}} {
    {{&retCFn}}VariableType::{{&nameC}}({{&argsC}}){{&retCFnEnd}};
   }{{&end}}{{&retHsFn}}
|]
    [("end","|]")
    ,("nameC", nameC)
    ,("nameHs", renameHs nameHs)
    ,("argTysHs", case T.intercalate " -> " $ map argTyHs $ map fst args of
                    "" -> ""
                    x  -> x <> " -> ")
    ,("retTyHs", let r = retTyHs retTy
                 in if T.isPrefixOf "(" r && T.isSuffixOf ")" r then
                      r else
                      "(" <> r <> ")")
    ,("argNames", T.unwords $ map (renameHs . snd) args)
    ,("preHsFn", T.intercalate "" $ map argPreFn args)
    ,("retTyC", if not isPure && (T.isSuffixOf "_" nameC || T.isSuffixOf "_out" nameC) then
                  "void" else
                  retTyC retTy)
    ,("argsC", T.intercalate ", " $ map argC args)
    ,("retCFn", if T.isSuffixOf "_" nameC || T.isSuffixOf "_out" nameC then
                  "" else
                  if needsAlloc retTy then
                    "return " <> (if isOpaqueC retTy then
                                    "(void*)" else
                                    "") <> "new "<> constructorC retTy <>"(" else
                    if needsCast retTy then
                      "return static_cast<" <> castTy retTy <> ">(" else
                      "return ")
    ,("retCFnEnd", if T.isSuffixOf "_" nameC || T.isSuffixOf "_out" nameC then
                     "" else
                     if needsAlloc retTy || needsCast retTy then
                       ")" else
                       "")
    ,("retHsFn", if not isPure && T.isSuffixOf "_" nameC && retTy /= "void" then
                   " >> pure self" else
                   if T.isSuffixOf "_out" nameC then
                     let r = rets retTy
                     in if r == 0 then
                          "" else
                          " >> pure ("<> (T.intercalate "," $ map (renameHs . snd) $ take r args) <> ")"
                   else
                     if retHsFn retTy == "" && needsAlloc retTy then
                       " >>= newForeignPtr " <> finalizerHs retTy else
                       retHsFn retTy)
    ,("monadHs", if isPure then "" else "IO ")
    ,("unsafePerformIO", if isPure then "unsafePerformIO $" else "")
    ,("inlineHs", if isPure then "{-# NOINLINE "<>renameHs nameHs<>" #-}" else "")
    ]
    where isPure = case (generatePure, args) of
                     (True, ((arg0, _):_)) -> (not (T.isSuffixOf "&" retTy)) && T.isPrefixOf "const " arg0
                     _                     -> False

fillTemplate :: Text -> [(Text, Text)] -> Text
fillTemplate m args =
  fillTemplate' (case M.compileMustacheText "fillTemplate" m of
                       Left x  -> error $ show x
                       Right x -> x) args

fillTemplate' :: M.Template -> [(Text, Text)] -> Text
fillTemplate' m args =
    TL.toStrict
  $ M.renderMustache m
  $ A.object (map (\(a,b) -> (a :: Text) A..= (b :: Text)) args)

splitTyAndName :: Text -> (Text, Text)
splitTyAndName = (\(a,b) -> (T.strip a, T.strip b)) . T.breakOnEnd " "

data MarshalRet = MarshalRet { _castType :: Maybe Text       -- ^ cast is needed from C to Hs
                             , _retTyHs :: Text        -- ^ Hs type when returning from C
                             , _retTyC :: Text         -- ^ C type when returning from C
                             , _needsAlloc :: Bool
                             , _constructorC :: Text
                             , _isOpaqueC :: Bool
                             , _retHsFn :: Text
                             , _finalizerHs :: Text }

marshalRet :: M.Map Text MarshalRet
marshalRet = M.fromList
  [("int64_t", MarshalRet { _castType = Nothing
                          , _retTyHs = "Int64"
                          , _retTyC = "int64_t"
                          , _needsAlloc = False
                          , _constructorC = ""
                          , _isOpaqueC = False
                          , _retHsFn = ""
                          , _finalizerHs = "" })
  ,("bool", MarshalRet { _castType = Nothing
                       , _retTyHs = "CBool"
                       , _retTyC = "bool"
                       , _needsAlloc = False
                       , _constructorC = ""
                       , _isOpaqueC = False
                       , _retHsFn = ""
                       , _finalizerHs = "" })
  ,("double", MarshalRet { _castType = Nothing
                         , _retTyHs = "CDouble"
                         , _retTyC = "double"
                         , _needsAlloc = False
                         , _constructorC = ""
                         , _isOpaqueC = False
                         , _retHsFn = ""
                         , _finalizerHs = "" })
  ,("Tensor &", MarshalRet { _castType = Nothing
                           , _retTyHs = "ForeignPtr CTensor"
                           , _retTyC = "Tensor*"
                           , _needsAlloc = True
                           , _constructorC = "Tensor"
                           , _isOpaqueC = False
                           , _retHsFn = ""
                           , _finalizerHs = "deleteTensor" })
  ,("Tensor", MarshalRet { _castType = Nothing
                         , _retTyHs = "ForeignPtr CTensor"
                         , _retTyC = "Tensor*"
                         , _needsAlloc = True
                         , _constructorC = "Tensor"
                         , _isOpaqueC = False
                         , _retHsFn = ""
                         , _finalizerHs = "deleteTensor" })
  ,("void*", MarshalRet { _castType = Nothing
                        , _retTyHs = "Ptr ()"
                        , _retTyC = "void*"
                        , _needsAlloc = False
                        , _constructorC = ""
                        , _isOpaqueC = False
                        , _retHsFn = ""
                        , _finalizerHs = "finalizerFree" })
  ,("void", MarshalRet { _castType = Nothing
                       , _retTyHs = "()"
                       , _retTyC = "void"
                       , _needsAlloc = False
                       , _constructorC = ""
                       , _isOpaqueC = False
                       , _retHsFn = ""
                       , _finalizerHs = "" })
  ,("Scalar", MarshalRet { _castType = Nothing
                         , _retTyHs = "ForeignPtr CScalar"
                         , _retTyC = "Scalar*"
                         , _needsAlloc = True
                         , _constructorC = "Scalar"
                         , _isOpaqueC = False
                         , _retHsFn = ""
                         , _finalizerHs = "deleteScalar'" })
  ,("std::tuple<Tensor,Tensor>", MarshalRet { _castType = Nothing
                                            , _retTyHs = "(ForeignPtr CTensor, ForeignPtr CTensor)"
                                            , _retTyC = "void*"
                                            , _needsAlloc = True
                                            , _constructorC = "std::tuple<Tensor,Tensor>"
                                            , _isOpaqueC = True
                                            , _retHsFn = " >>= unTupleTensorTensor"
                                            , _finalizerHs = "finalizerFree" })
  ,("std::tuple<Tensor,Tensor,Tensor>", MarshalRet { _castType = Nothing
                                                   , _retTyHs = "(ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)"
                                                   , _retTyC = "void*"
                                                   , _needsAlloc = True
                                                   , _constructorC = "std::tuple<Tensor,Tensor,Tensor>"
                                                   , _isOpaqueC = True
                                                   , _retHsFn = " >>= unTupleTensorTensorTensor"
                                                   , _finalizerHs = "finalizerFree" })
  ,("std::tuple<Tensor,Tensor,Tensor,Tensor>", MarshalRet { _castType = Nothing
                                                          , _retTyHs = "(ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)"
                                                          , _retTyC = "void*"
                                                          , _needsAlloc = True
                                                          , _constructorC = "std::tuple<Tensor,Tensor,Tensor,Tensor>"
                                                          , _isOpaqueC = True
                                                          , _retHsFn = " >>= unTupleTensorTensorTensorTensor"
                                                          , _finalizerHs = "finalizerFree" })
  ,("std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>", MarshalRet { _castType = Nothing
                                                                 , _retTyHs = "(ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)"
                                                                 , _retTyC = "void*"
                                                                 , _needsAlloc = True
                                                                 , _constructorC = "std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>"
                                                                 , _isOpaqueC = True
                                                                 , _retHsFn = " >>= unTupleTensorTensorTensorTensorTensor"
                                                                 , _finalizerHs = "finalizerFree" })
  ,("std::tuple<Tensor,Tensor,Tensor,int64_t>", MarshalRet { _castType = Nothing
                                                           , _retTyHs = "(ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, Int64)"
                                                           , _retTyC = "void*"
                                                           , _needsAlloc = True
                                                           , _constructorC = "std::tuple<Tensor,Tensor,Tensor,int64_t>"
                                                           , _isOpaqueC = True
                                                           , _retHsFn = " >>= unTupleTensorTensorTensorInt64"
                                                           , _finalizerHs = "finalizerFree" })
  ,("std::tuple<Tensor,Tensor,double,int64_t>", MarshalRet { _castType = Nothing
                                                           , _retTyHs = "(ForeignPtr CTensor, ForeignPtr CTensor, CDouble, Int64)"
                                                           , _retTyC = "void*"
                                                           , _needsAlloc = True
                                                           , _constructorC = "std::tuple<Tensor,Tensor,double,int64_t>"
                                                           , _isOpaqueC = True
                                                           , _retHsFn = " >>= unTupleTensorTensorDoubleInt64"
                                                           , _finalizerHs = "finalizerFree" })
  ,("std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t>", MarshalRet { _castType = Nothing
                                                                  , _retTyHs = "(ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, Int64)"
                                                                  , _retTyC = "void*"
                                                                  , _needsAlloc = True
                                                                  , _constructorC = "std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t>"
                                                                  , _isOpaqueC = True
                                                                  , _retHsFn = " >>= unTupleTensorTensorTensorTensorInt64"
                                                                  , _finalizerHs = "finalizerFree" })
  ,("std::vector<Tensor>", MarshalRet { _castType = Nothing
                                      , _retTyHs = "Vector (Ptr CTensor)"
                                      , _retTyC = "void*"
                                      , _needsAlloc = True
                                      , _constructorC = "std::vector<Tensor>"
                                      , _isOpaqueC = True
                                      , _retHsFn = " >>= unVectorTensor"
                                      , _finalizerHs = "finalizerFree" })
  ,("QScheme", MarshalRet { _castType = Just "uint8_t"
                          , _retTyHs = "Word8"
                          , _retTyC = "uint8_t"
                          , _needsAlloc = False
                          , _constructorC = ""
                          , _isOpaqueC = False
                          , _retHsFn = ""
                          , _finalizerHs = "" })
  ,("ScalarType", MarshalRet { _castType = Just "int8_t"
                             , _retTyHs = "Int8"
                             , _retTyC = "int8_t"
                             , _needsAlloc = False
                             , _constructorC = ""
                             , _isOpaqueC = False
                             , _retHsFn = ""
                             , _finalizerHs = "" })
  ]
  
data MarshalArg = MarshalArg { _argTyHs :: Text        -- ^ Hs type when this is passed as an argument
                             , _argTyC :: Text         -- ^ C type when passed as an argument to C
                             , _dereferenceC :: Text
                             , _argC :: Maybe (Text -> Text)
                             , _argPreFn :: Maybe (Text -> Text) }

marshalArg :: M.Map Text MarshalArg
marshalArg = M.fromList
  [("int64_t", MarshalArg { _argTyHs = "Int64"
                          , _argTyC = "int64_t"
                          , _dereferenceC = ""
                          , _argC = Nothing
                          , _argPreFn = Nothing })
  ,("c10::optional<int64_t>", MarshalArg { _argTyHs = "Maybe Int64"
                                         , _argTyC = "int64_t"
                                         , _dereferenceC = ""
                                         , _argC = Just $ \n -> "($(bool " <> renameHs n <> "__is_present) ? make_optional($(int64_t " <> renameHs n <> "__value)) : c10::nullopt)"
                                         , _argPreFn = Just $ \n -> "let ("<>n<>"__is_present, "<>n<>"__value) = splitMaybe "<>n<> " 0 in " })
  ,("bool", MarshalArg { _argTyHs = "CBool"
                       , _argTyC = "bool"
                       , _dereferenceC = ""
                       , _argC = Nothing
                       , _argPreFn = Nothing })
  ,("c10::optional<bool>", MarshalArg { _argTyHs = "CBool"
                                      , _argTyC = "bool"
                                      , _dereferenceC = ""
                                      , _argC = Nothing
                                      , _argPreFn = Nothing })
  ,("double", MarshalArg { _argTyHs = "CDouble"
                         , _argTyC = "double"
                         , _dereferenceC = ""
                         , _argC = Nothing
                         , _argPreFn = Nothing })
  ,("c10::optional<double>", MarshalArg { _argTyHs = "Maybe CDouble"
                                        , _argTyC = "double"
                                        , _dereferenceC = ""
                                        , _argC = Just $ \n -> "($(bool " <> renameHs n <> "__is_present) ? make_optional($(double " <> renameHs n <> "__value)) : c10::nullopt)"
                                        , _argPreFn = Just $ \n -> "let ("<>n<>"__is_present, "<>n<>"__value) = splitMaybe "<>n<> " 0 in " })
  ,("Tensor &", MarshalArg { _argTyHs = "ForeignPtr CTensor"
                           , _argTyC = "Tensor*"
                           , _dereferenceC = "*"
                           , _argC = Just $ \n -> "*$fptr-ptr:(Tensor* " <> renameHs n <> ")"
                           , _argPreFn = Nothing })
  ,("const Tensor &", MarshalArg { _argTyHs = "ForeignPtr CTensor"
                                 , _argTyC = "Tensor*"
                                 , _dereferenceC = "*"
                                 , _argC = Just $ \n -> "*$fptr-ptr:(Tensor* " <> renameHs n <> ")"
                                 , _argPreFn = Nothing })
  ,("Tensor", MarshalArg { _argTyHs = "ForeignPtr CTensor"
                         , _argTyC = "Tensor*"
                         , _dereferenceC = "*"
                         , _argC = Just $ \n -> "*$fptr-ptr:(Tensor* " <> renameHs n <> ")"
                         , _argPreFn = Nothing  })
  ,("const TensorOptions &", MarshalArg { _argTyHs = "ForeignPtr CTensorOptions"
                                        , _argTyC = "TensorOptions*"
                                        , _dereferenceC = "*"
                                        , _argC = Just $ \n -> "*$fptr-ptr:(TensorOptions* " <> renameHs n <> ")"
                                        , _argPreFn = Nothing })
  ,("Scalar", MarshalArg { _argTyHs = "ForeignPtr CScalar"
                         , _argTyC = "Scalar*"
                         , _dereferenceC = "*"
                         , _argC = Just $ \n -> "*$fptr-ptr:(Scalar* " <> renameHs n <> ")"
                         , _argPreFn = Nothing })
  ,("c10::optional<Scalar>", MarshalArg { _argTyHs = "ForeignPtr CScalar"
                                        , _argTyC = "Scalar*"
                                        , _dereferenceC = "*"
                                        , _argC = Just $ \n -> "*$fptr-ptr:(Scalar* " <> renameHs n <> ")"
                                        , _argPreFn = Nothing })
  ,("ScalarType", MarshalArg { _argTyHs = "Int8"
                             , _argTyC = "int8_t"
                             , _dereferenceC = ""
                             , _argC = Just $ \n -> "static_cast<ScalarType>($(int8_t " <> renameHs n <> "))"
                             , _argPreFn = Nothing })
  ,("c10::optional<ScalarType>", MarshalArg { _argTyHs = "Int8"
                                            , _argTyC = "int8_t"
                                            , _dereferenceC = ""
                                            , _argC = Just $ \n -> "static_cast<ScalarType>($(int8_t " <> renameHs n <> "))"
                                            , _argPreFn = Nothing })
  ,("MemoryFormat", MarshalArg { _argTyHs = "Int8"
                               , _argTyC = "int8_t"
                               , _dereferenceC = ""
                               , _argC = Just $ \n -> "static_cast<MemoryFormat>($(int8_t " <> renameHs n <> "))"
                               , _argPreFn = Nothing })
  ,("c10::optional<MemoryFormat>", MarshalArg { _argTyHs = "Int8"
                                              , _argTyC = "int8_t"
                                              , _dereferenceC = ""
                                              , _argC = Just $ \n -> "static_cast<MemoryFormat>($(int8_t " <> renameHs n <> "))"
                                              , _argPreFn = Nothing })
  ,("std::array<bool,2>", MarshalArg { _argTyHs = "Vector CBool"
                                     , _argTyC = "std::array<bool,2>"
                                     , _dereferenceC = ""
                                     , _argC = Just $ \n -> "make_array_bool_2($(bool* " <> renameHs n <> "__array))"
                                     , _argPreFn = Just $ \n -> "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in " })
  ,("std::array<bool,3>", MarshalArg { _argTyHs = "Vector CBool"
                                     , _argTyC = "std::array<bool,3>"
                                     , _dereferenceC = ""
                                     , _argC = Just $ \n -> "make_array_bool_3($(bool* " <> renameHs n <> "__array))"
                                     , _argPreFn = Just $ \n -> "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in " })
  ,("std::array<bool,4>", MarshalArg { _argTyHs = "Vector CBool"
                                     , _argTyC = "std::array<bool,4>"
                                     , _dereferenceC = ""
                                     , _argC = Just $ \n -> "make_array_bool_4($(bool* " <> renameHs n <> "__array))"
                                     , _argPreFn = Just $ \n -> "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in " })
  ,("IntArrayRef", MarshalArg { _argTyHs = "Vector Int64"
                              , _argTyC = "IntArrayRef"
                              , _dereferenceC = ""
                              , _argC = Just $ \n -> "ArrayRef<int64_t>($(int64_t* " <> renameHs n <> "__array), $(size_t "<> n <>"__size))"
                              , _argPreFn = Just $ \n -> "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in "})
  ,("TensorList", MarshalArg { _argTyHs = "Vector (Ptr CTensor)"
                             , _argTyC = "TensorList"
                             , _dereferenceC = ""
                             , _argC = Just $ \n -> "pack_tensor_list($(Tensor** " <> renameHs n <> "__array), $(size_t "<> n <>"__size))"
                             , _argPreFn = Just $ \n -> "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in " })
  ,("Generator *", MarshalArg { _argTyHs = "Ptr CGenerator"
                              , _argTyC = "Generator*"
                              , _dereferenceC = ""
                              , _argC = Nothing
                              , _argPreFn = Nothing })
  ,("std::string", MarshalArg { _argTyHs = "Ptr CChar"
                              , _argTyC = "char*"
                              , _dereferenceC = ""
                              , _argC = Nothing
                              , _argPreFn = Nothing })
  ,("Storage &", MarshalArg { _argTyHs = "Ptr CStorage"
                            , _argTyC = "Storage*"
                            , _dereferenceC = "*"
                            , _argC = Nothing
                            , _argPreFn = Nothing })
  ,("Storage", MarshalArg { _argTyHs = "Ptr CStorage"
                          , _argTyC = "Storage*"
                          , _dereferenceC = "*"
                          , _argC = Nothing
                          , _argPreFn = Nothing })
  ,("Device", MarshalArg { _argTyHs = "Ptr CDevice"
                         , _argTyC = "Device*"
                         , _dereferenceC = "*"
                         , _argC = Nothing
                         , _argPreFn = Nothing })
  ]

marshal from field err ty = case M.lookup ty from of
                              Nothing -> error (err ++ " doesn't know what to do with type " ++ show ty)
                              Just x -> field x

needsCast :: Text -> Bool
needsCast = isJust . marshal marshalRet _castType "needsCast"

castTy :: Text -> Text
castTy = fromJust . marshal marshalRet _castType "castTy"

-- What Haskell type should this return value have?
retTyHs :: Text -> Text
retTyHs x = case M.lookup x marshalRet of
              Just info -> _retTyHs info
              Nothing ->
                case T.splitOn "," <$> (T.stripPrefix "std::tuple<" =<< T.stripSuffix ">" x) of
                  Just r  -> "(" <> T.intercalate ", " (map retTyHs r) <> ")"
                  Nothing -> error $ "Don't know how to retTyHs this: " ++ show x

rets :: Text -> Int
rets "void" = 0
rets retTy =
  case T.splitOn "," <$> (T.stripPrefix "std::tuple<" =<< T.stripSuffix ">" retTy) of
    Nothing -> 1
    Just l  -> length l

-- When marshaling values what type is Haskell going to give C?
retTyC :: Text -> Text
retTyC = marshal marshalRet _retTyC "retTyC"

-- Do we need to alocate memory in order to marshal this type?
needsAlloc :: Text -> Bool
needsAlloc = marshal marshalRet _needsAlloc "needsAlloc"

-- How do we build this tpye in C?
constructorC :: Text -> Text
constructorC = marshal marshalRet _constructorC "constructorC"

isOpaqueC :: Text -> Bool
isOpaqueC = marshal marshalRet _isOpaqueC "isOpaqueC"

-- What function will marshal this from C to Hs
retHsFn :: Text -> Text
retHsFn = marshal marshalRet _retHsFn "retHsFn"

-- How will we free the memory in Haskell?
finalizerHs :: Text -> Text
finalizerHs = marshal marshalRet _finalizerHs "finalizerHs"

-- What type does this argument have in Haskell?
argTyHs :: Text -> Text
argTyHs = marshal marshalArg _argTyHs "argTyHs"

argTyC :: Text -> Text
argTyC = marshal marshalArg _argTyC "argTyC"

-- If this is a pointer, how do we access its memory?
dereferenceC :: Text -> Text
dereferenceC = marshal marshalArg _dereferenceC "dereferenceC"

argC :: (Text, Text) -> Text
argC (ty, n) = case marshal marshalArg _argC "argC" ty of
                 Just fn -> fn n
                 Nothing -> dereferenceC ty <> "$(" <> argTyC ty <> " " <> renameHs n <> ")"

argPreFn :: (Text, Text) -> Text
argPreFn (ty, n) = case marshal marshalArg _argPreFn "argPreFn" ty of
                     Just fn -> fn n
                     Nothing -> ""

renameHs :: Text -> Text
renameHs "where" = "whereX"
renameHs "data"  = "dataX"
renameHs "type"  = "typeX"
renameHs "in"    = "inX"
renameHs x       = T.toLower x

main :: IO ()
main = do
  args <- parseArgsOrExit patterns =<< getArgs
  destFile            <- T.pack <$> args `getArgOrExit` argument "dest-file"
  variableTypeHeader  <- T.pack <$> args `getArgOrExit` argument "variable-type-header"
  unlessM (doesFileExist $ T.unpack destFile) (exitWithUsageMessage patterns $ "Destination file doesn't exist! " ++ show destFile)
  unlessM (doesFileExist $ T.unpack variableTypeHeader) (exitWithUsageMessage patterns $ "Header doesn't exist! " ++ show variableTypeHeader)
  fin <- T.readFile $ T.unpack variableTypeHeader
  let os =
        filter (\(ty, name, args) -> (not (T.isInfixOf "_forward" name)) && (not (T.isInfixOf "_backward" name)))
        $ map (\x ->
         case T.splitOn "(" $ T.replace "virtual " "" x of
           [pre,args'] ->
             let args = case T.splitOn ", " $ fst $ T.breakOn ")" args' of
                          [""] -> []
                          x    -> x
                 (ty,name) = splitTyAndName pre
             in (ty, name, args))
        $ filter (not . T.isInfixOf "ConstQuantizerPtr") -- NB We don't yet support quantization
        $ filter (not . T.isInfixOf "Dimname") -- NB We don't support named dimensions
        $ filter (T.isInfixOf "{")
        $ T.lines
        $ fst
        $ T.breakOn "namespace"
        $ snd
        $ T.breakOn "Tensor __and__" (T.replace "  static " "" fin)
  let ls = snd $ mapAccumL (\m (ty, name, args) ->
                              (M.alter (\x -> Just $ case x of
                                           Nothing -> 1
                                           Just n  -> n + 1)
                                name
                                m
                              ,emit False ty name
                                (case M.lookup name m of
                                    Nothing -> name
                                    Just n  -> name <> "__" <> T.pack (show n))
                                (map splitTyAndName args)))
                M.empty os
  when (args `isPresent` longOption "verbose") $ mapM_ T.putStrLn ls
  dest <- T.readFile $ T.unpack destFile
  case T.breakOnEnd "-- Everything below is AUTOGENERATED from generate-ctensor" dest of
    (pre, _) -> T.writeFile (T.unpack destFile) (pre <> T.unlines ls)
  T.putStrLn $ "Updated " <> destFile
  pure ()
