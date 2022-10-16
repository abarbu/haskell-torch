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
  generate-ctensor <dest-file> <functions-header> <tensor-body-header> [--verbose]
  generate-ctensor -h | --help
  generate-ctensor -v | --version

Options:
  -h --help             Show this screen
  --verbose             No compiler messages
  -v --version          Show version
|]
  
emit :: Bool -> Bool -> Text -> Text -> Text -> [(Text, Text)] -> Text
emit generatePure member retTy nameC nameHs args =
  fillTemplate' [mustache|
-- {{&retTy}} {{&nameC}}({{&allArgs}})
--{{&inlineHs}}
{{&nameHs}} :: {{&argTysHs}}{{&monadHs}}{{&retTyHs}}
{{&nameHs}} {{&argNames}} = {{&unsafePerformIO}} {{&preHsFn}}
  [C.block|{{&retTyC}} {
    {{&retCFn}}{{&prefix}}{{&nameC}}({{&argsC}}){{&retCFnEnd}};
   }{{&end}}{{&retHsFn}}
|]
    [("end","|]")
    ,("prefix", if member then "$fptr-ptr:(Tensor *self)->" else "at::")
    ,("retTy", retTy)
    ,("allArgs", T.intercalate ", " (map (\(a,b) -> a <> " " <> b) args))
    ,("nameC", nameC)
    ,("nameHs", renameHs nameHs)
    ,("argTysHs", case T.intercalate " -> " $ map argTyHs $ map fst $ (if member then [("Tensor *", "self")] else []) ++ args of
                    "" -> ""
                    x  -> x <> " -> ")
    ,("retTyHs", let r = retTyHs retTy
                 in if T.isPrefixOf "(" r && T.isSuffixOf ")" r then
                      r else
                      "(" <> r <> ")")
    ,("argNames", T.unwords $ map (renameHs . snd) $ (if member then [("Tensor *", "self")] else []) ++ args)
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
    ,("inlineHs", if isPure then " {-# NOINLINE "<>renameHs nameHs<>" #-}" else "")
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
  ,("size_t", MarshalRet { _castType = Nothing
                          , _retTyHs = "CSize"
                          , _retTyC = "size_t"
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
  ,("Tensor&", MarshalRet { _castType = Nothing
                          , _retTyHs = "ForeignPtr CTensor"
                          , _retTyC = "Tensor*"
                          , _needsAlloc = True
                          , _constructorC = "Tensor"
                          , _isOpaqueC = False
                          , _retHsFn = ""
                          , _finalizerHs = "deleteTensor" })
  ,("const Tensor&", MarshalRet { _castType = Nothing
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
  ,("std::tuple<Tensor &,Tensor &>", MarshalRet { _castType = Nothing
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
  ,("std::tuple<Tensor &,Tensor &,Tensor &>", MarshalRet { _castType = Nothing
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
  ,("std::tuple<double,int64_t>", MarshalRet { _castType = Nothing
                                             , _retTyHs = "(CDouble, Int64)"
                                             , _retTyC = "void*"
                                             , _needsAlloc = True
                                             , _constructorC = "std::tuple<double,int64_t>"
                                             , _isOpaqueC = True
                                             , _retHsFn = " >>= unTupleDoubleInt64"
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
  ,("IntArrayRef", MarshalRet { _castType = Nothing
                              , _retTyHs = "Vector Int64"
                              , _retTyC = "void*"
                              , _needsAlloc = True
                              , _constructorC = "IntArrayRef"
                              , _isOpaqueC = True
                              , _retHsFn = " >>= unIntArrayRef"
                              , _finalizerHs = "finalizerFree" })
  ,("QScheme", MarshalRet { _castType = Just "uint8_t"
                          , _retTyHs = "Word8"
                          , _retTyC = "uint8_t"
                          , _needsAlloc = False
                          , _constructorC = ""
                          , _isOpaqueC = False
                          , _retHsFn = ""
                          , _finalizerHs = "" })
  ,("std::string", MarshalRet { _castType = Nothing
                              , _retTyHs = "Ptr CChar"
                              , _retTyC = "char*"
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
  ,("Layout", MarshalRet { _castType = Just "int8_t"
                         , _retTyHs = "Int8"
                         , _retTyC = "int8_t"
                         , _needsAlloc = False
                         , _constructorC = ""
                         , _isOpaqueC = False
                         , _retHsFn = ""
                         , _finalizerHs = "" })
  -- ,("Device", MarshalRet { _castType = Just "int"
  --                        , _retTyHs = "CInt"
  --                        , _retTyC = "int"
  --                        , _needsAlloc = False
  --                        , _constructorC = ""
  --                        , _isOpaqueC = False
  --                        , _retHsFn = ""
  --                        , _finalizerHs = "" })
  ,("TensorOptions", MarshalRet { _castType = Nothing
                                , _retTyHs = "ForeignPtr CTensorOptions"
                                , _retTyC = "TensorOptions*"
                                , _needsAlloc = True
                                , _constructorC = "TensorOptions"
                                , _isOpaqueC = False
                                , _retHsFn = ""
                                , _finalizerHs = "deleteTensorOptions" })
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
  ,("uint64_t", MarshalArg { _argTyHs = "Word64"
                           , _argTyC = "uint64_t"
                           , _dereferenceC = ""
                           , _argC = Nothing
                           , _argPreFn = Nothing })
  ,("int", MarshalArg { _argTyHs = "CInt"
                      , _argTyC = "int"
                      , _dereferenceC = ""
                      , _argC = Nothing
                      , _argPreFn = Nothing })
  ,("size_t", MarshalArg { _argTyHs = "CSize"
                         , _argTyC = "size_t"
                         , _dereferenceC = ""
                         , _argC = Nothing
                         , _argPreFn = Nothing })
  ,("c10::optional<int64_t>", MarshalArg { _argTyHs = "Maybe Int64"
                                         , _argTyC = "int64_t"
                                         , _dereferenceC = ""
                                         , _argC = optionalArgC $ \rn -> "$(int64_t "<>rn<>")"
                                           -- Just $ \n -> "($(bool " <> renameHs n <> "__is_present) ? make_optional($(int64_t " <> renameHs n <> "__value)) : c10::nullopt)"
                                         , _argPreFn = optionalArgPreFun "0"
                                           -- Just $ \n -> "let ("<>n<>"__is_present, "<>n<>"__value) = splitMaybe "<>n<> " 0 in "
                                         })
  ,("bool", MarshalArg { _argTyHs = "CBool"
                       , _argTyC = "bool"
                       , _dereferenceC = ""
                       , _argC = Nothing
                       , _argPreFn = Nothing })
  ,("c10::optional<bool>", MarshalArg { _argTyHs = "Maybe CBool"
                                      , _argTyC = "bool"
                                      , _dereferenceC = ""
                                      , _argC = optionalArgC $ \rn->"$(bool "<>rn<>")"
                                      , _argPreFn = optionalArgPreFun "0" })
  ,("double", MarshalArg { _argTyHs = "CDouble"
                         , _argTyC = "double"
                         , _dereferenceC = ""
                         , _argC = Nothing
                         , _argPreFn = Nothing })
  ,("c10::optional<double>", MarshalArg { _argTyHs = "Maybe CDouble"
                                        , _argTyC = "double"
                                        , _dereferenceC = ""
                                        , _argC = optionalArgC $ \rn->"$(double "<>rn<>")"
                                          -- Just $ \n -> "($(bool " <> renameHs n <> "__is_present) ? make_optional($(double " <> renameHs n <> "__value)) : c10::nullopt)"
                                        , _argPreFn = optionalArgPreFun "0"
                                          -- Just $ \n -> "let ("<>n<>"__is_present, "<>n<>"__value) = splitMaybe "<>n<> " 0 in "
                                        })
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
  ,("const Tensor&", MarshalArg { _argTyHs = "ForeignPtr CTensor"
                                , _argTyC = "Tensor*"
                                , _dereferenceC = "*"
                                , _argC = Just $ \n -> "*$fptr-ptr:(Tensor* " <> renameHs n <> ")"
                                , _argPreFn = Nothing })
  ,("const at::Tensor&", MarshalArg { _argTyHs = "ForeignPtr CTensor"
                                    , _argTyC = "Tensor*"
                                    , _dereferenceC = "*"
                                    , _argC = Just $ \n -> "*$fptr-ptr:(Tensor* " <> renameHs n <> ")"
                                    , _argPreFn = Nothing })
  ,("Tensor", MarshalArg { _argTyHs = "ForeignPtr CTensor"
                         , _argTyC = "Tensor*"
                         , _dereferenceC = "*"
                         , _argC = Just $ \n -> "*$fptr-ptr:(Tensor* " <> renameHs n <> ")"
                         , _argPreFn = Nothing  })
  ,("Tensor *", MarshalArg { _argTyHs = "ForeignPtr CTensor"
                           , _argTyC = "Tensor*"
                           , _dereferenceC = ""
                           , _argC = Just $ \n -> "$fptr-ptr:(Tensor* " <> renameHs n <> ")"
                           , _argPreFn = Nothing  })
  ,("const c10::optional<Tensor> &", MarshalArg { _argTyHs = "Maybe (ForeignPtr CTensor)"
                                                , _argTyC = "Tensor*"
                                                , _dereferenceC = "*"
                                                , _argC = optionalArgC $ \rn -> "*$fptr-ptr:(Tensor* " <> rn <> ")"
                                                  -- Just $ \n -> "$fptr-ptr:(Tensor* " <> renameHs n <> ")"
                                                , _argPreFn = optionalArgPreFun "nullForeignPtr"
                                                  -- Nothing
                                                })
  ,("const TensorOptions &", MarshalArg { _argTyHs = "ForeignPtr CTensorOptions"
                                        , _argTyC = "TensorOptions*"
                                        , _dereferenceC = "*"
                                        , _argC = Just $ \n -> "*$fptr-ptr:(TensorOptions* " <> renameHs n <> ")"
                                        , _argPreFn = Nothing })
  ,("TensorOptions", MarshalArg { _argTyHs = "ForeignPtr CTensorOptions"
                                , _argTyC = "TensorOptions*"
                                , _dereferenceC = "*"
                                , _argC = Just $ \n -> "*$fptr-ptr:(TensorOptions* " <> renameHs n <> ")"
                                , _argPreFn = Nothing })
  ,("Scalar", MarshalArg { _argTyHs = "ForeignPtr CScalar"
                         , _argTyC = "Scalar*"
                         , _dereferenceC = "*"
                         , _argC = Just $ \n -> "*$fptr-ptr:(Scalar* " <> renameHs n <> ")"
                         , _argPreFn = Nothing })
  ,("c10::optional<Scalar>", MarshalArg { _argTyHs = "Maybe (ForeignPtr CScalar)"
                                        , _argTyC = "Scalar*"
                                        , _dereferenceC = "*"
                                        , _argC = optionalArgC $ \rn -> "*$fptr-ptr:(Scalar* " <> rn <> ")"
                                        , _argPreFn = optionalArgPreFun "nullForeignPtr"
                                        })
  ,("ScalarType", MarshalArg { _argTyHs = "Int8"
                             , _argTyC = "int8_t"
                             , _dereferenceC = ""
                             , _argC = Just $ \n -> "static_cast<ScalarType>($(int8_t " <> renameHs n <> "))"
                             , _argPreFn = Nothing })
  ,("c10::optional<ScalarType>", MarshalArg { _argTyHs = "Maybe Int8"
                                            , _argTyC = "int8_t"
                                            , _dereferenceC = ""
                                            , _argC = optionalArgC $ \rn -> "static_cast<ScalarType>($(int8_t " <> rn <> "))"
                                            , _argPreFn = optionalArgPreFun "0" })
  ,("MemoryFormat", MarshalArg { _argTyHs = "Int8"
                               , _argTyC = "int8_t"
                               , _dereferenceC = ""
                               , _argC = Just $ \n -> "static_cast<MemoryFormat>($(int8_t " <> renameHs n <> "))"
                               , _argPreFn = Nothing })
  ,("at::MemoryFormat", MarshalArg { _argTyHs = "Int8"
                                   , _argTyC = "int8_t"
                                   , _dereferenceC = ""
                                   , _argC = Just $ \n -> "static_cast<MemoryFormat>($(int8_t " <> renameHs n <> "))"
                                   , _argPreFn = Nothing })
  ,("c10::optional<MemoryFormat>", MarshalArg { _argTyHs = "Maybe Int8"
                                              , _argTyC = "int8_t"
                                              , _dereferenceC = ""
                                              , _argC = optionalArgC $ \rn -> "static_cast<MemoryFormat>($(int8_t " <> rn <> "))"
                                              , _argPreFn = optionalArgPreFun "0" })
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
  ,("c10::optional<IntArrayRef>", MarshalArg { _argTyHs = "Maybe (Vector Int64)"
                                             , _argTyC = "IntArrayRef"
                                             , _dereferenceC = ""
                                             , _argC = optionalArgC $ \rn -> "ArrayRef<int64_t>($(int64_t* " <> rn <> "__array), $(size_t "<> rn <>"__size))"
                                             , _argPreFn =
                                               optionalArgPreFun' "V.empty" $ \n ->
                                                 "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in "})
  ,("ArrayRef<double>", MarshalArg { _argTyHs = "Vector CDouble"
                                   , _argTyC = "ArrayRef<double>"
                                   , _dereferenceC = ""
                                   , _argC = Just $ \n -> "ArrayRef<double>($(double* " <> renameHs n <> "__array), $(size_t "<> n <>"__size))"
                                   , _argPreFn = Just $ \n -> "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in "})
  ,("c10::optional<ArrayRef<double>>", MarshalArg { _argTyHs = "Maybe (Vector CDouble)"
                                                  , _argTyC = "ArrayRef<double>"
                                                  , _dereferenceC = ""
                                                  , _argC = optionalArgC $ \rn -> "ArrayRef<double>($(double* " <> rn <> "__array), $(size_t "<> rn <>"__size))"
                                                  , _argPreFn = optionalArgPreFun' "V.empty" $ \n ->
                                                      "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in "})
  ,("TensorList", MarshalArg { _argTyHs = "Vector (Ptr CTensor)"
                             , _argTyC = "TensorList"
                             , _dereferenceC = ""
                             , _argC = Just $ \n -> "pack_tensor_list($(Tensor** " <> renameHs n <> "__array), $(size_t "<> n <>"__size))"
                             , _argPreFn = Just $ \n -> "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in " })
  ,("c10::optional<TensorList>", MarshalArg { _argTyHs = "Vector (Ptr CTensor)"
                             , _argTyC = "TensorList"
                             , _dereferenceC = ""
                             , _argC = Just $ \n -> "pack_tensor_list($(Tensor** " <> renameHs n <> "__array), $(size_t "<> n <>"__size))"
                             , _argPreFn = Just $ \n -> "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in " })
  ,("const c10::List<c10::optional<Tensor>> &", MarshalArg { _argTyHs = "Vector (Ptr CTensor)"
                             , _argTyC = "c10::List<c10::optional<Tensor>>"
                             , _dereferenceC = ""
                             , _argC = Just $ \n -> "pack_tensor_optional_list($(Tensor** " <> renameHs n <> "__array), $(size_t "<> n <>"__size))"
                             , _argPreFn = Just $ \n -> "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in " })
  ,("Generator *", MarshalArg { _argTyHs = "Ptr CGenerator"
                              , _argTyC = "Generator*"
                              , _dereferenceC = ""
                              , _argC = Nothing
                              , _argPreFn = Nothing })
  ,("c10::optional<Generator>", MarshalArg { _argTyHs = "Maybe (Ptr CGenerator)"
                                           , _argTyC = "Generator*"
                                           , _dereferenceC = ""
                                           , _argC = optionalArgC $ \rn -> "*$(Generator* " <> rn <> ")"
                                           , _argPreFn = optionalArgPreFun "nullPtr" })
  ,("std::string", MarshalArg { _argTyHs = "Ptr CChar"
                              , _argTyC = "char*"
                              , _dereferenceC = ""
                              , _argC = Nothing
                              , _argPreFn = Nothing })
  ,("c10::optional<std::string>", MarshalArg { _argTyHs = "Ptr CChar"
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
  ,("Backend", MarshalArg { _argTyHs = "Int32"
                          , _argTyC = "int32_t"
                          , _dereferenceC = "(Backend)"
                          , _argC = Nothing
                          , _argPreFn = Nothing })
  ,("Device", MarshalArg { _argTyHs = "Ptr CDevice"
                         , _argTyC = "Device*"
                         , _dereferenceC = "*"
                         , _argC = Nothing
                         , _argPreFn = Nothing })
  ,("c10::optional<Device>", MarshalArg { _argTyHs = "Ptr CDevice"
                                        , _argTyC = "Device*"
                                        , _dereferenceC = "*"
                                        , _argC = Nothing
                                        , _argPreFn = Nothing })
  ,("Layout", MarshalArg { _argTyHs = "Int8"
                         , _argTyC = "(Layout)"
                         , _dereferenceC = ""
                         , _argC = Nothing
                         , _argPreFn = Nothing })
  ,("c10::optional<Layout>", MarshalArg { _argTyHs = "Maybe Int8"
                                        , _argTyC = "(Layout)"
                                        , _dereferenceC = ""
                                        , _argC = optionalArgC $ \rn -> "((Layout)$(int8_t "<>rn<>"))"
                                        , _argPreFn = optionalArgPreFun "0" })
  ]
  where optionalArgC argCFn = Just $ \n -> "($(bool " <> renameHs n <> "__is_present) ? make_optional("<> argCFn (renameHs n <> "__value") <>") : c10::nullopt)"
        optionalArgPreFun def = optionalArgPreFun' def (const "")
        optionalArgPreFun' def fn = Just $ \n -> "let ("<>renameHs n<>"__is_present, "<>renameHs n<>"__value) = splitMaybe "<>renameHs n<> " "<>def<>" in " <> fn (renameHs n<> "__value")

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

mangleNameByType :: Text -> [Text] -> Bool -> Text
mangleNameByType name tys member =
  name <> (if member then "_m" else "__") <> T.intercalate "" (map (\ty -> case M.lookup ty table of
                                                                       Nothing -> error $ "Can't mangle " ++ show ty
                                                                       Just t -> t) tys)
  where table = M.fromList [("int64_t", "6")
                           ,("uint64_t", "6")
                           ,("c10::optional<int64_t>", "6")
                           ,("int", "3")
                           ,("bool", "b")
                           ,("c10::optional<bool>", "b")
                           ,("double", "d")
                           ,("c10::optional<double>", "d")
                           ,("Tensor &", "t")
                           ,("const Tensor &", "t")
                           ,("const Tensor&", "t")
                           ,("const at::Tensor&", "t")
                           ,("Tensor", "t")
                           ,("Tensor *", "t")
                           ,("const c10::optional<Tensor> &", "t")
                           ,("const c10::List<c10::optional<Tensor>> &", "t")
                           ,("const TensorOptions &", "o")
                           ,("TensorOptions", "o")
                           ,("Scalar", "s")
                           ,("c10::optional<Scalar>", "s")
                           ,("ScalarType", "S")
                           ,("c10::optional<ScalarType>", "S")
                           ,("MemoryFormat", "M")
                           ,("at::MemoryFormat", "M")
                           ,("c10::optional<MemoryFormat>", "M")
                           ,("std::array<bool,2>", "a")
                           ,("std::array<bool,3>", "a")
                           ,("std::array<bool,4>", "a")
                           ,("IntArrayRef", "a")
                           ,("c10::optional<IntArrayRef>", "a")
                           ,("ArrayRef<double>", "a")
                           ,("c10::optional<ArrayRef<double>>", "a")
                           ,("TensorList", "l")
                           ,("c10::optional<TensorList>", "l")
                           ,("Generator *", "g")
                           ,("c10::optional<Generator>", "g")
                           ,("std::string", "s")
                           ,("c10::optional<std::string>", "s")
                           ,("Storage &", "S")
                           ,("Storage", "S")
                           ,("Device", "D")
                           ,("c10::optional<Device>", "D")
                           ,("Layout", "l")
                           ,("c10::optional<Layout>", "l")
                           ,("Backend", "b")]

generateFromFile :: Text -> Text -> Maybe Text -> ([Text] -> [Text]) -> Bool -> Bool -> IO [Text]
generateFromFile filename start end cleanup member verbose = do
  unlessM (doesFileExist $ T.unpack filename) (exitWithUsageMessage patterns $ "Header doesn't exist! " ++ show filename)
  fin <- T.readFile $ T.unpack filename
  let os =
        filter (\(ty, name, args, member) -> (not (T.isInfixOf "_forward" name)) && (not (T.isInfixOf "_backward" name)))
        $ map (\x ->
         case T.splitOn "(" x of
           [pre,args'] ->
             let args = case T.splitOn ", " $ fst $ T.breakOn ")" args' of
                          [""] -> []
                          x    -> map (T.takeWhile (/= '=')) x -- default arguments
                 (ty, name) = splitTyAndName pre
             in (ty, name, args, member)
           _ -> error $ "Failed to split function: " ++ show x
              )
        $ filter (not . T.isInfixOf "ConstQuantizerPtr") -- NB We don't yet support quantization
        $ filter (not . T.isInfixOf "Dimname") -- NB We don't support named dimensions
        $ cleanup
        $ T.lines
        $ (\x -> case end of
                  Nothing -> x
                  Just end' -> fst $ T.breakOn end' x)
        $ snd
        $ T.breakOn start fin
  when verbose $ mapM_ print os
  let ls = snd $ mapAccumL (\m (ty, nameC, nameHs, args, member) ->
                              (M.alter (\x -> Just $ case x of
                                           Nothing -> 1
                                           Just n  -> n + 1)
                                nameHs
                                m
                              ,emit False member ty nameC
                                (case M.lookup nameHs m of
                                    Nothing -> nameHs
                                    Just n  -> nameHs <> "__" <> T.pack (show n))
                                args))
                M.empty $ map (\(ty, nameC, args, member) ->
                             (ty
                             , nameC
                             , mangleNameByType nameC (map (fst . splitTyAndName) args) member
                             , map splitTyAndName args
                             , member)) os
  when verbose $ mapM_ T.putStrLn ls
  pure ls

generateFunctions :: Text -> Bool -> IO [Text]
generateFunctions filename verbose = do
  generateFromFile filename "TORCH_API Tensor _cast_Byte(" Nothing
    (-- filter (T.isInfixOf "{")
     --  . 
     map (T.replace "TORCH_API " "")
      . filter (T.isInfixOf "TORCH_API "))
    False
    verbose

generateTensorBody :: Text -> Bool -> IO [Text]
generateTensorBody filename verbose = do
  -- generateFromFile filename "  void backward(" (Just "// We changed .dtype")
  generateFromFile filename "  int64_t dim() const" (Just "// We changed .dtype")
    (map (T.replace ") const" ")")
      . filter (T.isInfixOf ") const")
      . filter (not . T.isInfixOf "is_variable") -- This will generate a deprecation warning
      . filter (not . T.isInfixOf "TensorImpl")
      . filter (not . T.isInfixOf "channels_last_strides_exact_match")
      . filter (not . T.isInfixOf "DeprecatedTypeProperties")
      . filter (not . T.isInfixOf "KeySet")
      . filter (not . T.isInfixOf "Storage")
      . filter (not . T.isInfixOf "TypeMeta")
      . filter (not . T.isInfixOf "QuantizerPtr")
      . filter (not . T.isInfixOf "TensorAccessor")
      . filter (not . T.isInfixOf "(Stream ")
      . filter (not . T.isInfixOf "  T ")
      . filter (not . T.isInfixOf "Device device() const")
      . filter (not . T.isInfixOf "toString()") -- TODO long term
      . filter (not . T.isInfixOf "operator") -- TODO long term
      . filter (not . T.isInfixOf "NamedTensorMeta") -- TODO
      -- . filter (not . T.isInfixOf "c10::List<c10::optional") -- TODO
      . filter (not . T.isInfixOf "std::initializer_list<at::indexing::TensorIndex>") -- TODO
      . filter (not . T.isInfixOf "ArrayRef<at::indexing::TensorIndex>") -- TODO short term
      . filter (not . T.isInfixOf "//"))
    True
    verbose

main :: IO ()
main = do
  args <- parseArgsOrExit patterns =<< getArgs
  destFile            <- T.pack <$> args `getArgOrExit` argument "dest-file"
  functionsHeader  <- T.pack <$> args `getArgOrExit` argument "functions-header"
  tensorBodyHeader  <- T.pack <$> args `getArgOrExit` argument "tensor-body-header"
  unlessM (doesFileExist $ T.unpack destFile) (exitWithUsageMessage patterns $ "Destination file doesn't exist! " ++ show destFile)
  dest <- T.readFile $ T.unpack destFile
  generatedFunctions <- generateFunctions functionsHeader (args `isPresent` longOption "verbose")
  generatedTensorBody <- generateTensorBody tensorBodyHeader (args `isPresent` longOption "verbose")
  case T.breakOnEnd "-- Everything below is AUTOGENERATED from generate-ctensor" dest of
    (pre, _) -> T.writeFile (T.unpack destFile) (pre <> "\n"
                                                <> T.unlines generatedFunctions <> "\n"
                                                <> T.unlines generatedTensorBody)
  T.putStrLn $ "Updated " <> destFile
  pure ()
