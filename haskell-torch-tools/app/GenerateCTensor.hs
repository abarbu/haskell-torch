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

needsCast :: Text -> Bool
needsCast "int64_t"                                        = False
needsCast "c10::optional<int64_t>"                         = False
needsCast "bool"                                           = False
needsCast "double"                                         = False
needsCast "Tensor &"                                       = False
needsCast "const Tensor &"                                 = False
needsCast "void*"                                          = False
needsCast "void"                                           = False
needsCast "Tensor"                                         = False
needsCast "SparseTensorRef"                                = False
needsCast "Scalar"                                         = False
needsCast "std::tuple<Tensor,Tensor>"                      = False
needsCast "std::tuple<Tensor,Tensor,Tensor>"               = False
needsCast "std::tuple<Tensor,Tensor,Tensor,Tensor>"        = False
needsCast "std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>" = False
needsCast "std::tuple<Tensor,Tensor,Tensor,int64_t>"       = False
needsCast "std::tuple<Tensor,Tensor,double,int64_t>"       = False
needsCast "std::vector<Tensor>"                            = False
needsCast "QScheme"                                        = True
needsCast "c10::optional<QScheme>"                         = True
needsCast x                                                = error $ "Don't know how to needsCast this: " ++ show x

castTy :: Text -> Text
castTy "QScheme"                = "uint8_t"
castTy "c10::optional<QScheme>" = "uint8_t"
castTy x                        = error $ "Don't know how to castTy this: " ++ show x

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

-- NB When we talk about vectors here, we mean Data.Vector.Storable

-- What Haskell type should this return value have?
retTyHs :: Text -> Text
retTyHs "Tensor &" = "ForeignPtr CTensor"
retTyHs "Tensor" = "ForeignPtr CTensor"
retTyHs "SparseTensorRef" = "ForeignPtr CSparseTensorRef"
retTyHs "Scalar" = "ForeignPtr CScalar"
retTyHs "int64_t" = "Int64"
retTyHs "double" = "CDouble"
retTyHs "bool" = "CBool"
retTyHs "void*" = "Ptr ()"
retTyHs "void" = "()"
retTyHs "QScheme" = "Word8"
retTyHs "std::vector<Tensor>" = "Vector (Ptr CTensor)"
retTyHs "std::vector<Tensor,Tensor,double,int64_t>" = "Vector (Ptr ())" -- that's ugly..
retTyHs x =
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
retTyC :: (Eq a, IsString a, IsString p, Show a) => a -> p
retTyC "Tensor &"                                       = "Tensor*"
retTyC "Tensor"                                         = "Tensor*"
retTyC "SparseTensorRef"                                = "SparseTensorRef*"
retTyC "Scalar"                                         = "Scalar*"
retTyC "int64_t"                                        = "int64_t"
retTyC "bool"                                           = "bool"
retTyC "double"                                         = "double"
retTyC "void*"                                          = "void*"
retTyC "void"                                           = "void"
retTyC "std::tuple<Tensor,Tensor>"                      = "void*"
retTyC "std::tuple<Tensor,Tensor,Tensor>"               = "void*"
retTyC "std::tuple<Tensor,Tensor,Tensor,Tensor>"        = "void*"
retTyC "std::tuple<Tensor,Tensor,Tensor,int64_t>"       = "void*"
retTyC "std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>" = "void*"
retTyC "std::vector<Tensor>"                            = "void*"
retTyC "std::tuple<Tensor,Tensor,double,int64_t>"       = "void*"
retTyC "QScheme"                                        = "uint8_t"
retTyC x                                                = error $ "Don't know how to retTyC this: " ++ show x

-- What type does this argument have in Haskell?
argTyHs :: (Eq a, IsString a, IsString p, Show a) => a -> p
argTyHs "const TensorOptions &"       = "ForeignPtr CTensorOptions"
argTyHs "Tensor &"                    = "ForeignPtr CTensor"
argTyHs "Tensor"                      = "ForeignPtr CTensor"
argTyHs "const Tensor &"              = "ForeignPtr CTensor"
argTyHs "SparseTensorRef"             = "ForeignPtr CSparseTensorRef"
argTyHs "Storage &"                   = "Ptr CStorage"
argTyHs "Storage"                     = "Ptr CStorage"
argTyHs "Scalar"                      = "ForeignPtr CScalar"
argTyHs "c10::optional<Scalar>"       = "ForeignPtr CScalar"
argTyHs "Generator *"                 = "Ptr CGenerator"
argTyHs "bool"                        = "CBool"
argTyHs "c10::optional<bool>"         = "CBool"
argTyHs "double"                      = "CDouble"
argTyHs "IntArrayRef"                 = "Vector Int64"
argTyHs "std::array<bool,2>"          = "Vector CBool"
argTyHs "std::array<bool,3>"          = "Vector CBool"
argTyHs "std::array<bool,4>"          = "Vector CBool"
argTyHs "TensorList"                  = "Vector (Ptr CTensor)"
argTyHs "int64_t"                     = "Int64"
argTyHs "c10::optional<int64_t>"      = "Maybe Int64"
argTyHs "ScalarType"                  = "Int8"
argTyHs "c10::optional<ScalarType>"   = "Int8"
argTyHs "Device"                      = "Ptr CDevice"
argTyHs "std::string"                 = "Ptr CChar"
argTyHs "const Type &"                = "Ptr CVariableType"
argTyHs "c10::optional<MemoryFormat>" = "Int8"
argTyHs "MemoryFormat"                = "Int8"
argTyHs x                             = error $ "Don't know how to argTyHs this: " ++ show x

argTyC :: (Eq a, IsString a, IsString p, Show a) => a -> p
argTyC "const TensorOptions &"       = "TensorOptions*"
argTyC "Tensor &"                    = "Tensor*"
argTyC "Tensor"                      = "Tensor*"
argTyC "const Tensor &"              = "Tensor*"
argTyC "SparseTensorRef"             = "SparseTensorRef*"
argTyC "Storage &"                   = "Storage*"
argTyC "Storage"                     = "Storage*"
argTyC "Scalar"                      = "Scalar*"
argTyC "c10::optional<Scalar>"       = "Scalar*"
argTyC "Generator *"                 = "Generator*"
argTyC "bool"                        = "bool"
argTyC "c10::optional<bool>"         = "bool"
argTyC "double"                      = "double"
argTyC "int64_t"                     = "int64_t"
argTyC "c10::optional<int64_t>"      = "int64_t"
argTyC "ScalarType"                  = "int8_t"
argTyC "c10::optional<ScalarType>"   = "int8_t"
argTyC "Device"                      = "Device*"
argTyC "std::string"                 = "char*"
argTyC "const Type &"                = "VariableType*"
argTyC "c10::optional<MemoryFormat>" = "int8_t"
argTyC "MemoryFormat"                = "int8_t"
argTyC x                             = error $ "Don't know how to argTyC this: " ++ show x

-- If this is a pointer, how do we access its memory?
dereferenceC :: (Eq a, IsString a, IsString p, Show a) => a -> p
dereferenceC "const TensorOptions &"       = "*"
dereferenceC "Tensor &"                    = "*"
dereferenceC "const Tensor &"              = "*"
dereferenceC "Tensor"                      = "*"
dereferenceC "SparseTensorRef"             = "*"
dereferenceC "Storage &"                   = "*"
dereferenceC "Storage"                     = "*"
dereferenceC "Scalar"                      = "*"
dereferenceC "c10::optional<Scalar>"       = "*"
dereferenceC "Generator *"                 = ""
dereferenceC "bool"                        = ""
dereferenceC "c10::optional<bool>"         = ""
dereferenceC "double"                      = ""
dereferenceC "IntArrayRef"                 = ""
dereferenceC "std::array<bool,2>"          = ""
dereferenceC "std::array<bool,3>"          = ""
dereferenceC "std::array<bool,4>"          = ""
dereferenceC "TensorList"                  = ""
dereferenceC "int64_t"                     = ""
dereferenceC "c10::optional<int64_t>"      = ""
dereferenceC "ScalarType"                  = ""
dereferenceC "c10::optional<ScalarType>"   = ""
dereferenceC "Device"                      = "*"
dereferenceC "std::string"                 = ""
dereferenceC "const Type &"                = "*"
dereferenceC "c10::optional<MemoryFormat>" = ""
dereferenceC "MemoryFormat"                = ""
dereferenceC x                             = error $ "Don't know how to dereferenceC this: " ++ show x

-- Do we need to alocate memory in order to marshal this type?
needsAlloc :: (Eq a, IsString a, Show a) => a -> Bool
needsAlloc "int64_t"                                        = False
needsAlloc "c10::optional<int64_t>"                         = False
needsAlloc "bool"                                           = False
needsAlloc "double"                                         = False
needsAlloc "Tensor &"                                       = False
needsAlloc "const Tensor &"                                 = False
needsAlloc "void*"                                          = False
needsAlloc "void"                                           = False
needsAlloc "Tensor"                                         = True
needsAlloc "SparseTensorRef"                                = True
needsAlloc "Scalar"                                         = True
needsAlloc "std::tuple<Tensor,Tensor>"                      = True
needsAlloc "std::tuple<Tensor,Tensor,Tensor>"               = True
needsAlloc "std::tuple<Tensor,Tensor,Tensor,Tensor>"        = True
needsAlloc "std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>" = True
needsAlloc "std::tuple<Tensor,Tensor,Tensor,int64_t>"       = True
needsAlloc "std::tuple<Tensor,Tensor,double,int64_t>"       = True
needsAlloc "std::vector<Tensor>"                            = True
needsAlloc "QScheme"                                        = False
needsAlloc x                                                = error $ "Don't know how to needsAlloc this: " ++ show x

-- How do we build this tpye in C?
constructorC :: (Eq a, IsString a, IsString p, Show a) => a -> p
constructorC "Tensor &"                                       = "Tensor"
constructorC "Tensor"                                         = "Tensor"
constructorC "SparseTensorRef"                                = "SparseTensorRef"
constructorC "Scalar"                                         = "Scalar"
constructorC "std::tuple<Tensor,Tensor>"                      = "std::tuple<Tensor,Tensor>"
constructorC "std::tuple<Tensor,Tensor,Tensor>"               = "std::tuple<Tensor,Tensor,Tensor>"
constructorC "std::tuple<Tensor,Tensor,Tensor,Tensor>"        = "std::tuple<Tensor,Tensor,Tensor,Tensor>"
constructorC "std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>" = "std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>"
constructorC "std::tuple<Tensor,Tensor,Tensor,int64_t>"       = "std::tuple<Tensor,Tensor,Tensor,int64_t>"
constructorC "std::tuple<Tensor,Tensor,double,int64_t>"       = "std::tuple<Tensor,Tensor,double,int64_t>"
constructorC "std::vector<Tensor>"                            = "std::vector<Tensor>"
constructorC x                                                = error $ "Don't know how to constructorC this: " ++ show x

isOpaqueC :: (Eq a, IsString a, Show a) => a -> Bool
isOpaqueC "Tensor"                                         = False
isOpaqueC "Tensor&"                                        = False
isOpaqueC "SparseTensorRef"                                = False
isOpaqueC "Scalar"                                         = False
isOpaqueC "bool"                                           = False
isOpaqueC "double"                                         = False
isOpaqueC "void*"                                          = False
isOpaqueC "int64_t"                                        = False
isOpaqueC "c10::optional<int64_t>"                         = False
isOpaqueC "std::tuple<Tensor,Tensor>"                      = True
isOpaqueC "std::tuple<Tensor,Tensor,Tensor>"               = True
isOpaqueC "std::tuple<Tensor,Tensor,Tensor,Tensor>"        = True
isOpaqueC "std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>" = True
isOpaqueC "std::tuple<Tensor,Tensor,Tensor,int64_t>"       = True
isOpaqueC "std::tuple<Tensor,Tensor,double,int64_t>"       = True
isOpaqueC "std::vector<Tensor>"                            = True
isOpaqueC x                                                = error $ "Don't know how to isOpaqueC this: " ++ show x

-- What function will marshal this from C to Hs
retHsFn :: (Eq a, IsString a, IsString p, Show a) => a -> p
retHsFn "Tensor"                                         = ""
retHsFn "Tensor&"                                        = ""
retHsFn "SparseTensorRef"                                = ""
retHsFn "Scalar"                                         = ""
retHsFn "bool"                                           = ""
retHsFn "double"                                         = ""
retHsFn "void*"                                          = ""
retHsFn "void"                                           = ""
retHsFn "int64_t"                                        = ""
retHsFn "std::tuple<Tensor,Tensor>"                      = " >>= unTupleTensorTensor"
retHsFn "std::tuple<Tensor,Tensor,Tensor>"               = " >>= unTupleTensorTensorTensor"
retHsFn "std::tuple<Tensor,Tensor,Tensor,Tensor>"        = " >>= unTupleTensorTensorTensorTensor"
retHsFn "std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>" = " >>= unTupleTensorTensorTensorTensorTensor"
retHsFn "std::tuple<Tensor,Tensor,Tensor,int64_t>"       = " >>= unTupleTensorTensorTensorInt64"
retHsFn "std::tuple<Tensor,Tensor,double,int64_t>"       = " >>= unTupleTensorTensorDoubleInt64"
retHsFn "std::vector<Tensor>"                            = " >>= unVectorTensor"
retHsFn "QScheme"                                        = ""
retHsFn x                                                = error $ "Don't know how to retHsFn this: " ++ show x

-- How will we free the memory in Haskell?
finalizerHs :: (Eq a, IsString a, IsString p, Show a) => a -> p
finalizerHs "Tensor"                                         = "deleteTensor"
finalizerHs "Tensor&"                                        = "deleteTensor"
finalizerHs "SparseTensorRef"                                = "deleteTensor"
finalizerHs "Scalar"                                         = "deleteScalar'"
finalizerHs "void*"                                          = "finalizerFree"
finalizerHs "std::tuple<Tensor,Tensor>"                      = "finalizerFree"
finalizerHs "std::tuple<Tensor,Tensor,Tensor>"               = "finalizerFree"
finalizerHs "std::tuple<Tensor,Tensor,Tensor,Tensor>"        = "finalizerFree"
finalizerHs "std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>" = "finalizerFree"
finalizerHs "std::tuple<Tensor,Tensor,Tensor,int64_t>"       = "finalizerFree"
finalizerHs "std::vector<Tensor>"                            = "finalizerFree"
finalizerHs x                                                = error $ "Don't know how to finalizerHs this: " ++ show x

renameHs :: Text -> Text
renameHs "where" = "whereX"
renameHs "data"  = "dataX"
renameHs "type"  = "typeX"
renameHs "in"    = "inX"
renameHs x       = T.toLower x

argC :: (Eq a, IsString a, Show a) => (a, Text) -> Text
argC ("IntArrayRef", n)                 = "ArrayRef<int64_t>($(int64_t* " <> renameHs n <> "__array), $(size_t "<> n <>"__size))"
argC ("std::array<bool,2>", n)          = "make_array_bool_2($(bool* " <> renameHs n <> "__array))"
argC ("std::array<bool,3>", n)          = "make_array_bool_3($(bool* " <> renameHs n <> "__array))"
argC ("std::array<bool,4>", n)          = "make_array_bool_4($(bool* " <> renameHs n <> "__array))"
argC ("TensorList", n)                  = "pack_tensor_list($(Tensor** " <> renameHs n <> "__array), $(size_t "<> n <>"__size))"
argC ("Tensor &", n)                    = "*$fptr-ptr:(Tensor* " <> renameHs n <> ")"
argC ("const Tensor &", n)              = "*$fptr-ptr:(Tensor* " <> renameHs n <> ")"
argC ("Tensor", n)                      = "*$fptr-ptr:(Tensor* " <> renameHs n <> ")"
argC ("Scalar", n)                      = "*$fptr-ptr:(Scalar* " <> renameHs n <> ")"
argC ("c10::optional<Scalar>", n)       = "*$fptr-ptr:(Scalar* " <> renameHs n <> ")"
argC ("const TensorOptions &", n)       = "*$fptr-ptr:(TensorOptions* " <> renameHs n <> ")"
argC ("SparseTensorRef", n)             = "*$fptr-ptr:(SparseTensorRef* " <> renameHs n <> ")"
argC ("MemoryFormat", n)                = "static_cast<MemoryFormat>($(int8_t " <> renameHs n <> "))"
argC ("c10::optional<MemoryFormat>", n) = "static_cast<MemoryFormat>($(int8_t " <> renameHs n <> "))"
argC ("ScalarType", n)                  = "static_cast<ScalarType>($(int8_t " <> renameHs n <> "))"
argC ("c10::optional<ScalarType>", n)   = "static_cast<ScalarType>($(int8_t " <> renameHs n <> "))"
argC ("c10::optional<int64_t>", n)      = "($(bool " <> renameHs n <> "__is_present) ? make_optional($(int64_t " <> renameHs n <> "__value)) : c10::nullopt)"
argC (ty, n)                            = dereferenceC ty <> "$(" <> argTyC ty <> " " <> renameHs n <> ")"

argPreFn :: (Eq a1, Semigroup a2, IsString a1, IsString a2) => (a1, a2) -> a2
argPreFn ("IntArrayRef", n) =
  "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in "
argPreFn ("std::array<bool,2>", n) =
  "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in "
argPreFn ("std::array<bool,3>", n) =
  "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in "
argPreFn ("std::array<bool,4>", n) =
  "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in "
argPreFn ("TensorList", n) =
  "V.unsafeWith "<>n<>" $ \\"<>n<>"__array -> let "<> n <>"__size = fromIntegral (V.length "<> n <>") in "
argPreFn ("c10::optional<int64_t>", n) =
  "let ("<>n<>"__is_present, "<>n<>"__value) = splitMaybe "<>n<> " 0 in "
argPreFn _ = ""

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
        $ filter (T.isInfixOf ";") $ T.lines $ fst $ T.breakOn "private:" $ snd $ T.breakOn "Tensor __and__" (T.replace "  static " "" fin)
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
