{-# LANGUAGE FlexibleContexts, MultiWayIf, OverloadedLists, OverloadedStrings, ScopedTypeVariables #-}
module Types where
import           Control.Applicative
import           Control.Lens             hiding (argument)
import           Control.Monad
import           Control.Monad.Extra
import qualified Data.Aeson               as A
import           Data.Aeson.Lens
import           Data.Aeson.TH
import           Data.Char
import           Data.Data
import qualified Data.HashMap.Strict      as H
import           Data.List
import           Data.Map                 (Map)
import qualified Data.Map                 as M
import           Data.Maybe
import           Data.Monoid
import           Data.Text                (Text)
import qualified Data.Text                as T
import qualified Data.Text.Encoding       as T
import qualified Data.Text.IO             as T
import qualified Data.Text.Lazy           as TL
import qualified Data.Text.Read           as T
import qualified Data.Vector              as V
import           Data.Version
import qualified Data.Yaml                as Y
import           System.Console.Docopt
import           System.Directory
import           System.Environment       (getArgs)
import           System.Exit
import qualified Text.Mustache            as M
import           Text.Mustache.Compile.TH (mustache)

scalarMap :: Map Text Text
scalarMap = [("Byte",        "Word8")
            ,("Char",        "Int8")
            ,("Double",      "CDouble")
            ,("Float",       "CFloat")
            ,("Half",        "Half")
            ,("Int",         "CInt")
            ,("Long",        "CLong")
            ,("Short",       "CShort")
            ,("CudaByte",    "Word8")
            ,("CudaChar",    "Int8")
            ,("CudaDouble",  "CDouble")
            ,("CudaFloat",   "CFloat")
            ,("CudaHalf",    "Half")
            ,("CudaInt",     "Int32")
            ,("CudaLong",    "Int64")
            ,("CudaShort",   "Int16")]

cscalarMap :: Map Text Text
cscalarMap = [("Byte",       "uint8_t")
             ,("Char",       "int8_t")
             ,("Double",     "double")
             ,("Float",      "float")
             ,("Half",       "THHalf")
             ,("Int",        "int")
             ,("Long",       "long")
             ,("Short",      "short")
             ,("CudaByte",   "uint8_t")
             ,("CudaChar",   "int8_t")
             ,("CudaDouble", "double")
             ,("CudaFloat",  "float")
             ,("CudaHalf",   "THHalf")
             ,("CudaInt",    "int32_t")
             ,("CudaLong",   "int64_t")
             ,("CudaShort",  "int16_t")]

cscalarMap' = M.fromList $ map (\(a,b) -> (b,a)) $ M.toList cscalarMap

hostMap :: Map Text Text
hostMap   = [("Byte",        "Byte")
            ,("Char",        "Char")
            ,("Double",      "Double")
            ,("Float",       "Float")
            ,("Half",        "Half")
            ,("Int",         "Int")
            ,("Long",        "Long")
            ,("Short",       "Short")
            ,("CudaByte",    "Byte")
            ,("CudaChar",    "Char")
            ,("CudaDouble",  "Double")
            ,("CudaFloat",   "Float")
            ,("CudaHalf",    "Half")
            ,("CudaInt",     "Int")
            ,("CudaLong",    "Long")
            ,("CudaShort",   "Short")]

ctensorMap :: Map Text Text
ctensorMap = [("Byte",       "Byte")
             ,("Char",       "Char")
             ,("Double",     "Double")
             ,("Float",      "Float")
             ,("Half",       "Half")
             ,("Int",        "Int")
             ,("Long",       "Long")
             ,("Short",      "Short")
             ,("CudaByte",   "CudaByte")
             ,("CudaChar",   "CudaChar")
             ,("CudaDouble", "CudaDouble")
             ,("CudaFloat",  "Cuda") -- This is the odd one out
             ,("CudaHalf",   "CudaHalf")
             ,("CudaInt",    "CudaInt")
             ,("CudaLong",   "CudaLong")
             ,("CudaShort",  "CudaShort")]

ctensorMapType :: Map Text Text
ctensorMapType = [("Byte",       "kCPU")
                 ,("Char",       "kCPU")
                 ,("Double",     "kCPU")
                 ,("Float",      "kCPU")
                 ,("Half",       "kCPU")
                 ,("Int",        "kCPU")
                 ,("Long",       "kCPU")
                 ,("Short",      "kCPU")
                 ,("CudaByte",   "kCUDA")
                 ,("CudaChar",   "kCUDA")
                 ,("CudaDouble", "kCUDA")
                 ,("CudaFloat",  "kCUDA")
                 ,("CudaHalf",   "kCUDA")
                 ,("CudaInt",    "kCUDA")
                 ,("CudaLong",   "kCUDA")
                 ,("CudaShort",  "kCUDA")]

ctensorMapScalarType :: Map Text Text
ctensorMapScalarType = [("Byte",       "ScalarType::Byte")
                       ,("Char",       "ScalarType::Char")
                       ,("Double",     "ScalarType::Double")
                       ,("Float",      "ScalarType::Float")
                       ,("Half",       "ScalarType::Half")
                       ,("Int",        "ScalarType::Int")
                       ,("Long",       "ScalarType::Long")
                       ,("Short",      "ScalarType::Short")
                       ,("CudaByte",   "ScalarType::Byte")
                       ,("CudaChar",   "ScalarType::Char")
                       ,("CudaDouble", "ScalarType::Double")
                       ,("CudaFloat",  "ScalarType::Float")
                       ,("CudaHalf",   "ScalarType::Half")
                       ,("CudaInt",    "ScalarType::Int")
                       ,("CudaLong",   "ScalarType::Long")
                       ,("CudaShort",  "ScalarType::Short")]

sparseMap :: Map Text Text
sparseMap = [("Byte",       "Byte")
            ,("Char",       "Char")
            ,("Double",     "Double")
            ,("Float",      "Float")
            ,("Half",       "Half")
            ,("Int",        "Int")
            ,("Long",       "Long")
            ,("Short",      "Short")
            ,("CudaByte",   "CudaByte")
            ,("CudaChar",   "CudaChar")
            ,("CudaDouble", "CudaDouble")
            ,("CudaFloat",  "Cuda") -- This is the odd one out
            ,("CudaHalf",   "CudaHalf")
            ,("CudaInt",    "CudaInt")
            ,("CudaLong",   "CudaLong")
            ,("CudaShort",  "CudaShort")]

accurateScalarMap :: Map Text Text
accurateScalarMap = [("Byte",        "Int64")
                    ,("Char",        "Int64")
                    ,("Double",      "CDouble")
                    ,("Float",       "CDouble")
                    ,("Half",        "Int64")
                    ,("Int",         "Int64")
                    ,("Long",        "Int64")
                    ,("Short",       "Int64")
                    ,("CudaByte",    "Int64")
                    ,("CudaChar",    "Int64")
                    ,("CudaDouble",  "CDouble")
                    ,("CudaFloat",   "CDouble")
                    ,("CudaHalf",    "Int64")
                    ,("CudaInt",     "Int64")
                    ,("CudaLong",    "Int64")
                    ,("CudaShort",   "Int64")]

caccurateScalarMap :: Map Text Text
caccurateScalarMap = [("Byte",       "int64_t")
                     ,("Char",       "int64_t")
                     ,("Double",     "double")
                     ,("Float",      "double")
                     ,("Half",       "int64_t")
                     ,("Int",        "int64_t")
                     ,("Long",       "int64_t")
                     ,("Short",      "int64_t")
                     ,("CudaByte",   "int64_t")
                     ,("CudaChar",   "int64_t")
                     ,("CudaDouble", "double")
                     ,("CudaFloat",  "double")
                     ,("CudaHalf",   "int64_t")
                     ,("CudaInt",    "int64_t")
                     ,("CudaLong",   "int64_t")
                     ,("CudaShort",  "int64_t")]

cscalar s = fromJust $ M.lookup s cscalarMap
cscalar' "Byte"     = "byte"
cscalar' "CudaByte" = "byte"
cscalar' s          = fromJust $ M.lookup s cscalarMap
scalar s = fromJust $ M.lookup s scalarMap
caccuratescalar s = fromJust $ M.lookup s caccurateScalarMap
accuratescalar s = fromJust $ M.lookup s accurateScalarMap
host s = fromJust $ M.lookup s hostMap
ctensor s = fromJust $ M.lookup s ctensorMap
ctensortype s = fromJust $ M.lookup s ctensorMapType
ctensorscalartype s = fromJust $ M.lookup s ctensorMapScalarType
