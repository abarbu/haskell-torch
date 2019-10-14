{-# LANGUAGE OverloadedStrings, QuasiQuotes, ScopedTypeVariables, TemplateHaskell #-}

module Foreign.Matio.Types where
import qualified Data.Map                         as Map
import           Data.Monoid                      (mempty, (<>))
import           Data.Text                        (Text)
import qualified Data.Text                        as T
import qualified Data.Vector                      as V'
import           Data.Vector.Storable             (Vector)
import qualified Data.Vector.Storable             as V
import           Foreign.C.Types
import qualified Language.C.Inline                as C
import qualified Language.C.Inline.Context        as C
import qualified Language.C.Inline.Cpp            as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Types                 as C
import qualified Language.Haskell.TH              as TH

C.include "<matio.h>"

data CMat
data CMatVar

matioCtx :: C.Context
matioCtx = C.baseCtx <> C.funCtx <> C.vecCtx <> C.fptrCtx <> ctx
  where ctx = mempty
          { C.ctxTypesTable = matioTypesTable }

matioTypesTable :: Map.Map C.TypeSpecifier TH.TypeQ
matioTypesTable = Map.fromList
  [ (C.TypeName "bool", [t| C.CBool |])
  , (C.TypeName "mat_t", [t| CMat |])
  , (C.TypeName "matvar_t", [t| CMatVar |])
  ]

-- | NB This structure is incomplete but it is also not meant by a 1-1 map between
-- matlab datatypes and haskell datatypes. For example, you get one text data
-- type, you don't get to redundantly pick between 3.
--
-- NB Matlab arrays are unfortunately column major! :(
data MatValue = MatCellArray (Vector CSize) (V'.Vector MatValue)
              | MatInt8Array (Vector CSize) (Vector CChar)
              | MatUInt8Array (Vector CSize) (Vector CUChar)
              | MatText Text
              deriving (Show, Eq)

data MatType = MatTypeInt8
             | MatTypeUInt8
             | MatTypeInt16
             | MatTypeUInt16
             | MatTypeInt32
             | MatTypeUInt32
             | MatTypeInt64
             | MatTypeUInt64
             | MatTypeFloat
             | MatTypeDouble
             | MatTypeMatrix
             | MatTypeCompressed
             | MatTypeUTF8
             | MatTypeUTF16
             | MatTypeUTF32
             | MatTypeString
             | MatTypeCell
             | MatTypeStruct
             | MatTypeArray
             | MatTypeFunction
             | MatTypeUnknown
             deriving (Show, Eq)

instance Enum MatType where
  toEnum x | x == fromIntegral [C.pure| int { (int)MAT_T_INT8 }|]       = MatTypeInt8
           | x == fromIntegral [C.pure| int { (int)MAT_T_UINT8 }|]      = MatTypeUInt8
           | x == fromIntegral [C.pure| int { (int)MAT_T_INT16 }|]      = MatTypeInt16
           | x == fromIntegral [C.pure| int { (int)MAT_T_UINT16 }|]     = MatTypeUInt16
           | x == fromIntegral [C.pure| int { (int)MAT_T_INT32 }|]      = MatTypeInt32
           | x == fromIntegral [C.pure| int { (int)MAT_T_UINT32 }|]     = MatTypeUInt32
           | x == fromIntegral [C.pure| int { (int)MAT_T_INT64 }|]      = MatTypeInt64
           | x == fromIntegral [C.pure| int { (int)MAT_T_UINT64 }|]     = MatTypeUInt64
           | x == fromIntegral [C.pure| int { (int)MAT_T_SINGLE }|]     = MatTypeFloat
           | x == fromIntegral [C.pure| int { (int)MAT_T_DOUBLE }|]     = MatTypeDouble
           | x == fromIntegral [C.pure| int { (int)MAT_T_MATRIX }|]     = MatTypeMatrix
           | x == fromIntegral [C.pure| int { (int)MAT_T_COMPRESSED }|] = MatTypeCompressed
           | x == fromIntegral [C.pure| int { (int)MAT_T_UTF8 }|]       = MatTypeUTF8
           | x == fromIntegral [C.pure| int { (int)MAT_T_UTF16 }|]      = MatTypeUTF16
           | x == fromIntegral [C.pure| int { (int)MAT_T_UTF32 }|]      = MatTypeUTF32
           | x == fromIntegral [C.pure| int { (int)MAT_T_STRING }|]     = MatTypeString
           | x == fromIntegral [C.pure| int { (int)MAT_T_CELL }|]       = MatTypeCell
           | x == fromIntegral [C.pure| int { (int)MAT_T_STRUCT }|]     = MatTypeStruct
           | x == fromIntegral [C.pure| int { (int)MAT_T_ARRAY }|]      = MatTypeArray
           | x == fromIntegral [C.pure| int { (int)MAT_T_FUNCTION }|]   = MatTypeFunction
           | x == fromIntegral [C.pure| int { (int)MAT_T_UNKNOWN }|]    = MatTypeUnknown
  fromEnum MatTypeInt8       = fromIntegral [C.pure|int { (int)MAT_T_INT8 }|]
  fromEnum MatTypeUInt8      = fromIntegral [C.pure|int { (int)MAT_T_UINT8 }|]
  fromEnum MatTypeInt16      = fromIntegral [C.pure|int { (int)MAT_T_INT16 }|]
  fromEnum MatTypeUInt16     = fromIntegral [C.pure|int { (int)MAT_T_UINT16 }|]
  fromEnum MatTypeInt32      = fromIntegral [C.pure|int { (int)MAT_T_INT32 }|]
  fromEnum MatTypeUInt32     = fromIntegral [C.pure|int { (int)MAT_T_UINT32 }|]
  fromEnum MatTypeInt64      = fromIntegral [C.pure|int { (int)MAT_T_INT64 }|]
  fromEnum MatTypeUInt64     = fromIntegral [C.pure|int { (int)MAT_T_UINT64 }|]
  fromEnum MatTypeFloat      = fromIntegral [C.pure|int { (int)MAT_T_SINGLE }|]
  fromEnum MatTypeDouble     = fromIntegral [C.pure|int { (int)MAT_T_DOUBLE }|]
  fromEnum MatTypeMatrix     = fromIntegral [C.pure|int { (int)MAT_T_MATRIX }|]
  fromEnum MatTypeCompressed = fromIntegral [C.pure|int { (int)MAT_T_COMPRESSED }|]
  fromEnum MatTypeUTF8       = fromIntegral [C.pure|int { (int)MAT_T_UTF8 }|]
  fromEnum MatTypeUTF16      = fromIntegral [C.pure|int { (int)MAT_T_UTF16 }|]
  fromEnum MatTypeUTF32      = fromIntegral [C.pure|int { (int)MAT_T_UTF32 }|]
  fromEnum MatTypeString     = fromIntegral [C.pure|int { (int)MAT_T_STRING }|]
  fromEnum MatTypeCell       = fromIntegral [C.pure|int { (int)MAT_T_CELL }|]
  fromEnum MatTypeStruct     = fromIntegral [C.pure|int { (int)MAT_T_STRUCT }|]
  fromEnum MatTypeArray      = fromIntegral [C.pure|int { (int)MAT_T_ARRAY }|]
  fromEnum MatTypeFunction   = fromIntegral [C.pure|int { (int)MAT_T_FUNCTION }|]
  fromEnum MatTypeUnknown    = fromIntegral [C.pure|int { (int)MAT_T_UNKNOWN }|]

data MatClass = MatClassEmpty
              | MatClassCell
              | MatClassStruct
              | MatClassObject
              | MatClassChar
              | MatClassSparse
              | MatClassDouble
              | MatClassFloat
              | MatClassInt8
              | MatClassUInt8
              | MatClassInt16
              | MatClassUInt16
              | MatClassInt32
              | MatClassUInt32
              | MatClassInt64
              | MatClassUInt64
              | MatClassFunction
              | MatClassOpaque
              deriving (Show, Eq)

instance Enum MatClass where
  toEnum x | x == fromIntegral [C.pure| int { (int) MAT_C_EMPTY }|]     = MatClassEmpty
           | x == fromIntegral [C.pure| int { (int) MAT_C_CELL }|]      = MatClassCell
           | x == fromIntegral [C.pure| int { (int) MAT_C_STRUCT }|]    = MatClassStruct
           | x == fromIntegral [C.pure| int { (int) MAT_C_OBJECT }|]    = MatClassObject
           | x == fromIntegral [C.pure| int { (int) MAT_C_CHAR }|]      = MatClassChar
           | x == fromIntegral [C.pure| int { (int) MAT_C_SPARSE }|]    = MatClassSparse
           | x == fromIntegral [C.pure| int { (int) MAT_C_DOUBLE }|]    = MatClassDouble
           | x == fromIntegral [C.pure| int { (int) MAT_C_SINGLE }|]    = MatClassFloat
           | x == fromIntegral [C.pure| int { (int) MAT_C_INT8 }|]      = MatClassInt8
           | x == fromIntegral [C.pure| int { (int) MAT_C_UINT8 }|]     = MatClassUInt8
           | x == fromIntegral [C.pure| int { (int) MAT_C_INT16 }|]     = MatClassInt16
           | x == fromIntegral [C.pure| int { (int) MAT_C_UINT16 }|]    = MatClassUInt16
           | x == fromIntegral [C.pure| int { (int) MAT_C_INT32 }|]     = MatClassInt32
           | x == fromIntegral [C.pure| int { (int) MAT_C_UINT32 }|]    = MatClassUInt32
           | x == fromIntegral [C.pure| int { (int) MAT_C_INT64 }|]     = MatClassInt64
           | x == fromIntegral [C.pure| int { (int) MAT_C_UINT64 }|]    = MatClassUInt64
           | x == fromIntegral [C.pure| int { (int) MAT_C_FUNCTION }|]  = MatClassFunction
           | x == fromIntegral [C.pure| int { (int) MAT_C_OPAQUE }|]    = MatClassOpaque
  fromEnum MatClassEmpty    = fromIntegral [C.pure|int { (int)MAT_C_EMPTY }|]
  fromEnum MatClassCell     = fromIntegral [C.pure|int { (int)MAT_C_CELL }|]
  fromEnum MatClassStruct   = fromIntegral [C.pure|int { (int)MAT_C_STRUCT }|]
  fromEnum MatClassObject   = fromIntegral [C.pure|int { (int)MAT_C_OBJECT }|]
  fromEnum MatClassChar     = fromIntegral [C.pure|int { (int)MAT_C_CHAR }|]
  fromEnum MatClassSparse   = fromIntegral [C.pure|int { (int)MAT_C_SPARSE }|]
  fromEnum MatClassDouble   = fromIntegral [C.pure|int { (int)MAT_C_DOUBLE }|]
  fromEnum MatClassFloat    = fromIntegral [C.pure|int { (int)MAT_C_SINGLE }|]
  fromEnum MatClassInt8     = fromIntegral [C.pure|int { (int)MAT_C_INT8 }|]
  fromEnum MatClassUInt8    = fromIntegral [C.pure|int { (int)MAT_C_UINT8 }|]
  fromEnum MatClassInt16    = fromIntegral [C.pure|int { (int)MAT_C_INT16 }|]
  fromEnum MatClassUInt16   = fromIntegral [C.pure|int { (int)MAT_C_UINT16 }|]
  fromEnum MatClassInt32    = fromIntegral [C.pure|int { (int)MAT_C_INT32 }|]
  fromEnum MatClassUInt32   = fromIntegral [C.pure|int { (int)MAT_C_UINT32 }|]
  fromEnum MatClassInt64    = fromIntegral [C.pure|int { (int)MAT_C_INT64 }|]
  fromEnum MatClassUInt64   = fromIntegral [C.pure|int { (int)MAT_C_UINT64 }|]
  fromEnum MatClassFunction = fromIntegral [C.pure|int { (int)MAT_C_FUNCTION }|]
  fromEnum MatClassOpaque   = fromIntegral [C.pure|int { (int)MAT_C_OPAQUE }|]
