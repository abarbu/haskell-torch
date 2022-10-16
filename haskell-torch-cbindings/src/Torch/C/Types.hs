{-# LANGUAGE OverloadedStrings, QuasiQuotes, ScopedTypeVariables, TemplateHaskell #-}

-- | All of the types we need to talk to PyTorch. These will be wrapped and not
-- shown to users.
module Torch.C.Types where
import qualified Data.Map                  as Map
import           Data.Monoid               (mempty, (<>))
import           Foreign.C.Types
import qualified Language.C.Inline         as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Inline.Cpp     as C
import qualified Language.C.Types          as C
import qualified Language.Haskell.TH       as TH

C.context C.cppCtx

data CStorage
data CGenerator
data CVariableType
data CType
data CDevice
data CVariable

-- Tensors
data CTensorOptions
data CTensor
data CScalar

-- Jit
data CTracingState
data CGraph
data CEdge
data CNode
data CJitNode
data CJitValue
data CJitIValue
data CJitBlock
data CJitAttributeKind
data CJitScriptModule

tensorCtx :: C.Context
tensorCtx = C.cppCtx <> C.funCtx <> C.vecCtx <> C.fptrCtx <> ctx
  where ctx = mempty
          { C.ctxTypesTable = tensorTypesTable }

tensorTypesTable :: Map.Map C.TypeSpecifier TH.TypeQ
tensorTypesTable = Map.fromList
  [ (C.TypeName "bool", [t| C.CBool |])
  -- tensors
  , (C.TypeName "TensorOptions", [t| CTensorOptions |])
  , (C.TypeName "Tensor", [t| CTensor |])
  , (C.TypeName "Scalar", [t| CScalar |])
  , (C.TypeName "Storage", [t| CStorage |])
  , (C.TypeName "Generator", [t| CGenerator |])
  , (C.TypeName "JitType", [t| CType |])
  , (C.TypeName "Device", [t| CDevice |])
  -- variables
  , (C.TypeName "VariableType", [t| CVariableType|])
  , (C.TypeName "Variable", [t| CVariable |])
  , (C.TypeName "Edge", [t| CEdge |])
  , (C.TypeName "TracingState", [t| CTracingState |])
  , (C.TypeName "Graph", [t| CGraph |])
  , (C.TypeName "Node", [t| CNode |])
  , (C.TypeName "JitNode", [t| CJitNode |])
  , (C.TypeName "JitValue", [t| CJitValue |])
  , (C.TypeName "JitIValue", [t| CJitIValue |])
  , (C.TypeName "JitBlock", [t| CJitBlock |])
  , (C.TypeName "JitAttributeKind", [t| CJitAttributeKind |])
  , (C.TypeName "JitScriptModule", [t| CJitScriptModule |])
  ]

data Backend = BackendCPU
             | BackendCUDA
             deriving (Show, Eq)

data Layout = LayoutStrided
            | LayoutSparse
            | LayoutMlkdnn
             deriving (Show, Eq)

data ScalarType = ScalarTypeBool
                | ScalarTypeByte
                | ScalarTypeChar
                | ScalarTypeShort
                | ScalarTypeInt
                | ScalarTypeLong
                | ScalarTypeHalf
                | ScalarTypeFloat
                | ScalarTypeDouble
                | ScalarTypeUndefined
                deriving (Show, Eq, Ord)

data TypeKind = TypeKindAny
              | TypeKindTensor
              | TypeKindTuple
              | TypeKindList
              | TypeKindDict
              | TypeKindNumber
              | TypeKindFloat
              | TypeKindFuture
              | TypeKindInt
              | TypeKindNone
              | TypeKindString
              | TypeKindGenerator
              | TypeKindBool
              | TypeKindOptional
              | TypeKindVar
              | TypeKindDeviceObj
              | TypeKindFunction
              | TypeKindClass
              | TypeKindCapsule
              | TypeKindInterface
             deriving (Show, Eq)

data Reduction = ReductionNone
               | ReductionMean
               | ReductionSum
             deriving (Show, Eq)

data MemoryFormat = MemoryFormatContiguous
                  | MemoryFormatPreserve
                  | MemoryFormatChannelsLast
                  | MemoryFormatChannelsLast3d
             deriving (Show, Eq)

data AttributeKind = AttributeKindFloat
                   | AttributeKindFloats
                   | AttributeKindInt
                   | AttributeKindInts
                   | AttributeKindString
                   | AttributeKindStrings
                   | AttributeKindTensor
                   | AttributeKindTensors
                   | AttributeKindGraph
                   | AttributeKindGraphs
             deriving (Show, Eq)

data ModuleEntityType = ModuleEntityType
                      | ParameterEntityType
                      | MethodEntityType
             deriving (Show, Eq)
