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
import           Torch.C.Language

C.context C.cppCtx

C.include "<torch/csrc/autograd/variable.h>"
C.include "<torch/csrc/autograd/function.h>"
C.include "<torch/csrc/jit/tracer.h>"
C.include "<torch/csrc/jit/ir.h>"
C.include "<torch/csrc/jit/script/module.h>"
C.include "<torch/csrc/jit/script/slot.h>"

C.using "namespace at"
C.using "namespace torch::autograd"
C.using "namespace torch::jit::tracer"

C.verbatim "using JitNode           = torch::jit::Node;"
C.verbatim "using JitValue          = torch::jit::Value;"
C.verbatim "using JitIValue         = torch::jit::IValue;"
C.verbatim "using JitBlock          = torch::jit::Block;"
C.verbatim "using JitType           = ::c10::Type;"
C.verbatim "using JitAttributeKind  = torch::jit::AttributeKind;"
C.verbatim "using JitScriptModule   = torch::jit::script::Module;"

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

instance Enum Backend where
  toEnum x | x == fromIntegral [C.pure|int { (int)kCPU }|] = BackendCPU
           | x == fromIntegral [C.pure|int { (int)kCUDA }|] = BackendCUDA
           | otherwise = error "Cannot convert value to enum"
  fromEnum BackendCPU  = fromIntegral [C.pure|int { (int)kCPU }|]
  fromEnum BackendCUDA = fromIntegral [C.pure|int { (int)kCUDA }|]

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

instance Enum ScalarType where
  toEnum x | x == fromIntegral [C.pure|int { (int)ScalarType::Bool }|] = ScalarTypeBool
           | x == fromIntegral [C.pure|int { (int)ScalarType::Byte }|] = ScalarTypeByte
           | x == fromIntegral [C.pure|int { (int)ScalarType::Char }|] = ScalarTypeChar
           | x == fromIntegral [C.pure|int { (int)ScalarType::Short }|] = ScalarTypeShort
           | x == fromIntegral [C.pure|int { (int)ScalarType::Int }|] = ScalarTypeInt
           | x == fromIntegral [C.pure|int { (int)ScalarType::Long }|] = ScalarTypeLong
           | x == fromIntegral [C.pure|int { (int)ScalarType::Half }|] = ScalarTypeHalf
           | x == fromIntegral [C.pure|int { (int)ScalarType::Float }|] = ScalarTypeFloat
           | x == fromIntegral [C.pure|int { (int)ScalarType::Double }|] = ScalarTypeDouble
           | x == fromIntegral [C.pure|int { (int)ScalarType::Undefined }|] = ScalarTypeUndefined
           | otherwise = error $ "Cannot convert ScalarType value to enum " ++ show x
  fromEnum ScalarTypeBool      = fromIntegral [C.pure|int { (int)ScalarType::Bool }|]
  fromEnum ScalarTypeByte      = fromIntegral [C.pure|int { (int)ScalarType::Byte }|]
  fromEnum ScalarTypeChar      = fromIntegral [C.pure|int { (int)ScalarType::Char }|]
  fromEnum ScalarTypeShort     = fromIntegral [C.pure|int { (int)ScalarType::Short }|]
  fromEnum ScalarTypeInt       = fromIntegral [C.pure|int { (int)ScalarType::Int }|]
  fromEnum ScalarTypeLong      = fromIntegral [C.pure|int { (int)ScalarType::Long }|]
  fromEnum ScalarTypeHalf      = fromIntegral [C.pure|int { (int)ScalarType::Half }|]
  fromEnum ScalarTypeFloat     = fromIntegral [C.pure|int { (int)ScalarType::Float }|]
  fromEnum ScalarTypeDouble    = fromIntegral [C.pure|int { (int)ScalarType::Double }|]
  fromEnum ScalarTypeUndefined = fromIntegral [C.pure|int { (int)ScalarType::Undefined }|]

data TypeKind = TypeKindTensor
              | TypeKindDimensionedTensor
              | TypeKindCompleteTensor
              | TypeKindAutogradZeroTensor
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
              | TypeKindProfiledTensor
              | TypeKindDeviceObj
              | TypeKindFunction
              | TypeKindClass
             deriving (Show, Eq)

instance Enum TypeKind where
  toEnum x | x == fromIntegral [C.pure|int { (int)TypeKind::TensorType }|] = TypeKindTensor
           | x == fromIntegral [C.pure|int { (int)TypeKind::DimensionedTensorType }|] = TypeKindDimensionedTensor
           | x == fromIntegral [C.pure|int { (int)TypeKind::CompleteTensorType }|] = TypeKindCompleteTensor
           | x == fromIntegral [C.pure|int { (int)TypeKind::AutogradZeroTensorType }|] = TypeKindAutogradZeroTensor
           | x == fromIntegral [C.pure|int { (int)TypeKind::TupleType }|] = TypeKindTuple
           | x == fromIntegral [C.pure|int { (int)TypeKind::ListType }|] = TypeKindList
           | x == fromIntegral [C.pure|int { (int)TypeKind::DictType }|] = TypeKindDict
           | x == fromIntegral [C.pure|int { (int)TypeKind::NumberType }|] = TypeKindNumber
           | x == fromIntegral [C.pure|int { (int)TypeKind::FloatType }|] = TypeKindFloat
           | x == fromIntegral [C.pure|int { (int)TypeKind::FutureType }|] = TypeKindFuture
           | x == fromIntegral [C.pure|int { (int)TypeKind::IntType }|] = TypeKindInt
           | x == fromIntegral [C.pure|int { (int)TypeKind::NoneType }|] = TypeKindNone
           | x == fromIntegral [C.pure|int { (int)TypeKind::StringType }|] = TypeKindString
           | x == fromIntegral [C.pure|int { (int)TypeKind::GeneratorType }|] = TypeKindGenerator
           | x == fromIntegral [C.pure|int { (int)TypeKind::BoolType }|] = TypeKindBool
           | x == fromIntegral [C.pure|int { (int)TypeKind::OptionalType }|] = TypeKindOptional
           | x == fromIntegral [C.pure|int { (int)TypeKind::VarType }|] = TypeKindVar
           | x == fromIntegral [C.pure|int { (int)TypeKind::ProfiledTensorType }|] = TypeKindProfiledTensor
           | x == fromIntegral [C.pure|int { (int)TypeKind::DeviceObjType }|] = TypeKindDeviceObj
           | x == fromIntegral [C.pure|int { (int)TypeKind::FunctionType }|] = TypeKindFunction
           | x == fromIntegral [C.pure|int { (int)TypeKind::ClassType }|] = TypeKindClass
           | otherwise = error "Cannot convert TypeKind to enum"
  fromEnum TypeKindTensor             = fromIntegral [C.pure|int { (int)TypeKind::TensorType }|]
  fromEnum TypeKindDimensionedTensor  = fromIntegral [C.pure|int { (int)TypeKind::DimensionedTensorType }|]
  fromEnum TypeKindCompleteTensor     = fromIntegral [C.pure|int { (int)TypeKind::CompleteTensorType }|]
  fromEnum TypeKindAutogradZeroTensor = fromIntegral [C.pure|int { (int)TypeKind::AutogradZeroTensorType }|]
  fromEnum TypeKindTuple              = fromIntegral [C.pure|int { (int)TypeKind::TupleType }|]
  fromEnum TypeKindList               = fromIntegral [C.pure|int { (int)TypeKind::ListType }|]
  fromEnum TypeKindDict               = fromIntegral [C.pure|int { (int)TypeKind::DictType }|]
  fromEnum TypeKindNumber             = fromIntegral [C.pure|int { (int)TypeKind::NumberType }|]
  fromEnum TypeKindFloat              = fromIntegral [C.pure|int { (int)TypeKind::FloatType }|]
  fromEnum TypeKindFuture             = fromIntegral [C.pure|int { (int)TypeKind::FutureType }|]
  fromEnum TypeKindInt                = fromIntegral [C.pure|int { (int)TypeKind::IntType }|]
  fromEnum TypeKindNone               = fromIntegral [C.pure|int { (int)TypeKind::NoneType }|]
  fromEnum TypeKindString             = fromIntegral [C.pure|int { (int)TypeKind::StringType }|]
  fromEnum TypeKindGenerator          = fromIntegral [C.pure|int { (int)TypeKind::GeneratorType }|]
  fromEnum TypeKindBool               = fromIntegral [C.pure|int { (int)TypeKind::BoolType }|]
  fromEnum TypeKindOptional           = fromIntegral [C.pure|int { (int)TypeKind::OptionalType }|]
  fromEnum TypeKindVar                = fromIntegral [C.pure|int { (int)TypeKind::VarType }|]
  fromEnum TypeKindProfiledTensor     = fromIntegral [C.pure|int { (int)TypeKind::ProfiledTensorType }|]
  fromEnum TypeKindDeviceObj          = fromIntegral [C.pure|int { (int)TypeKind::DeviceObjType }|]
  fromEnum TypeKindFunction           = fromIntegral [C.pure|int { (int)TypeKind::FunctionType }|]
  fromEnum TypeKindClass              = fromIntegral [C.pure|int { (int)TypeKind::ClassType }|]

data Reduction = ReductionNone
               | ReductionMean
               | ReductionSum
             deriving (Show, Eq)

instance Enum Reduction where
  toEnum x | x == fromIntegral [C.pure|int { (int)Reduction::None }|] = ReductionNone
           | x == fromIntegral [C.pure|int { (int)Reduction::Mean }|] = ReductionMean
           | x == fromIntegral [C.pure|int { (int)Reduction::Sum }|] = ReductionSum
           | otherwise = error "Cannot convert Reduction to enum"
  fromEnum ReductionNone = fromIntegral [C.pure|int { (int)Reduction::None }|]
  fromEnum ReductionMean = fromIntegral [C.pure|int { (int)Reduction::Mean }|]
  fromEnum ReductionSum  = fromIntegral [C.pure|int { (int)Reduction::Sum }|]

data MemoryFormat = MemoryFormatContiguous
                  | MemoryFormatPreserve
                  | MemoryFormatChannelsLast
             deriving (Show, Eq)

instance Enum MemoryFormat where
  toEnum x | x == fromIntegral [C.pure|int { (int)MemoryFormat::Contiguous }|]   = MemoryFormatContiguous
           | x == fromIntegral [C.pure|int { (int)MemoryFormat::Preserve }|]     = MemoryFormatPreserve
           | x == fromIntegral [C.pure|int { (int)MemoryFormat::ChannelsLast }|] = MemoryFormatChannelsLast
           | otherwise = error "Cannot convert MemoryFormat to enum"
  fromEnum MemoryFormatContiguous   = fromIntegral [C.pure|int { (int)MemoryFormat::Contiguous }|]
  fromEnum MemoryFormatPreserve     = fromIntegral [C.pure|int { (int)MemoryFormat::Preserve }|]
  fromEnum MemoryFormatChannelsLast = fromIntegral [C.pure|int { (int)MemoryFormat::ChannelsLast }|]

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

instance Enum AttributeKind where
  toEnum x | x == fromIntegral [C.pure|int { (int)JitAttributeKind::f }|]  = AttributeKindFloat
           | x == fromIntegral [C.pure|int { (int)JitAttributeKind::fs }|] = AttributeKindFloats
           | x == fromIntegral [C.pure|int { (int)JitAttributeKind::i }|]  = AttributeKindInt
           | x == fromIntegral [C.pure|int { (int)JitAttributeKind::is }|] = AttributeKindInts
           | x == fromIntegral [C.pure|int { (int)JitAttributeKind::s }|]  = AttributeKindString
           | x == fromIntegral [C.pure|int { (int)JitAttributeKind::ss }|] = AttributeKindStrings
           | x == fromIntegral [C.pure|int { (int)JitAttributeKind::t }|]  = AttributeKindTensor
           | x == fromIntegral [C.pure|int { (int)JitAttributeKind::ts }|] = AttributeKindTensors
           | x == fromIntegral [C.pure|int { (int)JitAttributeKind::g }|]  = AttributeKindGraph
           | x == fromIntegral [C.pure|int { (int)JitAttributeKind::gs }|] = AttributeKindGraphs
           | otherwise = error "Cannot convert AttributeKind to enum"
  fromEnum AttributeKindFloat   = fromIntegral [C.pure|int { (int)JitAttributeKind::f }|]
  fromEnum AttributeKindFloats  = fromIntegral [C.pure|int { (int)JitAttributeKind::fs }|]
  fromEnum AttributeKindInt     = fromIntegral [C.pure|int { (int)JitAttributeKind::i }|]
  fromEnum AttributeKindInts    = fromIntegral [C.pure|int { (int)JitAttributeKind::is }|]
  fromEnum AttributeKindString  = fromIntegral [C.pure|int { (int)JitAttributeKind::s }|]
  fromEnum AttributeKindStrings = fromIntegral [C.pure|int { (int)JitAttributeKind::ss }|]
  fromEnum AttributeKindTensor  = fromIntegral [C.pure|int { (int)JitAttributeKind::t }|]
  fromEnum AttributeKindTensors = fromIntegral [C.pure|int { (int)JitAttributeKind::ts }|]
  fromEnum AttributeKindGraph   = fromIntegral [C.pure|int { (int)JitAttributeKind::g }|]
  fromEnum AttributeKindGraphs  = fromIntegral [C.pure|int { (int)JitAttributeKind::gs }|]

data ModuleEntityType = ModuleEntityType
                      | ParameterEntityType
                      | AttributeEntityType
                      | MethodEntityType
             deriving (Show, Eq)

instance Enum ModuleEntityType where
  toEnum x | x == fromIntegral [C.pure|int { (int)torch::jit::script::EntityType::MODULE }|]  = ModuleEntityType
           | x == fromIntegral [C.pure|int { (int)torch::jit::script::EntityType::PARAMETER }|] = ParameterEntityType
           | x == fromIntegral [C.pure|int { (int)torch::jit::script::EntityType::ATTRIBUTE }|] = AttributeEntityType
           | x == fromIntegral [C.pure|int { (int)torch::jit::script::EntityType::METHOD }|] = MethodEntityType
           | otherwise = error "Cannot convert ModuleEntityType to enum"
  fromEnum ModuleEntityType    = fromIntegral [C.pure|int { (int)torch::jit::script::EntityType::MODULE }|]
  fromEnum ParameterEntityType = fromIntegral [C.pure|int { (int)torch::jit::script::EntityType::PARAMETER }|]
  fromEnum AttributeEntityType = fromIntegral [C.pure|int { (int)torch::jit::script::EntityType::ATTRIBUTE }|]
  fromEnum MethodEntityType    = fromIntegral [C.pure|int { (int)torch::jit::script::EntityType::METHOD }|]

cstorable ''CTensorOptions            "TensorOptions"
cstorable ''CVariable                 "Variable"
cstorable ''CTensor                   "Tensor"
cstorable ''CScalar                   "Scalar"
cstorable ''CStorage                  "Storage"
cstorable ''CGenerator                "Generator"
cstorable ''CNode                     "Node"
cstorable ''CDevice                   "Device"
cstorable ''CJitNode                  "JitNode"
cstorable ''CJitValue                 "JitValue"
cstorable ''CJitIValue                "JitIValue"
cstorable ''CJitBlock                 "JitBlock"
cstorable ''CType                     "JitType"
cstorable ''CJitScriptModule          "JitScriptModule"
