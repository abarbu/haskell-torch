{-# LANGUAGE FlexibleContexts, FlexibleInstances, OverloadedStrings, QuasiQuotes, ScopedTypeVariables, TemplateHaskell #-}

-- | Bindings to ATen's VariableType API for dynamic tensors. This adds extra
-- layers of indirection that could be removed since we know the types in
-- Haskell but the overhead is so small it's irrelevant.

module Torch.C.Tensor where
import           Data.Int
import           Data.Monoid           ((<>))
import           Data.Vector.Storable  (Vector)
import qualified Data.Vector.Storable  as V
import           Data.Word
import           Foreign.C.String
import           Foreign.C.Types
import           Foreign.ForeignPtr
import           Foreign.Marshal.Alloc
import           Foreign.Ptr
import qualified Language.C.Inline     as C
import qualified Language.C.Inline.Cpp as C
import           Prelude               hiding (max, min)
import           System.IO.Unsafe
import           Torch.C.Types
import           Torch.C.Language

C.context (C.cppCtx <> tensorCtx)

C.include "<stdexcept>"
C.include "<iostream>"
C.include "<string.h>"

C.include "<torch/csrc/autograd/variable.h>"
C.include "<torch/csrc/autograd/function.h>"
C.include "<algorithm>"
C.include "<sstream>"

C.include "<torch/csrc/autograd/engine.h>"
C.include "<torch/csrc/autograd/grad_mode.h>"

C.include "<torch/csrc/jit/frontend/tracer.h>"
C.include "<torch/csrc/jit/ir/ir.h>"
C.include "<torch/csrc/jit/api/module.h>"
C.include "<c10/core/MemoryFormat.h>"

C.include "<ATen/ArrayRef.h>"
C.include "<torch/csrc/autograd/generated/VariableType.h>"
-- C.include "<torch/csrc/autograd/generated/VariableTypeExtras.h>"

C.using "namespace at"
C.using "namespace torch::autograd"
C.using "namespace torch::jit::tracer"

C.verbatim "using edge_list = std::vector<torch::autograd::Edge>;"

C.verbatim "using JitNode           = torch::jit::Node;"
C.verbatim "using JitValue          = torch::jit::Value;"
C.verbatim "using JitIValue         = torch::jit::IValue;"
C.verbatim "using JitBlock          = torch::jit::Block;"
C.verbatim "using JitType           = ::c10::Type;"
C.verbatim "using JitAttributeKind  = torch::jit::AttributeKind;"
C.verbatim "using JitScriptModule   = torch::jit::script::Module;"

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

instance Enum TypeKind where
  toEnum x | x == fromIntegral [C.pure|int { (int)TypeKind::AnyType }|] = TypeKindAny
           | x == fromIntegral [C.pure|int { (int)TypeKind::TensorType }|] = TypeKindTensor
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
           | x == fromIntegral [C.pure|int { (int)TypeKind::DeviceObjType }|] = TypeKindDeviceObj
           | x == fromIntegral [C.pure|int { (int)TypeKind::FunctionType }|] = TypeKindFunction
           | x == fromIntegral [C.pure|int { (int)TypeKind::ClassType }|] = TypeKindClass
           | x == fromIntegral [C.pure|int { (int)TypeKind::CapsuleType }|] = TypeKindCapsule
           | x == fromIntegral [C.pure|int { (int)TypeKind::InterfaceType }|] = TypeKindInterface
           | otherwise = error "Cannot convert TypeKind to enum"
  fromEnum TypeKindAny                = fromIntegral [C.pure|int { (int)TypeKind::AnyType }|]
  fromEnum TypeKindTensor             = fromIntegral [C.pure|int { (int)TypeKind::TensorType }|]
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
  fromEnum TypeKindDeviceObj          = fromIntegral [C.pure|int { (int)TypeKind::DeviceObjType }|]
  fromEnum TypeKindFunction           = fromIntegral [C.pure|int { (int)TypeKind::FunctionType }|]
  fromEnum TypeKindClass              = fromIntegral [C.pure|int { (int)TypeKind::ClassType }|]
  fromEnum TypeKindCapsule            = fromIntegral [C.pure|int { (int)TypeKind::CapsuleType }|]
  fromEnum TypeKindInterface          = fromIntegral [C.pure|int { (int)TypeKind::InterfaceType }|]

instance Enum Reduction where
  toEnum x | x == fromIntegral [C.pure|int { (int)Reduction::None }|] = ReductionNone
           | x == fromIntegral [C.pure|int { (int)Reduction::Mean }|] = ReductionMean
           | x == fromIntegral [C.pure|int { (int)Reduction::Sum }|] = ReductionSum
           | otherwise = error "Cannot convert Reduction to enum"
  fromEnum ReductionNone = fromIntegral [C.pure|int { (int)Reduction::None }|]
  fromEnum ReductionMean = fromIntegral [C.pure|int { (int)Reduction::Mean }|]
  fromEnum ReductionSum  = fromIntegral [C.pure|int { (int)Reduction::Sum }|]

instance Enum MemoryFormat where
  toEnum x | x == fromIntegral [C.pure|int { (int)MemoryFormat::Contiguous }|]   = MemoryFormatContiguous
           | x == fromIntegral [C.pure|int { (int)MemoryFormat::Preserve }|]     = MemoryFormatPreserve
           | x == fromIntegral [C.pure|int { (int)MemoryFormat::ChannelsLast }|] = MemoryFormatChannelsLast
           | otherwise = error "Cannot convert MemoryFormat to enum"
  fromEnum MemoryFormatContiguous     = fromIntegral [C.pure|int { (int)MemoryFormat::Contiguous }|]
  fromEnum MemoryFormatPreserve       = fromIntegral [C.pure|int { (int)MemoryFormat::Preserve }|]
  fromEnum MemoryFormatChannelsLast   = fromIntegral [C.pure|int { (int)MemoryFormat::ChannelsLast }|]

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

instance Enum Backend where
  toEnum x | x == fromIntegral [C.pure|int { (int)kCPU }|] = BackendCPU
           | x == fromIntegral [C.pure|int { (int)kCUDA }|] = BackendCUDA
           | otherwise = error "Cannot convert value to enum"
  fromEnum BackendCPU  = fromIntegral [C.pure|int { (int)kCPU }|]
  fromEnum BackendCUDA = fromIntegral [C.pure|int { (int)kCUDA }|]

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

-- TODO concurrent calls to backward?
backward1 :: ForeignPtr CTensor -> CBool -> CBool -> IO ()
backward1 var keepGraph createGraph = [C.block|void {
   edge_list edgelst;
   Variable *v = (Variable*)$fptr-ptr:(Tensor* var);
   edgelst.emplace_back(v->grad_fn(), v->output_nr());
   variable_list inputs;
   inputs.emplace_back(make_variable(at::ones_like(v->tensor_data()), false));
   Engine::get_default_engine().execute(edgelst, inputs, $(bool keepGraph), $(bool createGraph));
  } |]

-- TODO concurrent calls to backward?
backwardN :: Vector (Ptr CTensor) -> CBool -> CBool -> IO ()
backwardN tensors keepGraph createGraph =
  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in
  [C.block|void {
   edge_list edgelst;
   variable_list inputs;
   Variable ** vs = (Variable **)$(Tensor** tensors__array);
   for(size_t i = 0; i < $(size_t tensors__size); i++) {
     Variable *v = vs[i];
     edgelst.emplace_back(v->grad_fn(), v->output_nr());
     inputs.emplace_back(make_variable(at::ones_like(v->tensor_data()), false));
   }
   Engine::get_default_engine().execute(edgelst, inputs, $(bool keepGraph), $(bool createGraph));
  } |]

toStringT t = [C.exp|char*{(char*) $fptr-ptr:(Tensor *t)->toString().c_str()}|] >>= peekCString

getType :: ForeignPtr CTensor -> IO ScalarType
getType t = toEnum . fromIntegral <$> [C.exp|int{(int) $fptr-ptr:(Tensor *t)->scalar_type() }|]

getScalarTypeString t = [C.exp|char*{(char*) toString($fptr-ptr:(Tensor *t)->scalar_type()) }|]

getTypeString t = do
  s  <- peekCString =<< backendString t
  s' <- peekCString =<< getScalarTypeString t
  pure $ s ++ " " ++ s'

is_contiguous t = [C.exp|bool{$fptr-ptr:(Tensor *t)->is_contiguous() }|]

is_sparse t = [C.exp|bool{$fptr-ptr:(Tensor *t)->is_sparse() }|]

numel t = [C.exp|int64_t{ $fptr-ptr:(Tensor *t)->numel() }|]

backend t = [C.exp|int{(int)$fptr-ptr:(Tensor *t)->options().backend() }|]

backendString t = [C.exp|char*{(char*)toString($fptr-ptr:(Tensor *t)->options().backend()) }|]

device t = [C.exp|int{(int)$fptr-ptr:(Tensor *t)->options().device().type() }|]

debugPrintCType t = [C.exp|void{std::cout << typeid((void*)$fptr-ptr:(Tensor *t)).name() << '\n';}|]

C.verbatim "std::vector<Tensor> pack_tensor_list(Tensor** arr, size_t len) { std::vector<Tensor> v; for(size_t i = 0; i < len; i++) { v.push_back(*(arr[i])); }; return v; }"

C.verbatim "std::array<bool,2> make_array_bool_2(bool *arr) { return std::array<bool,2>{arr[0], arr[1]}; }"
C.verbatim "std::array<bool,3> make_array_bool_3(bool *arr) { return std::array<bool,3>{arr[0], arr[1], arr[2]}; }"
C.verbatim "std::array<bool,4> make_array_bool_4(bool *arr) { return std::array<bool,4>{arr[0], arr[1], arr[2], arr[3]}; }"

-- TODO This is gross..
C.verbatim "extern \"C\" void delete_scalar1(Scalar* o) { delete o; }"
C.verbatim "extern \"C\" void delete_tensor(Tensor* o) { delete o; }"
C.verbatim "extern \"C\" void delete_tensor_storage(Tensor* o) { free(o->data_ptr()); }"
C.verbatim "extern \"C\" void delete_tensor_options(TensorOptions* o) { delete(o); }"

foreign import ccall "&delete_tensor" deleteTensor :: FunPtr (Ptr CTensor -> IO ())
foreign import ccall "&delete_tensor_storage" deleteTensorStorage :: FunPtr (Ptr CTensor -> IO ())
foreign import ccall "&delete_tensor_options" deleteTensorOptions :: FunPtr (Ptr CTensorOptions -> IO ())

-- TODO We should just not export this, but we're not at the stage where we
-- handle export lists yet.
foreign import ccall "&delete_scalar1" deleteScalar' :: FunPtr (Ptr CScalar -> IO ())

undefinedTensor :: IO (ForeignPtr CTensor)
undefinedTensor = [C.block|Tensor* { return new Tensor(); }|] >>= newForeignPtr deleteTensor

splitMaybe :: Maybe a -> a -> (CBool, a)
splitMaybe Nothing def = (fromIntegral 0, def)
splitMaybe (Just v) _  = (fromIntegral 1, v)

data_ptr :: ForeignPtr CTensor -> IO (Ptr ())
data_ptr ten = [C.exp|void *{ $fptr-ptr:(Tensor *ten)->data_ptr() }|]

emptyTensorOptions :: Backend -> ScalarType -> CBool -> IO (ForeignPtr CTensorOptions)
emptyTensorOptions backend scalarType requiresGrad =
  newForeignPtr deleteTensorOptions
   =<< [C.exp|TensorOptions*{
         new TensorOptions(TensorOptions(Device((DeviceType)$(int dt)))
                           .dtype((ScalarType)$(int sc))
                           .requires_grad($(bool requiresGrad)))}|]
  where dt = fromIntegral $ fromEnum backend
        sc = fromIntegral $ fromEnum scalarType

unTupleTensorTensor :: Ptr () -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
unTupleTensorTensor ptr = do
  a <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* { new Tensor(std::get<0>(*(std::tuple<Tensor,Tensor>*)$(void* ptr))) }|]
  b <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* { new Tensor(std::get<1>(*(std::tuple<Tensor,Tensor>*)$(void* ptr))) }|]
  [C.exp|void { delete (std::tuple<Tensor,Tensor>*)$(void* ptr) }|]
  pure (a, b)

unTupleTensorTensorTensor :: Ptr () -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
unTupleTensorTensorTensor ptr = do
  a <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<0>(*(std::tuple<Tensor,Tensor,Tensor>*)$(void* ptr))) }|]
  b <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<1>(*(std::tuple<Tensor,Tensor,Tensor>*)$(void* ptr))) }|]
  c <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<2>(*(std::tuple<Tensor,Tensor,Tensor>*)$(void* ptr))) }|]
  [C.exp|void { delete (std::tuple<Tensor,Tensor,Tensor>*)$(void* ptr) }|]
  pure (a, b, c)

unTupleTensorTensorTensorTensor :: Ptr () -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
unTupleTensorTensorTensorTensor ptr = do
  a <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<0>(*(std::tuple<Tensor,Tensor,Tensor,Tensor>*)$(void* ptr))) }|]
  b <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<1>(*(std::tuple<Tensor,Tensor,Tensor,Tensor>*)$(void* ptr))) }|]
  c <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<2>(*(std::tuple<Tensor,Tensor,Tensor,Tensor>*)$(void* ptr))) }|]
  d <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<3>(*(std::tuple<Tensor,Tensor,Tensor,Tensor>*)$(void* ptr))) }|]
  [C.exp|void { delete (std::tuple<Tensor,Tensor,Tensor,Tensor>*)$(void* ptr) }|]
  pure (a, b, c, d)

unTupleTensorTensorTensorInt64 :: Ptr () -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, Int64)
unTupleTensorTensorTensorInt64 ptr = do
  a <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<0>(*(std::tuple<Tensor,Tensor,Tensor,int64_t>*)$(void* ptr))) }|]
  b <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<1>(*(std::tuple<Tensor,Tensor,Tensor,int64_t>*)$(void* ptr))) }|]
  c <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<2>(*(std::tuple<Tensor,Tensor,Tensor,int64_t>*)$(void* ptr))) }|]
  d <- [C.exp|int64_t {
          std::get<3>(*(std::tuple<Tensor,Tensor,Tensor,int64_t>*)$(void* ptr)) }|]
  [C.exp|void { delete (std::tuple<Tensor,Tensor,Tensor,int64_t>*)$(void* ptr) }|]
  pure (a, b, c, d)

unTupleTensorTensorTensorTensorInt64 :: Ptr () -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, Int64)
unTupleTensorTensorTensorTensorInt64 ptr = do
  a <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<0>(*(std::tuple<Tensor,Tensor,Tensor,int64_t>*)$(void* ptr))) }|]
  b <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<1>(*(std::tuple<Tensor,Tensor,Tensor,int64_t>*)$(void* ptr))) }|]
  c <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<2>(*(std::tuple<Tensor,Tensor,Tensor,int64_t>*)$(void* ptr))) }|]
  d <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<2>(*(std::tuple<Tensor,Tensor,Tensor,int64_t>*)$(void* ptr))) }|]
  e <- [C.exp|int64_t {
          std::get<3>(*(std::tuple<Tensor,Tensor,Tensor,int64_t>*)$(void* ptr)) }|]
  [C.exp|void { delete (std::tuple<Tensor,Tensor,Tensor,int64_t>*)$(void* ptr) }|]
  pure (a, b, c, d, e)

unTupleTensorTensorDoubleInt64 :: Ptr () -> IO (ForeignPtr CTensor, ForeignPtr CTensor, CDouble, Int64)
unTupleTensorTensorDoubleInt64 ptr = do
  a <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<0>(*(std::tuple<Tensor,Tensor,double,int64_t>*)$(void* ptr))) }|]
  b <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<1>(*(std::tuple<Tensor,Tensor,double,int64_t>*)$(void* ptr))) }|]
  c <- [C.exp|double {
          std::get<2>(*(std::tuple<Tensor,Tensor,double,int64_t>*)$(void* ptr)) }|]
  d <- [C.exp|int64_t {
          std::get<3>(*(std::tuple<Tensor,Tensor,double,int64_t>*)$(void* ptr)) }|]
  [C.exp|void { delete (std::tuple<Tensor,Tensor,double,int64_t>*)$(void* ptr) }|]
  pure (a, b, c, d)

unTupleTensorTensorTensorTensorTensor :: Ptr () -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
unTupleTensorTensorTensorTensorTensor ptr = do
  a <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<0>(*(std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>*)$(void* ptr))) }|]
  b <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<1>(*(std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>*)$(void* ptr))) }|]
  c <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<2>(*(std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>*)$(void* ptr))) }|]
  d <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<3>(*(std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>*)$(void* ptr))) }|]
  e <- newForeignPtr deleteTensor
    =<< [C.exp|Tensor* {
          new Tensor(std::get<4>(*(std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>*)$(void* ptr))) }|]
  [C.exp|void { delete (std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>*)$(void* ptr) }|]
  pure (a, b, c, d, e)

unVectorTensor :: Ptr () -> IO (Vector (Ptr CTensor))
unVectorTensor ptr = do
  s <- [C.exp|size_t { ((std::vector<Tensor>*)$(void* ptr))->size() }|]
  r <- V.generateM (fromIntegral s)
    (\i -> let i' = fromIntegral i
          in [C.exp|Tensor *{ new Tensor(((std::vector<Tensor>*)$(void* ptr))->at($(int i'))) }|])
  [C.exp|void { delete ((std::vector<Tensor>*)$(void* ptr)) }|]
  pure r

str :: ForeignPtr CTensor -> IO String
str ten = do
  s <- [C.block|char* {
           std::stringstream s;
           print(s,*$fptr-ptr:(Tensor *ten),80);
           return strdup(s.str().c_str());
           } |]
  s' <- peekCString s
  free s
  pure s'

is_defined :: ForeignPtr CTensor -> IO (CBool)
is_defined self =
  [C.block|bool {
    return $fptr-ptr:(Tensor* self)->defined();
   }|]

tensorFromBlob :: ForeignPtr CTensorOptions -> Ptr () -> Vector Int64 -> IO (ForeignPtr CTensor)
tensorFromBlob ty ptr size = V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.exp|Tensor *{
      new Variable(at::from_blob($(void *ptr),
                              ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)),
                               *$fptr-ptr:(TensorOptions *ty))) }|]
   >>= newForeignPtr deleteTensor

-------------------------------------------------------------------------------
-- Everything below is AUTOGENERATED from generate-ctensor

-- Tensor avg_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad)
--
avg_pool1d__taaabb :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
avg_pool1d__taaabb self kernel_size stride padding ceil_mode count_include_pad =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|Tensor* {
    return new Tensor(at::avg_pool1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor arange(Scalar end, const TensorOptions & options)
--
arange__so :: ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
arange__so end options =  
  [C.block|Tensor* {
    return new Tensor(at::arange(*$fptr-ptr:(Scalar* end), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor arange(Scalar start, Scalar end, const TensorOptions & options)
--
arange__sso :: ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
arange__sso start end options =  
  [C.block|Tensor* {
    return new Tensor(at::arange(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor arange(Scalar start, Scalar end, Scalar step, const TensorOptions & options)
--
arange__ssso :: ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
arange__ssso start end step options =  
  [C.block|Tensor* {
    return new Tensor(at::arange(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(Scalar* step), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bartlett_window(int64_t window_length, const TensorOptions & options)
--
bartlett_window__6o :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
bartlett_window__6o window_length options =  
  [C.block|Tensor* {
    return new Tensor(at::bartlett_window($(int64_t window_length), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bartlett_window(int64_t window_length, bool periodic, const TensorOptions & options)
--
bartlett_window__6bo :: Int64 -> CBool -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
bartlett_window__6bo window_length periodic options =  
  [C.block|Tensor* {
    return new Tensor(at::bartlett_window($(int64_t window_length), $(bool periodic), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction)
--
binary_cross_entropy__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
binary_cross_entropy__ttt6 self target weight reduction =  
  [C.block|Tensor* {
    return new Tensor(at::binary_cross_entropy(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & binary_cross_entropy_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction)
--
binary_cross_entropy_out__tttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
binary_cross_entropy_out__tttt6 out self target weight reduction =  
  [C.block|void {
    at::binary_cross_entropy_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction));
   }|] >> pure (out)


-- Tensor binary_cross_entropy_with_logits(const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction)
--
binary_cross_entropy_with_logits__tttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
binary_cross_entropy_with_logits__tttt6 self target weight pos_weight reduction =  
  [C.block|Tensor* {
    return new Tensor(at::binary_cross_entropy_with_logits(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* pos_weight), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bincount(const Tensor & self, const Tensor & weights, int64_t minlength)
--
bincount__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
bincount__tt6 self weights minlength =  
  [C.block|Tensor* {
    return new Tensor(at::bincount(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weights), $(int64_t minlength)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor blackman_window(int64_t window_length, const TensorOptions & options)
--
blackman_window__6o :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
blackman_window__6o window_length options =  
  [C.block|Tensor* {
    return new Tensor(at::blackman_window($(int64_t window_length), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor blackman_window(int64_t window_length, bool periodic, const TensorOptions & options)
--
blackman_window__6bo :: Int64 -> CBool -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
blackman_window__6bo window_length periodic options =  
  [C.block|Tensor* {
    return new Tensor(at::blackman_window($(int64_t window_length), $(bool periodic), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups)
--
conv1d__tttaaa6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
conv1d__tttaaa6 input weight bias stride padding dilation groups =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::conv1d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups)
--
conv2d__tttaaa6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
conv2d__tttaaa6 input weight bias stride padding dilation groups =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::conv2d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups)
--
conv3d__tttaaa6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
conv3d__tttaaa6 input weight bias stride padding dilation groups =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::conv3d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation)
--
conv_transpose1d__tttaaa6a :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose1d__tttaaa6a input weight bias stride padding output_padding groups dilation =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::conv_transpose1d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv_transpose2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation)
--
conv_transpose2d__tttaaa6a :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose2d__tttaaa6a input weight bias stride padding output_padding groups dilation =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::conv_transpose2d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation)
--
conv_transpose3d__tttaaa6a :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose3d__tttaaa6a input weight bias stride padding output_padding groups dilation =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::conv_transpose3d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor,Tensor,Tensor> embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset)
--
embedding_bag__tttb6btb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> Int64 -> CBool -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
embedding_bag__tttb6btb weight indices offsets scale_grad_by_freq mode sparse per_sample_weights include_last_offset =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,Tensor>(at::embedding_bag(*$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* offsets), $(bool scale_grad_by_freq), $(int64_t mode), $(bool sparse), *$fptr-ptr:(Tensor* per_sample_weights), $(bool include_last_offset)));
   }|] >>= unTupleTensorTensorTensorTensor


-- std::tuple<Tensor,Tensor,Tensor,Tensor> _embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset)
--
_embedding_bag__tttb6btb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> Int64 -> CBool -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_embedding_bag__tttb6btb weight indices offsets scale_grad_by_freq mode sparse per_sample_weights include_last_offset =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,Tensor>(at::_embedding_bag(*$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* offsets), $(bool scale_grad_by_freq), $(int64_t mode), $(bool sparse), *$fptr-ptr:(Tensor* per_sample_weights), $(bool include_last_offset)));
   }|] >>= unTupleTensorTensorTensorTensor


-- Tensor empty(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
empty__aom :: Vector Int64 -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
empty__aom size options memory_format =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::empty(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _empty_affine_quantized(IntArrayRef size, const TensorOptions & options, double scale, int64_t zero_point, c10::optional<MemoryFormat> memory_format)
--
_empty_affine_quantized__aod6m :: Vector Int64 -> ForeignPtr CTensorOptions -> CDouble -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
_empty_affine_quantized__aod6m size options scale zero_point memory_format =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::_empty_affine_quantized(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options), $(double scale), $(int64_t zero_point), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _empty_per_channel_affine_quantized(IntArrayRef size, const Tensor & scales, const Tensor & zero_points, int64_t axis, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
_empty_per_channel_affine_quantized__att6om :: Vector Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
_empty_per_channel_affine_quantized__att6om size scales zero_points axis options memory_format =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::_empty_per_channel_affine_quantized(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(Tensor* scales), *$fptr-ptr:(Tensor* zero_points), $(int64_t axis), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor empty_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
empty_like__tom :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
empty_like__tom self options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::empty_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor empty_strided(IntArrayRef size, IntArrayRef stride, const TensorOptions & options)
--
empty_strided__aao :: Vector Int64 -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
empty_strided__aao size stride options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in 
  [C.block|Tensor* {
    return new Tensor(at::empty_strided(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor eye(int64_t n, const TensorOptions & options)
--
eye__6o :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
eye__6o n options =  
  [C.block|Tensor* {
    return new Tensor(at::eye($(int64_t n), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor eye(int64_t n, int64_t m, const TensorOptions & options)
--
eye__66o :: Int64 -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
eye__66o n m options =  
  [C.block|Tensor* {
    return new Tensor(at::eye($(int64_t n), $(int64_t m), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor full(IntArrayRef size, Scalar fill_value, const TensorOptions & options)
--
full__aso :: Vector Int64 -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
full__aso size fill_value options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::full(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(Scalar* fill_value), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor full_like(const Tensor & self, Scalar fill_value, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
full_like__tsom :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
full_like__tsom self fill_value options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::full_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* fill_value), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor from_file(std::string filename, c10::optional<bool> shared, c10::optional<int64_t> size, const TensorOptions & options)
--
from_file__sb6o :: Ptr CChar -> CBool -> Maybe Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
from_file__sb6o filename shared size options =  let (size__is_present, size__value) = splitMaybe size 0 in 
  [C.block|Tensor* {
    return new Tensor(at::from_file($(char* filename), $(bool shared), ($(bool size__is_present) ? make_optional($(int64_t size__value)) : c10::nullopt), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hann_window(int64_t window_length, const TensorOptions & options)
--
hann_window__6o :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hann_window__6o window_length options =  
  [C.block|Tensor* {
    return new Tensor(at::hann_window($(int64_t window_length), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hann_window(int64_t window_length, bool periodic, const TensorOptions & options)
--
hann_window__6bo :: Int64 -> CBool -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hann_window__6bo window_length periodic options =  
  [C.block|Tensor* {
    return new Tensor(at::hann_window($(int64_t window_length), $(bool periodic), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hamming_window(int64_t window_length, const TensorOptions & options)
--
hamming_window__6o :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hamming_window__6o window_length options =  
  [C.block|Tensor* {
    return new Tensor(at::hamming_window($(int64_t window_length), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hamming_window(int64_t window_length, bool periodic, const TensorOptions & options)
--
hamming_window__6bo :: Int64 -> CBool -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hamming_window__6bo window_length periodic options =  
  [C.block|Tensor* {
    return new Tensor(at::hamming_window($(int64_t window_length), $(bool periodic), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hamming_window(int64_t window_length, bool periodic, double alpha, const TensorOptions & options)
--
hamming_window__6bdo :: Int64 -> CBool -> CDouble -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hamming_window__6bdo window_length periodic alpha options =  
  [C.block|Tensor* {
    return new Tensor(at::hamming_window($(int64_t window_length), $(bool periodic), $(double alpha), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hamming_window(int64_t window_length, bool periodic, double alpha, double beta, const TensorOptions & options)
--
hamming_window__6bddo :: Int64 -> CBool -> CDouble -> CDouble -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hamming_window__6bddo window_length periodic alpha beta options =  
  [C.block|Tensor* {
    return new Tensor(at::hamming_window($(int64_t window_length), $(bool periodic), $(double alpha), $(double beta), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor group_norm(const Tensor & input, int64_t num_groups, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enabled)
--
group_norm__t6ttdb :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
group_norm__t6ttdb input num_groups weight bias eps cudnn_enabled =  
  [C.block|Tensor* {
    return new Tensor(at::group_norm(*$fptr-ptr:(Tensor* input), $(int64_t num_groups), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), $(double eps), $(bool cudnn_enabled)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes)
--
irfft__t6bba :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> Vector Int64 -> IO (ForeignPtr CTensor)
irfft__t6bba self signal_ndim normalized onesided signal_sizes =  V.unsafeWith signal_sizes $ \signal_sizes__array -> let signal_sizes__size = fromIntegral (V.length signal_sizes) in 
  [C.block|Tensor* {
    return new Tensor(at::irfft(*$fptr-ptr:(Tensor* self), $(int64_t signal_ndim), $(bool normalized), $(bool onesided), ArrayRef<int64_t>($(int64_t* signal_sizes__array), $(size_t signal_sizes__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor layer_norm(const Tensor & input, IntArrayRef normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable)
--
layer_norm__tattdb :: ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
layer_norm__tattdb input normalized_shape weight bias eps cudnn_enable =  V.unsafeWith normalized_shape $ \normalized_shape__array -> let normalized_shape__size = fromIntegral (V.length normalized_shape) in 
  [C.block|Tensor* {
    return new Tensor(at::layer_norm(*$fptr-ptr:(Tensor* input), ArrayRef<int64_t>($(int64_t* normalized_shape__array), $(size_t normalized_shape__size)), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), $(double eps), $(bool cudnn_enable)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor linear(const Tensor & input, const Tensor & weight, const Tensor & bias)
--
linear__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
linear__ttt input weight bias =  
  [C.block|Tensor* {
    return new Tensor(at::linear(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mkldnn_linear(const Tensor & input, const Tensor & weight, const Tensor & bias)
--
mkldnn_linear__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mkldnn_linear__ttt input weight bias =  
  [C.block|Tensor* {
    return new Tensor(at::mkldnn_linear(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor linspace(Scalar start, Scalar end, int64_t steps, const TensorOptions & options)
--
linspace__ss6o :: ForeignPtr CScalar -> ForeignPtr CScalar -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
linspace__ss6o start end steps options =  
  [C.block|Tensor* {
    return new Tensor(at::linspace(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), $(int64_t steps), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor logspace(Scalar start, Scalar end, int64_t steps, double base, const TensorOptions & options)
--
logspace__ss6do :: ForeignPtr CScalar -> ForeignPtr CScalar -> Int64 -> CDouble -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
logspace__ss6do start end steps base options =  
  [C.block|Tensor* {
    return new Tensor(at::logspace(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), $(int64_t steps), $(double base), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> max_pool1d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool1d_with_indices__taaaab :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool1d_with_indices__taaaab self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::max_pool1d_with_indices(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= unTupleTensorTensor


-- Tensor max_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool1d__taaaab :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
max_pool1d__taaaab self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::max_pool1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool2d__taaaab :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
max_pool2d__taaaab self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mkldnn_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
mkldnn_max_pool2d__taaaab :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
mkldnn_max_pool2d__taaaab self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::mkldnn_max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor quantized_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
quantized_max_pool2d__taaaab :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
quantized_max_pool2d__taaaab self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::quantized_max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool3d__taaaab :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
max_pool3d__taaaab self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::max_pool3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor ones(IntArrayRef size, const TensorOptions & options)
--
ones__ao :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
ones__ao size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::ones(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor ones_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
ones_like__tom :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
ones_like__tom self options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::ones_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor scalar_tensor(Scalar s, const TensorOptions & options)
--
scalar_tensor__so :: ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
scalar_tensor__so s options =  
  [C.block|Tensor* {
    return new Tensor(at::scalar_tensor(*$fptr-ptr:(Scalar* s), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rand(IntArrayRef size, const TensorOptions & options)
--
rand__ao :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
rand__ao size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::rand(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rand(IntArrayRef size, Generator * generator, const TensorOptions & options)
--
rand__ago :: Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
rand__ago size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::rand(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rand_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
rand_like__tom :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
rand_like__tom self options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::rand_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randint(int64_t high, IntArrayRef size, const TensorOptions & options)
--
randint__6ao :: Int64 -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint__6ao high size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::randint($(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randint(int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options)
--
randint__6ago :: Int64 -> Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint__6ago high size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::randint($(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randint(int64_t low, int64_t high, IntArrayRef size, const TensorOptions & options)
--
randint__66ao :: Int64 -> Int64 -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint__66ao low high size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::randint($(int64_t low), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randint(int64_t low, int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options)
--
randint__66ago :: Int64 -> Int64 -> Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint__66ago low high size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::randint($(int64_t low), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randint_like(const Tensor & self, int64_t high, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
randint_like__t6om :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
randint_like__t6om self high options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::randint_like(*$fptr-ptr:(Tensor* self), $(int64_t high), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randint_like(const Tensor & self, int64_t low, int64_t high, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
randint_like__t66om :: ForeignPtr CTensor -> Int64 -> Int64 -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
randint_like__t66om self low high options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::randint_like(*$fptr-ptr:(Tensor* self), $(int64_t low), $(int64_t high), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randn(IntArrayRef size, const TensorOptions & options)
--
randn__ao :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randn__ao size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::randn(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randn(IntArrayRef size, Generator * generator, const TensorOptions & options)
--
randn__ago :: Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randn__ago size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::randn(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randn_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
randn_like__tom :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
randn_like__tom self options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::randn_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randperm(int64_t n, const TensorOptions & options)
--
randperm__6o :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randperm__6o n options =  
  [C.block|Tensor* {
    return new Tensor(at::randperm($(int64_t n), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randperm(int64_t n, Generator * generator, const TensorOptions & options)
--
randperm__6go :: Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randperm__6go n generator options =  
  [C.block|Tensor* {
    return new Tensor(at::randperm($(int64_t n), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor range(Scalar start, Scalar end, Scalar step, const TensorOptions & options)
--
range__ssso :: ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
range__ssso start end step options =  
  [C.block|Tensor* {
    return new Tensor(at::range(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(Scalar* step), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor range(Scalar start, Scalar end, const TensorOptions & options)
--
range__sso :: ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
range__sso start end options =  
  [C.block|Tensor* {
    return new Tensor(at::range(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor stft(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided)
--
stft__t666tbb :: ForeignPtr CTensor -> Int64 -> Maybe Int64 -> Maybe Int64 -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor)
stft__t666tbb self n_fft hop_length win_length window normalized onesided =  let (hop_length__is_present, hop_length__value) = splitMaybe hop_length 0 in let (win_length__is_present, win_length__value) = splitMaybe win_length 0 in 
  [C.block|Tensor* {
    return new Tensor(at::stft(*$fptr-ptr:(Tensor* self), $(int64_t n_fft), ($(bool hop_length__is_present) ? make_optional($(int64_t hop_length__value)) : c10::nullopt), ($(bool win_length__is_present) ? make_optional($(int64_t win_length__value)) : c10::nullopt), *$fptr-ptr:(Tensor* window), $(bool normalized), $(bool onesided)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor roll(const Tensor & self, IntArrayRef shifts, IntArrayRef dims)
--
roll__taa :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
roll__taa self shifts dims =  V.unsafeWith shifts $ \shifts__array -> let shifts__size = fromIntegral (V.length shifts) in V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in 
  [C.block|Tensor* {
    return new Tensor(at::roll(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* shifts__array), $(size_t shifts__size)), ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rot90(const Tensor & self, int64_t k, IntArrayRef dims)
--
rot90__t6a :: ForeignPtr CTensor -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
rot90__t6a self k dims =  V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in 
  [C.block|Tensor* {
    return new Tensor(at::rot90(*$fptr-ptr:(Tensor* self), $(int64_t k), ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor zeros(IntArrayRef size, const TensorOptions & options)
--
zeros__ao :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
zeros__ao size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::zeros(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor zeros_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
zeros_like__tom :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
zeros_like__tom self options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::zeros_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values, const TensorOptions & options)
--
sparse_coo_tensor__tto :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
sparse_coo_tensor__tto indices values options =  
  [C.block|Tensor* {
    return new Tensor(at::sparse_coo_tensor(*$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* values), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options)
--
sparse_coo_tensor__ttao :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
sparse_coo_tensor__ttao indices values size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::sparse_coo_tensor(*$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* values), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _sparse_coo_tensor_unsafe(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options)
--
_sparse_coo_tensor_unsafe__ttao :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
_sparse_coo_tensor_unsafe__ttao indices values size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::_sparse_coo_tensor_unsafe(*$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* values), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor,Tensor> _thnn_fused_lstm_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & cx, const Tensor & input_bias, const Tensor & hidden_bias)
--
_thnn_fused_lstm_cell__ttttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_thnn_fused_lstm_cell__ttttt input_gates hidden_gates cx input_bias hidden_bias =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::_thnn_fused_lstm_cell(*$fptr-ptr:(Tensor* input_gates), *$fptr-ptr:(Tensor* hidden_gates), *$fptr-ptr:(Tensor* cx), *$fptr-ptr:(Tensor* input_bias), *$fptr-ptr:(Tensor* hidden_bias)));
   }|] >>= unTupleTensorTensorTensor


-- std::tuple<Tensor,Tensor> _thnn_fused_gru_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const Tensor & input_bias, const Tensor & hidden_bias)
--
_thnn_fused_gru_cell__ttttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_thnn_fused_gru_cell__ttttt input_gates hidden_gates hx input_bias hidden_bias =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_thnn_fused_gru_cell(*$fptr-ptr:(Tensor* input_gates), *$fptr-ptr:(Tensor* hidden_gates), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* input_bias), *$fptr-ptr:(Tensor* hidden_bias)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh)
--
lstm_cell__tltttt :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
lstm_cell__tltttt input hx w_ih w_hh b_ih b_hh =  V.unsafeWith hx $ \hx__array -> let hx__size = fromIntegral (V.length hx) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::lstm_cell(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** hx__array), $(size_t hx__size)), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh)));
   }|] >>= unTupleTensorTensor


-- Tensor gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh)
--
gru_cell__tttttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
gru_cell__tttttt input hx w_ih w_hh b_ih b_hh =  
  [C.block|Tensor* {
    return new Tensor(at::gru_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh)
--
rnn_tanh_cell__tttttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rnn_tanh_cell__tttttt input hx w_ih w_hh b_ih b_hh =  
  [C.block|Tensor* {
    return new Tensor(at::rnn_tanh_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh)
--
rnn_relu_cell__tttttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rnn_relu_cell__tttttt input hx w_ih w_hh b_ih b_hh =  
  [C.block|Tensor* {
    return new Tensor(at::rnn_relu_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor normal(double mean, double std, IntArrayRef size, Generator * generator, const TensorOptions & options)
--
normal__ddago :: CDouble -> CDouble -> Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
normal__ddago mean std size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::normal($(double mean), $(double std), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & multi_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction)
--
multi_margin_loss_out__tttsst6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
multi_margin_loss_out__tttsst6 out self target p margin weight reduction =  
  [C.block|void {
    at::multi_margin_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Scalar* p), *$fptr-ptr:(Scalar* margin), *$fptr-ptr:(Tensor* weight), $(int64_t reduction));
   }|] >> pure (out)


-- Tensor multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction)
--
multi_margin_loss__ttsst6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
multi_margin_loss__ttsst6 self target p margin weight reduction =  
  [C.block|Tensor* {
    return new Tensor(at::multi_margin_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Scalar* p), *$fptr-ptr:(Scalar* margin), *$fptr-ptr:(Tensor* weight), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & nll_loss_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index)
--
nll_loss_out__tttt66 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
nll_loss_out__tttt66 out self target weight reduction ignore_index =  
  [C.block|void {
    at::nll_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction), $(int64_t ignore_index));
   }|] >> pure (out)


-- Tensor nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index)
--
nll_loss__ttt66 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
nll_loss__ttt66 self target weight reduction ignore_index =  
  [C.block|Tensor* {
    return new Tensor(at::nll_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction), $(int64_t ignore_index)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & nll_loss2d_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index)
--
nll_loss2d_out__tttt66 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
nll_loss2d_out__tttt66 out self target weight reduction ignore_index =  
  [C.block|void {
    at::nll_loss2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction), $(int64_t ignore_index));
   }|] >> pure (out)


-- Tensor nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index)
--
nll_loss2d__ttt66 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
nll_loss2d__ttt66 self target weight reduction ignore_index =  
  [C.block|Tensor* {
    return new Tensor(at::nll_loss2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction), $(int64_t ignore_index)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override)
--
avg_pool2d_out__ttaaabb6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor)
avg_pool2d_out__ttaaabb6 out self kernel_size stride padding ceil_mode count_include_pad divisor_override =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in let (divisor_override__is_present, divisor_override__value) = splitMaybe divisor_override 0 in 
  [C.block|void {
    at::avg_pool2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad), ($(bool divisor_override__is_present) ? make_optional($(int64_t divisor_override__value)) : c10::nullopt));
   }|] >> pure (out)


-- Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override)
--
avg_pool2d__taaabb6 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor)
avg_pool2d__taaabb6 self kernel_size stride padding ceil_mode count_include_pad divisor_override =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in let (divisor_override__is_present, divisor_override__value) = splitMaybe divisor_override 0 in 
  [C.block|Tensor* {
    return new Tensor(at::avg_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad), ($(bool divisor_override__is_present) ? make_optional($(int64_t divisor_override__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & avg_pool3d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override)
--
avg_pool3d_out__ttaaabb6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor)
avg_pool3d_out__ttaaabb6 out self kernel_size stride padding ceil_mode count_include_pad divisor_override =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in let (divisor_override__is_present, divisor_override__value) = splitMaybe divisor_override 0 in 
  [C.block|void {
    at::avg_pool3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad), ($(bool divisor_override__is_present) ? make_optional($(int64_t divisor_override__value)) : c10::nullopt));
   }|] >> pure (out)


-- Tensor avg_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override)
--
avg_pool3d__taaabb6 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor)
avg_pool3d__taaabb6 self kernel_size stride padding ceil_mode count_include_pad divisor_override =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in let (divisor_override__is_present, divisor_override__value) = splitMaybe divisor_override 0 in 
  [C.block|Tensor* {
    return new Tensor(at::avg_pool3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad), ($(bool divisor_override__is_present) ? make_optional($(int64_t divisor_override__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool2d_with_indices_out__tttaaaab :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool2d_with_indices_out__tttaaaab out indices self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void {
    at::max_pool2d_with_indices_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode));
   }|] >> pure (out,indices)


-- std::tuple<Tensor,Tensor> max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool2d_with_indices__taaaab :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool2d_with_indices__taaaab self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::max_pool2d_with_indices(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> max_pool3d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool3d_with_indices_out__tttaaaab :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool3d_with_indices_out__tttaaaab out indices self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void {
    at::max_pool3d_with_indices_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode));
   }|] >> pure (out,indices)


-- std::tuple<Tensor,Tensor> max_pool3d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool3d_with_indices__taaaab :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool3d_with_indices__taaaab self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::max_pool3d_with_indices(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= unTupleTensorTensor


-- Tensor & slow_conv_transpose2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation)
--
slow_conv_transpose2d_out__tttataaaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv_transpose2d_out__tttataaaa out self weight kernel_size bias stride padding output_padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void {
    at::slow_conv_transpose2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)));
   }|] >> pure (out)


-- Tensor slow_conv_transpose2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation)
--
slow_conv_transpose2d__ttataaaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv_transpose2d__ttataaaa self weight kernel_size bias stride padding output_padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::slow_conv_transpose2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & slow_conv_transpose3d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation)
--
slow_conv_transpose3d_out__tttataaaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv_transpose3d_out__tttataaaa out self weight kernel_size bias stride padding output_padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void {
    at::slow_conv_transpose3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)));
   }|] >> pure (out)


-- Tensor slow_conv_transpose3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation)
--
slow_conv_transpose3d__ttataaaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv_transpose3d__ttataaaa self weight kernel_size bias stride padding output_padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::slow_conv_transpose3d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & thnn_conv2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding)
--
thnn_conv2d_out__tttataa :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv2d_out__tttataa out self weight kernel_size bias stride padding =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|void {
    at::thnn_conv2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


-- Tensor thnn_conv2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding)
--
thnn_conv2d__ttataa :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv2d__ttataa self weight kernel_size bias stride padding =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|Tensor* {
    return new Tensor(at::thnn_conv2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & thnn_conv_depthwise2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation)
--
thnn_conv_depthwise2d_out__tttataaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv_depthwise2d_out__tttataaa out self weight kernel_size bias stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void {
    at::thnn_conv_depthwise2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)));
   }|] >> pure (out)


-- Tensor thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation)
--
thnn_conv_depthwise2d__ttataaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv_depthwise2d__ttataaa self weight kernel_size bias stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::thnn_conv_depthwise2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & slow_conv3d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding)
--
slow_conv3d_out__tttataa :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv3d_out__tttataa out self weight kernel_size bias stride padding =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|void {
    at::slow_conv3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


-- Tensor slow_conv3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding)
--
slow_conv3d__ttataa :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv3d__ttataa self weight kernel_size bias stride padding =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|Tensor* {
    return new Tensor(at::slow_conv3d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor slow_conv_dilated2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation)
--
slow_conv_dilated2d__ttataaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv_dilated2d__ttataaa self weight kernel_size bias stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::slow_conv_dilated2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor slow_conv_dilated3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation)
--
slow_conv_dilated3d__ttataaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv_dilated3d__ttataaa self weight kernel_size bias stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::slow_conv_dilated3d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _cast_Byte(const Tensor & self, bool non_blocking)
--
_cast_byte__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_byte__tb self non_blocking =  
  [C.block|Tensor* {
    return new Tensor(at::_cast_Byte(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _cast_Char(const Tensor & self, bool non_blocking)
--
_cast_char__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_char__tb self non_blocking =  
  [C.block|Tensor* {
    return new Tensor(at::_cast_Char(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _cast_Double(const Tensor & self, bool non_blocking)
--
_cast_double__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_double__tb self non_blocking =  
  [C.block|Tensor* {
    return new Tensor(at::_cast_Double(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _cast_Float(const Tensor & self, bool non_blocking)
--
_cast_float__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_float__tb self non_blocking =  
  [C.block|Tensor* {
    return new Tensor(at::_cast_Float(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _cast_Int(const Tensor & self, bool non_blocking)
--
_cast_int__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_int__tb self non_blocking =  
  [C.block|Tensor* {
    return new Tensor(at::_cast_Int(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _cast_Long(const Tensor & self, bool non_blocking)
--
_cast_long__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_long__tb self non_blocking =  
  [C.block|Tensor* {
    return new Tensor(at::_cast_Long(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _cast_Short(const Tensor & self, bool non_blocking)
--
_cast_short__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_short__tb self non_blocking =  
  [C.block|Tensor* {
    return new Tensor(at::_cast_Short(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _cast_Half(const Tensor & self, bool non_blocking)
--
_cast_half__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_half__tb self non_blocking =  
  [C.block|Tensor* {
    return new Tensor(at::_cast_Half(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


-- std::vector<Tensor> align_tensors(TensorList tensors)
--
align_tensors__l :: Vector (Ptr CTensor) -> IO (Vector (Ptr CTensor))
align_tensors__l tensors =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in 
  [C.block|void* {
    return (void*)new std::vector<Tensor>(at::align_tensors(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size))));
   }|] >>= unVectorTensor


-- bool _use_cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank)
--
_use_cudnn_ctc_loss__ttaa6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Int64 -> IO (CBool)
_use_cudnn_ctc_loss__ttaa6 log_probs targets input_lengths target_lengths blank =  V.unsafeWith input_lengths $ \input_lengths__array -> let input_lengths__size = fromIntegral (V.length input_lengths) in V.unsafeWith target_lengths $ \target_lengths__array -> let target_lengths__size = fromIntegral (V.length target_lengths) in 
  [C.block|bool {
    return at::_use_cudnn_ctc_loss(*$fptr-ptr:(Tensor* log_probs), *$fptr-ptr:(Tensor* targets), ArrayRef<int64_t>($(int64_t* input_lengths__array), $(size_t input_lengths__size)), ArrayRef<int64_t>($(int64_t* target_lengths__array), $(size_t target_lengths__size)), $(int64_t blank));
   }|]


-- std::tuple<Tensor,Tensor> _cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity)
--
_cudnn_ctc_loss__ttaa6bb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_cudnn_ctc_loss__ttaa6bb log_probs targets input_lengths target_lengths blank deterministic zero_infinity =  V.unsafeWith input_lengths $ \input_lengths__array -> let input_lengths__size = fromIntegral (V.length input_lengths) in V.unsafeWith target_lengths $ \target_lengths__array -> let target_lengths__size = fromIntegral (V.length target_lengths) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_cudnn_ctc_loss(*$fptr-ptr:(Tensor* log_probs), *$fptr-ptr:(Tensor* targets), ArrayRef<int64_t>($(int64_t* input_lengths__array), $(size_t input_lengths__size)), ArrayRef<int64_t>($(int64_t* target_lengths__array), $(size_t target_lengths__size)), $(int64_t blank), $(bool deterministic), $(bool zero_infinity)));
   }|] >>= unTupleTensorTensor


-- bool _use_cudnn_rnn_flatten_weight()
--
_use_cudnn_rnn_flatten_weight__ :: IO (CBool)
_use_cudnn_rnn_flatten_weight__  =  
  [C.block|bool {
    return at::_use_cudnn_rnn_flatten_weight();
   }|]


-- Tensor _cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional)
--
_cudnn_rnn_flatten_weight__l66666bb :: Vector (Ptr CTensor) -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
_cudnn_rnn_flatten_weight__l66666bb weight_arr weight_stride0 input_size mode hidden_size num_layers batch_first bidirectional =  V.unsafeWith weight_arr $ \weight_arr__array -> let weight_arr__size = fromIntegral (V.length weight_arr) in 
  [C.block|Tensor* {
    return new Tensor(at::_cudnn_rnn_flatten_weight(pack_tensor_list($(Tensor** weight_arr__array), $(size_t weight_arr__size)), $(int64_t weight_stride0), $(int64_t input_size), $(int64_t mode), $(int64_t hidden_size), $(int64_t num_layers), $(bool batch_first), $(bool bidirectional)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state)
--
_cudnn_rnn__tl6ttt666bdbbat :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> CBool -> CDouble -> CBool -> CBool -> Vector Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_cudnn_rnn__tl6ttt666bdbbat input weight weight_stride0 weight_buf hx cx mode hidden_size num_layers batch_first dropout train bidirectional batch_sizes dropout_state =  V.unsafeWith weight $ \weight__array -> let weight__size = fromIntegral (V.length weight) in V.unsafeWith batch_sizes $ \batch_sizes__array -> let batch_sizes__size = fromIntegral (V.length batch_sizes) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>(at::_cudnn_rnn(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** weight__array), $(size_t weight__size)), $(int64_t weight_stride0), *$fptr-ptr:(Tensor* weight_buf), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* cx), $(int64_t mode), $(int64_t hidden_size), $(int64_t num_layers), $(bool batch_first), $(double dropout), $(bool train), $(bool bidirectional), ArrayRef<int64_t>($(int64_t* batch_sizes__array), $(size_t batch_sizes__size)), *$fptr-ptr:(Tensor* dropout_state)));
   }|] >>= unTupleTensorTensorTensorTensorTensor


-- Tensor _cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const TensorOptions & options)
--
_cudnn_init_dropout_state__db6o :: CDouble -> CBool -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
_cudnn_init_dropout_state__db6o dropout train dropout_seed options =  
  [C.block|Tensor* {
    return new Tensor(at::_cudnn_init_dropout_state($(double dropout), $(bool train), $(int64_t dropout_seed), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- int64_t _debug_has_internal_overlap(const Tensor & self)
--
_debug_has_internal_overlap__t :: ForeignPtr CTensor -> IO (Int64)
_debug_has_internal_overlap__t self =  
  [C.block|int64_t {
    return at::_debug_has_internal_overlap(*$fptr-ptr:(Tensor* self));
   }|]


-- std::tuple<Tensor,Tensor> _fused_dropout(const Tensor & self, double p, Generator * generator)
--
_fused_dropout__tdg :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_fused_dropout__tdg self p generator =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_fused_dropout(*$fptr-ptr:(Tensor* self), $(double p), $(Generator* generator)));
   }|] >>= unTupleTensorTensor


-- Tensor _masked_scale(const Tensor & self, const Tensor & mask, double scale)
--
_masked_scale__ttd :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor)
_masked_scale__ttd self mask scale =  
  [C.block|Tensor* {
    return new Tensor(at::_masked_scale(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask), $(double scale)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> _sobol_engine_draw(const Tensor & quasi, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated, c10::optional<ScalarType> dtype)
--
_sobol_engine_draw__t6t66s :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> Int64 -> Int64 -> Int8 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_sobol_engine_draw__t6t66s quasi n sobolstate dimension num_generated dtype =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_sobol_engine_draw(*$fptr-ptr:(Tensor* quasi), $(int64_t n), *$fptr-ptr:(Tensor* sobolstate), $(int64_t dimension), $(int64_t num_generated), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= unTupleTensorTensor


-- Tensor & _sobol_engine_ff_(Tensor & self, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated)
--
_sobol_engine_ff___t6t66 :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
_sobol_engine_ff___t6t66 self n sobolstate dimension num_generated =  
  [C.block|void {
    at::_sobol_engine_ff_(*$fptr-ptr:(Tensor* self), $(int64_t n), *$fptr-ptr:(Tensor* sobolstate), $(int64_t dimension), $(int64_t num_generated));
   }|] >> pure self


-- Tensor & _sobol_engine_scramble_(Tensor & self, const Tensor & ltm, int64_t dimension)
--
_sobol_engine_scramble___tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_sobol_engine_scramble___tt6 self ltm dimension =  
  [C.block|void {
    at::_sobol_engine_scramble_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* ltm), $(int64_t dimension));
   }|] >> pure self


-- Tensor & _sobol_engine_initialize_state_(Tensor & self, int64_t dimension)
--
_sobol_engine_initialize_state___t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_sobol_engine_initialize_state___t6 self dimension =  
  [C.block|void {
    at::_sobol_engine_initialize_state_(*$fptr-ptr:(Tensor* self), $(int64_t dimension));
   }|] >> pure self


-- Tensor _reshape_from_tensor(const Tensor & self, const Tensor & shape)
--
_reshape_from_tensor__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_reshape_from_tensor__tt self shape =  
  [C.block|Tensor* {
    return new Tensor(at::_reshape_from_tensor(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* shape)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _shape_as_tensor(const Tensor & self)
--
_shape_as_tensor__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_shape_as_tensor__t self =  
  [C.block|Tensor* {
    return new Tensor(at::_shape_as_tensor(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor dropout(const Tensor & input, double p, bool train)
--
dropout__tdb :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
dropout__tdb input p train =  
  [C.block|Tensor* {
    return new Tensor(at::dropout(*$fptr-ptr:(Tensor* input), $(double p), $(bool train)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & dropout_(Tensor & self, double p, bool train)
--
dropout___tdb :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
dropout___tdb self p train =  
  [C.block|void {
    at::dropout_(*$fptr-ptr:(Tensor* self), $(double p), $(bool train));
   }|] >> pure self


-- Tensor feature_dropout(const Tensor & input, double p, bool train)
--
feature_dropout__tdb :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
feature_dropout__tdb input p train =  
  [C.block|Tensor* {
    return new Tensor(at::feature_dropout(*$fptr-ptr:(Tensor* input), $(double p), $(bool train)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & feature_dropout_(Tensor & self, double p, bool train)
--
feature_dropout___tdb :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
feature_dropout___tdb self p train =  
  [C.block|void {
    at::feature_dropout_(*$fptr-ptr:(Tensor* self), $(double p), $(bool train));
   }|] >> pure self


-- Tensor alpha_dropout(const Tensor & input, double p, bool train)
--
alpha_dropout__tdb :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
alpha_dropout__tdb input p train =  
  [C.block|Tensor* {
    return new Tensor(at::alpha_dropout(*$fptr-ptr:(Tensor* input), $(double p), $(bool train)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & alpha_dropout_(Tensor & self, double p, bool train)
--
alpha_dropout___tdb :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
alpha_dropout___tdb self p train =  
  [C.block|void {
    at::alpha_dropout_(*$fptr-ptr:(Tensor* self), $(double p), $(bool train));
   }|] >> pure self


-- Tensor feature_alpha_dropout(const Tensor & input, double p, bool train)
--
feature_alpha_dropout__tdb :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
feature_alpha_dropout__tdb input p train =  
  [C.block|Tensor* {
    return new Tensor(at::feature_alpha_dropout(*$fptr-ptr:(Tensor* input), $(double p), $(bool train)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & feature_alpha_dropout_(Tensor & self, double p, bool train)
--
feature_alpha_dropout___tdb :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
feature_alpha_dropout___tdb self p train =  
  [C.block|void {
    at::feature_alpha_dropout_(*$fptr-ptr:(Tensor* self), $(double p), $(bool train));
   }|] >> pure self


-- Tensor abs(const Tensor & self)
--
abs__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
abs__t self =  
  [C.block|Tensor* {
    return new Tensor(at::abs(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & abs_(Tensor & self)
--
abs___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
abs___t self =  
  [C.block|void {
    at::abs_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & abs_out(Tensor & out, const Tensor & self)
--
abs_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
abs_out__tt out self =  
  [C.block|void {
    at::abs_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor angle(const Tensor & self)
--
angle__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
angle__t self =  
  [C.block|Tensor* {
    return new Tensor(at::angle(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & angle_out(Tensor & out, const Tensor & self)
--
angle_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
angle_out__tt out self =  
  [C.block|void {
    at::angle_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor real(const Tensor & self)
--
real__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
real__t self =  
  [C.block|Tensor* {
    return new Tensor(at::real(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor imag(const Tensor & self)
--
imag__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
imag__t self =  
  [C.block|Tensor* {
    return new Tensor(at::imag(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conj(const Tensor & self)
--
conj__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
conj__t self =  
  [C.block|Tensor* {
    return new Tensor(at::conj(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & conj_out(Tensor & out, const Tensor & self)
--
conj_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
conj_out__tt out self =  
  [C.block|void {
    at::conj_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor acos(const Tensor & self)
--
acos__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
acos__t self =  
  [C.block|Tensor* {
    return new Tensor(at::acos(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & acos_(Tensor & self)
--
acos___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
acos___t self =  
  [C.block|void {
    at::acos_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & acos_out(Tensor & out, const Tensor & self)
--
acos_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
acos_out__tt out self =  
  [C.block|void {
    at::acos_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor avg_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad)
--
avg_pool1d__taaabb__1 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
avg_pool1d__taaabb__1 self kernel_size stride padding ceil_mode count_include_pad =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|Tensor* {
    return new Tensor(at::avg_pool1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor adaptive_avg_pool1d(const Tensor & self, IntArrayRef output_size)
--
adaptive_avg_pool1d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
adaptive_avg_pool1d__ta self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|Tensor* {
    return new Tensor(at::adaptive_avg_pool1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> adaptive_max_pool1d(const Tensor & self, IntArrayRef output_size)
--
adaptive_max_pool1d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
adaptive_max_pool1d__ta self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::adaptive_max_pool1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= unTupleTensorTensor


-- Tensor add(const Tensor & self, const Tensor & other, Scalar alpha)
--
add__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
add__tts self other alpha =  
  [C.block|Tensor* {
    return new Tensor(at::add(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha)
--
add_out__ttts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
add_out__ttts out self other alpha =  
  [C.block|void {
    at::add_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


-- Tensor add(const Tensor & self, Scalar other, Scalar alpha)
--
add__tss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
add__tss self other alpha =  
  [C.block|Tensor* {
    return new Tensor(at::add(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha)
--
addmv__tttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmv__tttss self mat vec beta alpha =  
  [C.block|Tensor* {
    return new Tensor(at::addmv(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat), *$fptr-ptr:(Tensor* vec), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha)
--
addmv___tttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmv___tttss self mat vec beta alpha =  
  [C.block|void {
    at::addmv_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat), *$fptr-ptr:(Tensor* vec), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor & addmv_out(Tensor & out, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha)
--
addmv_out__ttttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmv_out__ttttss out self mat vec beta alpha =  
  [C.block|void {
    at::addmv_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat), *$fptr-ptr:(Tensor* vec), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


-- Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha)
--
addr__tttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addr__tttss self vec1 vec2 beta alpha =  
  [C.block|Tensor* {
    return new Tensor(at::addr(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & addr_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha)
--
addr_out__ttttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addr_out__ttttss out self vec1 vec2 beta alpha =  
  [C.block|void {
    at::addr_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


-- Tensor affine_grid_generator(const Tensor & theta, IntArrayRef size, bool align_corners)
--
affine_grid_generator__tab :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
affine_grid_generator__tab theta size align_corners =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::affine_grid_generator(*$fptr-ptr:(Tensor* theta), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(bool align_corners)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor all(const Tensor & self, int64_t dim, bool keepdim)
--
all__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
all__t6b self dim keepdim =  
  [C.block|Tensor* {
    return new Tensor(at::all(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & all_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim)
--
all_out__tt6b :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
all_out__tt6b out self dim keepdim =  
  [C.block|void {
    at::all_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (out)


-- bool allclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan)
--
allclose__ttddb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> CBool -> IO (CBool)
allclose__ttddb self other rtol atol equal_nan =  
  [C.block|bool {
    return at::allclose(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), $(double rtol), $(double atol), $(bool equal_nan));
   }|]


-- Tensor any(const Tensor & self, int64_t dim, bool keepdim)
--
any__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
any__t6b self dim keepdim =  
  [C.block|Tensor* {
    return new Tensor(at::any(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & any_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim)
--
any_out__tt6b :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
any_out__tt6b out self dim keepdim =  
  [C.block|void {
    at::any_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (out)


-- Tensor arange(Scalar end, const TensorOptions & options)
--
arange__so__1 :: ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
arange__so__1 end options =  
  [C.block|Tensor* {
    return new Tensor(at::arange(*$fptr-ptr:(Scalar* end), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor arange(Scalar start, Scalar end, const TensorOptions & options)
--
arange__sso__1 :: ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
arange__sso__1 start end options =  
  [C.block|Tensor* {
    return new Tensor(at::arange(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor arange(Scalar start, Scalar end, Scalar step, const TensorOptions & options)
--
arange__ssso__1 :: ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
arange__ssso__1 start end step options =  
  [C.block|Tensor* {
    return new Tensor(at::arange(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(Scalar* step), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & arange_out(Tensor & out, Scalar end)
--
arange_out__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
arange_out__ts out end =  
  [C.block|void {
    at::arange_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Scalar* end));
   }|] >> pure (out)


-- Tensor & arange_out(Tensor & out, Scalar start, Scalar end, Scalar step)
--
arange_out__tsss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
arange_out__tsss out start end step =  
  [C.block|void {
    at::arange_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(Scalar* step));
   }|] >> pure (out)


-- Tensor _dim_arange(const Tensor & like, int64_t dim)
--
_dim_arange__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_dim_arange__t6 like dim =  
  [C.block|Tensor* {
    return new Tensor(at::_dim_arange(*$fptr-ptr:(Tensor* like), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor argmax(const Tensor & self, c10::optional<int64_t> dim, bool keepdim)
--
argmax__t6b :: ForeignPtr CTensor -> Maybe Int64 -> CBool -> IO (ForeignPtr CTensor)
argmax__t6b self dim keepdim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in 
  [C.block|Tensor* {
    return new Tensor(at::argmax(*$fptr-ptr:(Tensor* self), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor argmin(const Tensor & self, c10::optional<int64_t> dim, bool keepdim)
--
argmin__t6b :: ForeignPtr CTensor -> Maybe Int64 -> CBool -> IO (ForeignPtr CTensor)
argmin__t6b self dim keepdim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in 
  [C.block|Tensor* {
    return new Tensor(at::argmin(*$fptr-ptr:(Tensor* self), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset)
--
as_strided__taa6 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Maybe Int64 -> IO (ForeignPtr CTensor)
as_strided__taa6 self size stride storage_offset =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in let (storage_offset__is_present, storage_offset__value) = splitMaybe storage_offset 0 in 
  [C.block|Tensor* {
    return new Tensor(at::as_strided(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ($(bool storage_offset__is_present) ? make_optional($(int64_t storage_offset__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & as_strided_(Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset)
--
as_strided___taa6 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Maybe Int64 -> IO (ForeignPtr CTensor)
as_strided___taa6 self size stride storage_offset =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in let (storage_offset__is_present, storage_offset__value) = splitMaybe storage_offset 0 in 
  [C.block|void {
    at::as_strided_(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ($(bool storage_offset__is_present) ? make_optional($(int64_t storage_offset__value)) : c10::nullopt));
   }|] >> pure self


-- Tensor asin(const Tensor & self)
--
asin__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
asin__t self =  
  [C.block|Tensor* {
    return new Tensor(at::asin(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & asin_(Tensor & self)
--
asin___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
asin___t self =  
  [C.block|void {
    at::asin_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & asin_out(Tensor & out, const Tensor & self)
--
asin_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
asin_out__tt out self =  
  [C.block|void {
    at::asin_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor atan(const Tensor & self)
--
atan__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan__t self =  
  [C.block|Tensor* {
    return new Tensor(at::atan(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & atan_(Tensor & self)
--
atan___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan___t self =  
  [C.block|void {
    at::atan_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & atan_out(Tensor & out, const Tensor & self)
--
atan_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan_out__tt out self =  
  [C.block|void {
    at::atan_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha)
--
baddbmm__tttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
baddbmm__tttss self batch1 batch2 beta alpha =  
  [C.block|Tensor* {
    return new Tensor(at::baddbmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & _baddbmm_mkl_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha)
--
_baddbmm_mkl___tttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_baddbmm_mkl___tttss self batch1 batch2 beta alpha =  
  [C.block|void {
    at::_baddbmm_mkl_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor & baddbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha)
--
baddbmm_out__ttttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
baddbmm_out__ttttss out self batch1 batch2 beta alpha =  
  [C.block|void {
    at::baddbmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


-- Tensor bartlett_window(int64_t window_length, const TensorOptions & options)
--
bartlett_window__6o__1 :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
bartlett_window__6o__1 window_length options =  
  [C.block|Tensor* {
    return new Tensor(at::bartlett_window($(int64_t window_length), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bartlett_window(int64_t window_length, bool periodic, const TensorOptions & options)
--
bartlett_window__6bo__1 :: Int64 -> CBool -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
bartlett_window__6bo__1 window_length periodic options =  
  [C.block|Tensor* {
    return new Tensor(at::bartlett_window($(int64_t window_length), $(bool periodic), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled)
--
batch_norm__tttttbddb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> CBool -> IO (ForeignPtr CTensor)
batch_norm__tttttbddb input weight bias running_mean running_var training momentum eps cudnn_enabled =  
  [C.block|Tensor* {
    return new Tensor(at::batch_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool training), $(double momentum), $(double eps), $(bool cudnn_enabled)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor quantized_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & var, double eps, double output_scale, int64_t output_zero_point)
--
quantized_batch_norm__tttttdd6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
quantized_batch_norm__tttttdd6 input weight bias mean var eps output_scale output_zero_point =  
  [C.block|Tensor* {
    return new Tensor(at::quantized_batch_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* mean), *$fptr-ptr:(Tensor* var), $(double eps), $(double output_scale), $(int64_t output_zero_point)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t> _batch_norm_impl_index(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled)
--
_batch_norm_impl_index__tttttbddb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, Int64)
_batch_norm_impl_index__tttttbddb input weight bias running_mean running_var training momentum eps cudnn_enabled =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t>(at::_batch_norm_impl_index(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool training), $(double momentum), $(double eps), $(bool cudnn_enabled)));
   }|] >>= unTupleTensorTensorTensorTensorInt64


-- Tensor bernoulli(const Tensor & self, Generator * generator)
--
bernoulli__tg :: ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
bernoulli__tg self generator =  
  [C.block|Tensor* {
    return new Tensor(at::bernoulli(*$fptr-ptr:(Tensor* self), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & bernoulli_out(Tensor & out, const Tensor & self, Generator * generator)
--
bernoulli_out__ttg :: ForeignPtr CTensor -> ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
bernoulli_out__ttg out self generator =  
  [C.block|void {
    at::bernoulli_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(Generator* generator));
   }|] >> pure (out)


-- Tensor bernoulli(const Tensor & self, double p, Generator * generator)
--
bernoulli__tdg :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
bernoulli__tdg self p generator =  
  [C.block|Tensor* {
    return new Tensor(at::bernoulli(*$fptr-ptr:(Tensor* self), $(double p), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bilinear(const Tensor & input1, const Tensor & input2, const Tensor & weight, const Tensor & bias)
--
bilinear__tttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bilinear__tttt input1 input2 weight bias =  
  [C.block|Tensor* {
    return new Tensor(at::bilinear(*$fptr-ptr:(Tensor* input1), *$fptr-ptr:(Tensor* input2), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction)
--
binary_cross_entropy__ttt6__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
binary_cross_entropy__ttt6__1 self target weight reduction =  
  [C.block|Tensor* {
    return new Tensor(at::binary_cross_entropy(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & binary_cross_entropy_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction)
--
binary_cross_entropy_out__tttt6__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
binary_cross_entropy_out__tttt6__1 out self target weight reduction =  
  [C.block|void {
    at::binary_cross_entropy_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction));
   }|] >> pure (out)


-- Tensor binary_cross_entropy_with_logits(const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction)
--
binary_cross_entropy_with_logits__tttt6__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
binary_cross_entropy_with_logits__tttt6__1 self target weight pos_weight reduction =  
  [C.block|Tensor* {
    return new Tensor(at::binary_cross_entropy_with_logits(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* pos_weight), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bincount(const Tensor & self, const Tensor & weights, int64_t minlength)
--
bincount__tt6__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
bincount__tt6__1 self weights minlength =  
  [C.block|Tensor* {
    return new Tensor(at::bincount(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weights), $(int64_t minlength)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bitwise_not(const Tensor & self)
--
bitwise_not__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_not__t self =  
  [C.block|Tensor* {
    return new Tensor(at::bitwise_not(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & bitwise_not_out(Tensor & out, const Tensor & self)
--
bitwise_not_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_not_out__tt out self =  
  [C.block|void {
    at::bitwise_not_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor logical_not(const Tensor & self)
--
logical_not__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_not__t self =  
  [C.block|Tensor* {
    return new Tensor(at::logical_not(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & logical_not_out(Tensor & out, const Tensor & self)
--
logical_not_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_not_out__tt out self =  
  [C.block|void {
    at::logical_not_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor logical_xor(const Tensor & self, const Tensor & other)
--
logical_xor__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_xor__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::logical_xor(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & logical_xor_out(Tensor & out, const Tensor & self, const Tensor & other)
--
logical_xor_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_xor_out__ttt out self other =  
  [C.block|void {
    at::logical_xor_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor logical_and(const Tensor & self, const Tensor & other)
--
logical_and__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_and__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::logical_and(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & logical_and_out(Tensor & out, const Tensor & self, const Tensor & other)
--
logical_and_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_and_out__ttt out self other =  
  [C.block|void {
    at::logical_and_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor logical_or(const Tensor & self, const Tensor & other)
--
logical_or__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_or__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::logical_or(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & logical_or_out(Tensor & out, const Tensor & self, const Tensor & other)
--
logical_or_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_or_out__ttt out self other =  
  [C.block|void {
    at::logical_or_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor blackman_window(int64_t window_length, const TensorOptions & options)
--
blackman_window__6o__1 :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
blackman_window__6o__1 window_length options =  
  [C.block|Tensor* {
    return new Tensor(at::blackman_window($(int64_t window_length), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor blackman_window(int64_t window_length, bool periodic, const TensorOptions & options)
--
blackman_window__6bo__1 :: Int64 -> CBool -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
blackman_window__6bo__1 window_length periodic options =  
  [C.block|Tensor* {
    return new Tensor(at::blackman_window($(int64_t window_length), $(bool periodic), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bmm(const Tensor & self, const Tensor & mat2)
--
bmm__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bmm__tt self mat2 =  
  [C.block|Tensor* {
    return new Tensor(at::bmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & bmm_out(Tensor & out, const Tensor & self, const Tensor & mat2)
--
bmm_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bmm_out__ttt out self mat2 =  
  [C.block|void {
    at::bmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat2));
   }|] >> pure (out)


-- std::vector<Tensor> broadcast_tensors(TensorList tensors)
--
broadcast_tensors__l :: Vector (Ptr CTensor) -> IO (Vector (Ptr CTensor))
broadcast_tensors__l tensors =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in 
  [C.block|void* {
    return (void*)new std::vector<Tensor>(at::broadcast_tensors(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size))));
   }|] >>= unVectorTensor


-- Tensor cat(TensorList tensors, int64_t dim)
--
cat__l6 :: Vector (Ptr CTensor) -> Int64 -> IO (ForeignPtr CTensor)
cat__l6 tensors dim =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in 
  [C.block|Tensor* {
    return new Tensor(at::cat(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size)), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & cat_out(Tensor & out, TensorList tensors, int64_t dim)
--
cat_out__tl6 :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Int64 -> IO (ForeignPtr CTensor)
cat_out__tl6 out tensors dim =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in 
  [C.block|void {
    at::cat_out(*$fptr-ptr:(Tensor* out), pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size)), $(int64_t dim));
   }|] >> pure (out)


-- Tensor ceil(const Tensor & self)
--
ceil__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ceil__t self =  
  [C.block|Tensor* {
    return new Tensor(at::ceil(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & ceil_(Tensor & self)
--
ceil___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ceil___t self =  
  [C.block|void {
    at::ceil_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & ceil_out(Tensor & out, const Tensor & self)
--
ceil_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ceil_out__tt out self =  
  [C.block|void {
    at::ceil_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor chain_matmul(TensorList matrices)
--
chain_matmul__l :: Vector (Ptr CTensor) -> IO (ForeignPtr CTensor)
chain_matmul__l matrices =  V.unsafeWith matrices $ \matrices__array -> let matrices__size = fromIntegral (V.length matrices) in 
  [C.block|Tensor* {
    return new Tensor(at::chain_matmul(pack_tensor_list($(Tensor** matrices__array), $(size_t matrices__size))));
   }|] >>= newForeignPtr deleteTensor


-- std::vector<Tensor> chunk(const Tensor & self, int64_t chunks, int64_t dim)
--
chunk__t66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (Vector (Ptr CTensor))
chunk__t66 self chunks dim =  
  [C.block|void* {
    return (void*)new std::vector<Tensor>(at::chunk(*$fptr-ptr:(Tensor* self), $(int64_t chunks), $(int64_t dim)));
   }|] >>= unVectorTensor


-- Tensor clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max)
--
clamp__tss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp__tss self min max =  
  [C.block|Tensor* {
    return new Tensor(at::clamp(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max)
--
clamp___tss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp___tss self min max =  
  [C.block|void {
    at::clamp_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max));
   }|] >> pure self


-- Tensor & clamp_out(Tensor & out, const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max)
--
clamp_out__ttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_out__ttss out self min max =  
  [C.block|void {
    at::clamp_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max));
   }|] >> pure (out)


-- Tensor clamp_max(const Tensor & self, Scalar max)
--
clamp_max__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_max__ts self max =  
  [C.block|Tensor* {
    return new Tensor(at::clamp_max(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* max)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & clamp_max_(Tensor & self, Scalar max)
--
clamp_max___ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_max___ts self max =  
  [C.block|void {
    at::clamp_max_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* max));
   }|] >> pure self


-- Tensor & clamp_max_out(Tensor & out, const Tensor & self, Scalar max)
--
clamp_max_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_max_out__tts out self max =  
  [C.block|void {
    at::clamp_max_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* max));
   }|] >> pure (out)


-- Tensor clamp_min(const Tensor & self, Scalar min)
--
clamp_min__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_min__ts self min =  
  [C.block|Tensor* {
    return new Tensor(at::clamp_min(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & clamp_min_(Tensor & self, Scalar min)
--
clamp_min___ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_min___ts self min =  
  [C.block|void {
    at::clamp_min_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min));
   }|] >> pure self


-- Tensor & clamp_min_out(Tensor & out, const Tensor & self, Scalar min)
--
clamp_min_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_min_out__tts out self min =  
  [C.block|void {
    at::clamp_min_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min));
   }|] >> pure (out)


-- bool cudnn_is_acceptable(const Tensor & self)
--
cudnn_is_acceptable__t :: ForeignPtr CTensor -> IO (CBool)
cudnn_is_acceptable__t self =  
  [C.block|bool {
    return at::cudnn_is_acceptable(*$fptr-ptr:(Tensor* self));
   }|]


-- Tensor constant_pad_nd(const Tensor & self, IntArrayRef pad, Scalar value)
--
constant_pad_nd__tas :: ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
constant_pad_nd__tas self pad value =  V.unsafeWith pad $ \pad__array -> let pad__size = fromIntegral (V.length pad) in 
  [C.block|Tensor* {
    return new Tensor(at::constant_pad_nd(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* pad__array), $(size_t pad__size)), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups)
--
convolution__tttaaaba6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
convolution__tttaaaba6 input weight bias stride padding dilation transposed output_padding groups =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in 
  [C.block|Tensor* {
    return new Tensor(at::convolution(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool transposed), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor convolution_overrideable(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups)
--
convolution_overrideable__tttaaaba6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
convolution_overrideable__tttaaaba6 input weight bias stride padding dilation transposed output_padding groups =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in 
  [C.block|Tensor* {
    return new Tensor(at::convolution_overrideable(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool transposed), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled)
--
_convolution__tttaaaba6bbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> Vector Int64 -> Int64 -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor)
_convolution__tttaaaba6bbb input weight bias stride padding dilation transposed output_padding groups benchmark deterministic cudnn_enabled =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in 
  [C.block|Tensor* {
    return new Tensor(at::_convolution(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool transposed), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic), $(bool cudnn_enabled)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding)
--
_convolution_nogroup__tttaaaba :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> Vector Int64 -> IO (ForeignPtr CTensor)
_convolution_nogroup__tttaaaba input weight bias stride padding dilation transposed output_padding =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in 
  [C.block|Tensor* {
    return new Tensor(at::_convolution_nogroup(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool transposed), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups)
--
conv1d__tttaaa6__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
conv1d__tttaaa6__1 input weight bias stride padding dilation groups =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::conv1d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups)
--
conv2d__tttaaa6__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
conv2d__tttaaa6__1 input weight bias stride padding dilation groups =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::conv2d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups)
--
conv3d__tttaaa6__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
conv3d__tttaaa6__1 input weight bias stride padding dilation groups =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::conv3d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad)
--
conv_tbc__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
conv_tbc__ttt6 self weight bias pad =  
  [C.block|Tensor* {
    return new Tensor(at::conv_tbc(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), $(int64_t pad)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation)
--
conv_transpose1d__tttaaa6a__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose1d__tttaaa6a__1 input weight bias stride padding output_padding groups dilation =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::conv_transpose1d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv_transpose2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation)
--
conv_transpose2d__tttaaa6a__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose2d__tttaaa6a__1 input weight bias stride padding output_padding groups dilation =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::conv_transpose2d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation)
--
conv_transpose3d__tttaaa6a__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose3d__tttaaa6a__1 input weight bias stride padding output_padding groups dilation =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::conv_transpose3d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _copy_from(const Tensor & self, const Tensor & dst, bool non_blocking)
--
_copy_from__ttb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_copy_from__ttb self dst non_blocking =  
  [C.block|Tensor* {
    return new Tensor(at::_copy_from(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* dst), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor cos(const Tensor & self)
--
cos__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cos__t self =  
  [C.block|Tensor* {
    return new Tensor(at::cos(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & cos_(Tensor & self)
--
cos___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cos___t self =  
  [C.block|void {
    at::cos_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & cos_out(Tensor & out, const Tensor & self)
--
cos_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cos_out__tt out self =  
  [C.block|void {
    at::cos_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor cosh(const Tensor & self)
--
cosh__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cosh__t self =  
  [C.block|Tensor* {
    return new Tensor(at::cosh(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & cosh_(Tensor & self)
--
cosh___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cosh___t self =  
  [C.block|void {
    at::cosh_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & cosh_out(Tensor & out, const Tensor & self)
--
cosh_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cosh_out__tt out self =  
  [C.block|void {
    at::cosh_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction)
--
cosine_embedding_loss__tttd6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
cosine_embedding_loss__tttd6 input1 input2 target margin reduction =  
  [C.block|Tensor* {
    return new Tensor(at::cosine_embedding_loss(*$fptr-ptr:(Tensor* input1), *$fptr-ptr:(Tensor* input2), *$fptr-ptr:(Tensor* target), $(double margin), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W)
--
cudnn_affine_grid_generator__t6666 :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
cudnn_affine_grid_generator__t6666 theta n c h w =  
  [C.block|Tensor* {
    return new Tensor(at::cudnn_affine_grid_generator(*$fptr-ptr:(Tensor* theta), $(int64_t n), $(int64_t c), $(int64_t h), $(int64_t w)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor,Tensor,Tensor> cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon)
--
cudnn_batch_norm__tttttbdd :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
cudnn_batch_norm__tttttbdd input weight bias running_mean running_var training exponential_average_factor epsilon =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,Tensor>(at::cudnn_batch_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool training), $(double exponential_average_factor), $(double epsilon)));
   }|] >>= unTupleTensorTensorTensorTensor


-- Tensor cudnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)
--
cudnn_convolution__tttaaa6bb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
cudnn_convolution__tttaaa6bb self weight bias padding stride dilation groups benchmark deterministic =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::cudnn_convolution(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor cudnn_convolution(const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)
--
cudnn_convolution__ttaaa6bb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
cudnn_convolution__ttaaa6bb self weight padding stride dilation groups benchmark deterministic =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::cudnn_convolution(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)
--
cudnn_convolution_transpose__tttaaaa6bb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
cudnn_convolution_transpose__tttaaaa6bb self weight bias padding output_padding stride dilation groups benchmark deterministic =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::cudnn_convolution_transpose(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)
--
cudnn_convolution_transpose__ttaaaa6bb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
cudnn_convolution_transpose__ttaaaa6bb self weight padding output_padding stride dilation groups benchmark deterministic =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::cudnn_convolution_transpose(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor cudnn_grid_sampler(const Tensor & self, const Tensor & grid)
--
cudnn_grid_sampler__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cudnn_grid_sampler__tt self grid =  
  [C.block|Tensor* {
    return new Tensor(at::cudnn_grid_sampler(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* grid)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> cummax(const Tensor & self, int64_t dim)
--
cummax__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
cummax__t6 self dim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::cummax(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> cummax_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim)
--
cummax_out__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
cummax_out__ttt6 values indices self dim =  
  [C.block|void {
    at::cummax_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|] >> pure (values,indices)


-- void _cummax_helper(const Tensor & self, Tensor & values, Tensor & indices, int64_t dim)
--
_cummax_helper__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO ()
_cummax_helper__ttt6 self values indices dim =  
  [C.block|void {
    return at::_cummax_helper(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), $(int64_t dim));
   }|]


-- std::tuple<Tensor,Tensor> cummin(const Tensor & self, int64_t dim)
--
cummin__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
cummin__t6 self dim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::cummin(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> cummin_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim)
--
cummin_out__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
cummin_out__ttt6 values indices self dim =  
  [C.block|void {
    at::cummin_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|] >> pure (values,indices)


-- void _cummin_helper(const Tensor & self, Tensor & values, Tensor & indices, int64_t dim)
--
_cummin_helper__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO ()
_cummin_helper__ttt6 self values indices dim =  
  [C.block|void {
    return at::_cummin_helper(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), $(int64_t dim));
   }|]


-- Tensor cumprod(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype)
--
cumprod__t6s :: ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
cumprod__t6s self dim dtype =  
  [C.block|Tensor* {
    return new Tensor(at::cumprod(*$fptr-ptr:(Tensor* self), $(int64_t dim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & cumprod_out(Tensor & out, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype)
--
cumprod_out__tt6s :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
cumprod_out__tt6s out self dim dtype =  
  [C.block|void {
    at::cumprod_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), static_cast<ScalarType>($(int8_t dtype)));
   }|] >> pure (out)


-- Tensor cumsum(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype)
--
cumsum__t6s :: ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
cumsum__t6s self dim dtype =  
  [C.block|Tensor* {
    return new Tensor(at::cumsum(*$fptr-ptr:(Tensor* self), $(int64_t dim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & cumsum_out(Tensor & out, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype)
--
cumsum_out__tt6s :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
cumsum_out__tt6s out self dim dtype =  
  [C.block|void {
    at::cumsum_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), static_cast<ScalarType>($(int8_t dtype)));
   }|] >> pure (out)


-- Tensor ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity)
--
ctc_loss__ttaa66b :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor)
ctc_loss__ttaa66b log_probs targets input_lengths target_lengths blank reduction zero_infinity =  V.unsafeWith input_lengths $ \input_lengths__array -> let input_lengths__size = fromIntegral (V.length input_lengths) in V.unsafeWith target_lengths $ \target_lengths__array -> let target_lengths__size = fromIntegral (V.length target_lengths) in 
  [C.block|Tensor* {
    return new Tensor(at::ctc_loss(*$fptr-ptr:(Tensor* log_probs), *$fptr-ptr:(Tensor* targets), ArrayRef<int64_t>($(int64_t* input_lengths__array), $(size_t input_lengths__size)), ArrayRef<int64_t>($(int64_t* target_lengths__array), $(size_t target_lengths__size)), $(int64_t blank), $(int64_t reduction), $(bool zero_infinity)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor ctc_loss(const Tensor & log_probs, const Tensor & targets, const Tensor & input_lengths, const Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity)
--
ctc_loss__tttt66b :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor)
ctc_loss__tttt66b log_probs targets input_lengths target_lengths blank reduction zero_infinity =  
  [C.block|Tensor* {
    return new Tensor(at::ctc_loss(*$fptr-ptr:(Tensor* log_probs), *$fptr-ptr:(Tensor* targets), *$fptr-ptr:(Tensor* input_lengths), *$fptr-ptr:(Tensor* target_lengths), $(int64_t blank), $(int64_t reduction), $(bool zero_infinity)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> _ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool zero_infinity)
--
_ctc_loss__ttaa6b :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_ctc_loss__ttaa6b log_probs targets input_lengths target_lengths blank zero_infinity =  V.unsafeWith input_lengths $ \input_lengths__array -> let input_lengths__size = fromIntegral (V.length input_lengths) in V.unsafeWith target_lengths $ \target_lengths__array -> let target_lengths__size = fromIntegral (V.length target_lengths) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_ctc_loss(*$fptr-ptr:(Tensor* log_probs), *$fptr-ptr:(Tensor* targets), ArrayRef<int64_t>($(int64_t* input_lengths__array), $(size_t input_lengths__size)), ArrayRef<int64_t>($(int64_t* target_lengths__array), $(size_t target_lengths__size)), $(int64_t blank), $(bool zero_infinity)));
   }|] >>= unTupleTensorTensor


-- Tensor det(const Tensor & self)
--
det__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
det__t self =  
  [C.block|Tensor* {
    return new Tensor(at::det(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor diag_embed(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2)
--
diag_embed__t666 :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
diag_embed__t666 self offset dim1 dim2 =  
  [C.block|Tensor* {
    return new Tensor(at::diag_embed(*$fptr-ptr:(Tensor* self), $(int64_t offset), $(int64_t dim1), $(int64_t dim2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor diagflat(const Tensor & self, int64_t offset)
--
diagflat__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
diagflat__t6 self offset =  
  [C.block|Tensor* {
    return new Tensor(at::diagflat(*$fptr-ptr:(Tensor* self), $(int64_t offset)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor diagonal(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2)
--
diagonal__t666 :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
diagonal__t666 self offset dim1 dim2 =  
  [C.block|Tensor* {
    return new Tensor(at::diagonal(*$fptr-ptr:(Tensor* self), $(int64_t offset), $(int64_t dim1), $(int64_t dim2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor div(const Tensor & self, const Tensor & other)
--
div__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
div__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::div(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & div_out(Tensor & out, const Tensor & self, const Tensor & other)
--
div_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
div_out__ttt out self other =  
  [C.block|void {
    at::div_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor div(const Tensor & self, Scalar other)
--
div__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
div__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::div(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor dot(const Tensor & self, const Tensor & tensor)
--
dot__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
dot__tt self tensor =  
  [C.block|Tensor* {
    return new Tensor(at::dot(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & dot_out(Tensor & out, const Tensor & self, const Tensor & tensor)
--
dot_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
dot_out__ttt out self tensor =  
  [C.block|void {
    at::dot_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor));
   }|] >> pure (out)


-- Tensor einsum(std::string equation, TensorList tensors)
--
einsum__sl :: Ptr CChar -> Vector (Ptr CTensor) -> IO (ForeignPtr CTensor)
einsum__sl equation tensors =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in 
  [C.block|Tensor* {
    return new Tensor(at::einsum($(char* equation), pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse)
--
embedding__tt6bb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
embedding__tt6bb weight indices padding_idx scale_grad_by_freq sparse =  
  [C.block|Tensor* {
    return new Tensor(at::embedding(*$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* indices), $(int64_t padding_idx), $(bool scale_grad_by_freq), $(bool sparse)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type)
--
embedding_renorm___ttdd :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> IO (ForeignPtr CTensor)
embedding_renorm___ttdd self indices max_norm norm_type =  
  [C.block|void {
    at::embedding_renorm_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* indices), $(double max_norm), $(double norm_type));
   }|] >> pure self


-- std::tuple<Tensor,Tensor,Tensor,Tensor> embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset)
--
embedding_bag__tttb6btb__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> Int64 -> CBool -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
embedding_bag__tttb6btb__1 weight indices offsets scale_grad_by_freq mode sparse per_sample_weights include_last_offset =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,Tensor>(at::embedding_bag(*$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* offsets), $(bool scale_grad_by_freq), $(int64_t mode), $(bool sparse), *$fptr-ptr:(Tensor* per_sample_weights), $(bool include_last_offset)));
   }|] >>= unTupleTensorTensorTensorTensor


-- std::tuple<Tensor,Tensor,Tensor,Tensor> _embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset)
--
_embedding_bag__tttb6btb__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> Int64 -> CBool -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_embedding_bag__tttb6btb__1 weight indices offsets scale_grad_by_freq mode sparse per_sample_weights include_last_offset =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,Tensor>(at::_embedding_bag(*$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* offsets), $(bool scale_grad_by_freq), $(int64_t mode), $(bool sparse), *$fptr-ptr:(Tensor* per_sample_weights), $(bool include_last_offset)));
   }|] >>= unTupleTensorTensorTensorTensor


-- Tensor empty(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
empty__aom__1 :: Vector Int64 -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
empty__aom__1 size options memory_format =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::empty(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _empty_affine_quantized(IntArrayRef size, const TensorOptions & options, double scale, int64_t zero_point, c10::optional<MemoryFormat> memory_format)
--
_empty_affine_quantized__aod6m__1 :: Vector Int64 -> ForeignPtr CTensorOptions -> CDouble -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
_empty_affine_quantized__aod6m__1 size options scale zero_point memory_format =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::_empty_affine_quantized(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options), $(double scale), $(int64_t zero_point), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _empty_per_channel_affine_quantized(IntArrayRef size, const Tensor & scales, const Tensor & zero_points, int64_t axis, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
_empty_per_channel_affine_quantized__att6om__1 :: Vector Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
_empty_per_channel_affine_quantized__att6om__1 size scales zero_points axis options memory_format =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::_empty_per_channel_affine_quantized(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(Tensor* scales), *$fptr-ptr:(Tensor* zero_points), $(int64_t axis), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & empty_out(Tensor & out, IntArrayRef size, c10::optional<MemoryFormat> memory_format)
--
empty_out__tam :: ForeignPtr CTensor -> Vector Int64 -> Int8 -> IO (ForeignPtr CTensor)
empty_out__tam out size memory_format =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::empty_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), static_cast<MemoryFormat>($(int8_t memory_format)));
   }|] >> pure (out)


-- Tensor empty_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
empty_like__tom__1 :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
empty_like__tom__1 self options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::empty_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor empty_strided(IntArrayRef size, IntArrayRef stride, const TensorOptions & options)
--
empty_strided__aao__1 :: Vector Int64 -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
empty_strided__aao__1 size stride options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in 
  [C.block|Tensor* {
    return new Tensor(at::empty_strided(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor erf(const Tensor & self)
--
erf__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erf__t self =  
  [C.block|Tensor* {
    return new Tensor(at::erf(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & erf_(Tensor & self)
--
erf___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erf___t self =  
  [C.block|void {
    at::erf_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & erf_out(Tensor & out, const Tensor & self)
--
erf_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erf_out__tt out self =  
  [C.block|void {
    at::erf_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor erfc(const Tensor & self)
--
erfc__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfc__t self =  
  [C.block|Tensor* {
    return new Tensor(at::erfc(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & erfc_(Tensor & self)
--
erfc___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfc___t self =  
  [C.block|void {
    at::erfc_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & erfc_out(Tensor & out, const Tensor & self)
--
erfc_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfc_out__tt out self =  
  [C.block|void {
    at::erfc_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor exp(const Tensor & self)
--
exp__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
exp__t self =  
  [C.block|Tensor* {
    return new Tensor(at::exp(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & exp_(Tensor & self)
--
exp___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
exp___t self =  
  [C.block|void {
    at::exp_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & exp_out(Tensor & out, const Tensor & self)
--
exp_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
exp_out__tt out self =  
  [C.block|void {
    at::exp_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor expm1(const Tensor & self)
--
expm1__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
expm1__t self =  
  [C.block|Tensor* {
    return new Tensor(at::expm1(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & expm1_(Tensor & self)
--
expm1___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
expm1___t self =  
  [C.block|void {
    at::expm1_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & expm1_out(Tensor & out, const Tensor & self)
--
expm1_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
expm1_out__tt out self =  
  [C.block|void {
    at::expm1_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor eye(int64_t n, const TensorOptions & options)
--
eye__6o__1 :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
eye__6o__1 n options =  
  [C.block|Tensor* {
    return new Tensor(at::eye($(int64_t n), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor eye(int64_t n, int64_t m, const TensorOptions & options)
--
eye__66o__1 :: Int64 -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
eye__66o__1 n m options =  
  [C.block|Tensor* {
    return new Tensor(at::eye($(int64_t n), $(int64_t m), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & eye_out(Tensor & out, int64_t n)
--
eye_out__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
eye_out__t6 out n =  
  [C.block|void {
    at::eye_out(*$fptr-ptr:(Tensor* out), $(int64_t n));
   }|] >> pure (out)


-- Tensor & eye_out(Tensor & out, int64_t n, int64_t m)
--
eye_out__t66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
eye_out__t66 out n m =  
  [C.block|void {
    at::eye_out(*$fptr-ptr:(Tensor* out), $(int64_t n), $(int64_t m));
   }|] >> pure (out)


-- Tensor flatten(const Tensor & self, int64_t start_dim, int64_t end_dim)
--
flatten__t66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
flatten__t66 self start_dim end_dim =  
  [C.block|Tensor* {
    return new Tensor(at::flatten(*$fptr-ptr:(Tensor* self), $(int64_t start_dim), $(int64_t end_dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & fill_(Tensor & self, Scalar value)
--
fill___ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
fill___ts self value =  
  [C.block|void {
    at::fill_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


-- Tensor & fill_(Tensor & self, const Tensor & value)
--
fill___tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fill___tt self value =  
  [C.block|void {
    at::fill_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* value));
   }|] >> pure self


-- Tensor floor(const Tensor & self)
--
floor__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
floor__t self =  
  [C.block|Tensor* {
    return new Tensor(at::floor(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & floor_(Tensor & self)
--
floor___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
floor___t self =  
  [C.block|void {
    at::floor_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & floor_out(Tensor & out, const Tensor & self)
--
floor_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
floor_out__tt out self =  
  [C.block|void {
    at::floor_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor floor_divide(const Tensor & self, const Tensor & other)
--
floor_divide__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
floor_divide__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::floor_divide(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & floor_divide_out(Tensor & out, const Tensor & self, const Tensor & other)
--
floor_divide_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
floor_divide_out__ttt out self other =  
  [C.block|void {
    at::floor_divide_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor floor_divide(const Tensor & self, Scalar other)
--
floor_divide__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
floor_divide__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::floor_divide(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor frac(const Tensor & self)
--
frac__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
frac__t self =  
  [C.block|Tensor* {
    return new Tensor(at::frac(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & frac_(Tensor & self)
--
frac___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
frac___t self =  
  [C.block|void {
    at::frac_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & frac_out(Tensor & out, const Tensor & self)
--
frac_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
frac_out__tt out self =  
  [C.block|void {
    at::frac_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor full(IntArrayRef size, Scalar fill_value, const TensorOptions & options)
--
full__aso__1 :: Vector Int64 -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
full__aso__1 size fill_value options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::full(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(Scalar* fill_value), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & full_out(Tensor & out, IntArrayRef size, Scalar fill_value)
--
full_out__tas :: ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
full_out__tas out size fill_value =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::full_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(Scalar* fill_value));
   }|] >> pure (out)


-- Tensor full_like(const Tensor & self, Scalar fill_value, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
full_like__tsom__1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
full_like__tsom__1 self fill_value options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::full_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* fill_value), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor from_file(std::string filename, c10::optional<bool> shared, c10::optional<int64_t> size, const TensorOptions & options)
--
from_file__sb6o__1 :: Ptr CChar -> CBool -> Maybe Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
from_file__sb6o__1 filename shared size options =  let (size__is_present, size__value) = splitMaybe size 0 in 
  [C.block|Tensor* {
    return new Tensor(at::from_file($(char* filename), $(bool shared), ($(bool size__is_present) ? make_optional($(int64_t size__value)) : c10::nullopt), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor grid_sampler(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)
--
grid_sampler__tt66b :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor)
grid_sampler__tt66b input grid interpolation_mode padding_mode align_corners =  
  [C.block|Tensor* {
    return new Tensor(at::grid_sampler(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* grid), $(int64_t interpolation_mode), $(int64_t padding_mode), $(bool align_corners)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor grid_sampler_2d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)
--
grid_sampler_2d__tt66b :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor)
grid_sampler_2d__tt66b input grid interpolation_mode padding_mode align_corners =  
  [C.block|Tensor* {
    return new Tensor(at::grid_sampler_2d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* grid), $(int64_t interpolation_mode), $(int64_t padding_mode), $(bool align_corners)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor grid_sampler_3d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)
--
grid_sampler_3d__tt66b :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor)
grid_sampler_3d__tt66b input grid interpolation_mode padding_mode align_corners =  
  [C.block|Tensor* {
    return new Tensor(at::grid_sampler_3d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* grid), $(int64_t interpolation_mode), $(int64_t padding_mode), $(bool align_corners)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hann_window(int64_t window_length, const TensorOptions & options)
--
hann_window__6o__1 :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hann_window__6o__1 window_length options =  
  [C.block|Tensor* {
    return new Tensor(at::hann_window($(int64_t window_length), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hann_window(int64_t window_length, bool periodic, const TensorOptions & options)
--
hann_window__6bo__1 :: Int64 -> CBool -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hann_window__6bo__1 window_length periodic options =  
  [C.block|Tensor* {
    return new Tensor(at::hann_window($(int64_t window_length), $(bool periodic), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hamming_window(int64_t window_length, const TensorOptions & options)
--
hamming_window__6o__1 :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hamming_window__6o__1 window_length options =  
  [C.block|Tensor* {
    return new Tensor(at::hamming_window($(int64_t window_length), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hamming_window(int64_t window_length, bool periodic, const TensorOptions & options)
--
hamming_window__6bo__1 :: Int64 -> CBool -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hamming_window__6bo__1 window_length periodic options =  
  [C.block|Tensor* {
    return new Tensor(at::hamming_window($(int64_t window_length), $(bool periodic), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hamming_window(int64_t window_length, bool periodic, double alpha, const TensorOptions & options)
--
hamming_window__6bdo__1 :: Int64 -> CBool -> CDouble -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hamming_window__6bdo__1 window_length periodic alpha options =  
  [C.block|Tensor* {
    return new Tensor(at::hamming_window($(int64_t window_length), $(bool periodic), $(double alpha), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hamming_window(int64_t window_length, bool periodic, double alpha, double beta, const TensorOptions & options)
--
hamming_window__6bddo__1 :: Int64 -> CBool -> CDouble -> CDouble -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hamming_window__6bddo__1 window_length periodic alpha beta options =  
  [C.block|Tensor* {
    return new Tensor(at::hamming_window($(int64_t window_length), $(bool periodic), $(double alpha), $(double beta), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, int64_t reduction)
--
hinge_embedding_loss__ttd6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
hinge_embedding_loss__ttd6 self target margin reduction =  
  [C.block|Tensor* {
    return new Tensor(at::hinge_embedding_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(double margin), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor ger(const Tensor & self, const Tensor & vec2)
--
ger__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ger__tt self vec2 =  
  [C.block|Tensor* {
    return new Tensor(at::ger(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & ger_out(Tensor & out, const Tensor & self, const Tensor & vec2)
--
ger_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ger_out__ttt out self vec2 =  
  [C.block|void {
    at::ger_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec2));
   }|] >> pure (out)


-- Tensor group_norm(const Tensor & input, int64_t num_groups, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enabled)
--
group_norm__t6ttdb__1 :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
group_norm__t6ttdb__1 input num_groups weight bias eps cudnn_enabled =  
  [C.block|Tensor* {
    return new Tensor(at::group_norm(*$fptr-ptr:(Tensor* input), $(int64_t num_groups), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), $(double eps), $(bool cudnn_enabled)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor fft(const Tensor & self, int64_t signal_ndim, bool normalized)
--
fft__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
fft__t6b self signal_ndim normalized =  
  [C.block|Tensor* {
    return new Tensor(at::fft(*$fptr-ptr:(Tensor* self), $(int64_t signal_ndim), $(bool normalized)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor ifft(const Tensor & self, int64_t signal_ndim, bool normalized)
--
ifft__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
ifft__t6b self signal_ndim normalized =  
  [C.block|Tensor* {
    return new Tensor(at::ifft(*$fptr-ptr:(Tensor* self), $(int64_t signal_ndim), $(bool normalized)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided)
--
rfft__t6bb :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
rfft__t6bb self signal_ndim normalized onesided =  
  [C.block|Tensor* {
    return new Tensor(at::rfft(*$fptr-ptr:(Tensor* self), $(int64_t signal_ndim), $(bool normalized), $(bool onesided)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes)
--
irfft__t6bba__1 :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> Vector Int64 -> IO (ForeignPtr CTensor)
irfft__t6bba__1 self signal_ndim normalized onesided signal_sizes =  V.unsafeWith signal_sizes $ \signal_sizes__array -> let signal_sizes__size = fromIntegral (V.length signal_sizes) in 
  [C.block|Tensor* {
    return new Tensor(at::irfft(*$fptr-ptr:(Tensor* self), $(int64_t signal_ndim), $(bool normalized), $(bool onesided), ArrayRef<int64_t>($(int64_t* signal_sizes__array), $(size_t signal_sizes__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntArrayRef checked_signal_sizes, bool normalized, bool onesided, IntArrayRef output_sizes)
--
_fft_with_size__t6bbbabba :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> CBool -> Vector Int64 -> CBool -> CBool -> Vector Int64 -> IO (ForeignPtr CTensor)
_fft_with_size__t6bbbabba self signal_ndim complex_input complex_output inverse checked_signal_sizes normalized onesided output_sizes =  V.unsafeWith checked_signal_sizes $ \checked_signal_sizes__array -> let checked_signal_sizes__size = fromIntegral (V.length checked_signal_sizes) in V.unsafeWith output_sizes $ \output_sizes__array -> let output_sizes__size = fromIntegral (V.length output_sizes) in 
  [C.block|Tensor* {
    return new Tensor(at::_fft_with_size(*$fptr-ptr:(Tensor* self), $(int64_t signal_ndim), $(bool complex_input), $(bool complex_output), $(bool inverse), ArrayRef<int64_t>($(int64_t* checked_signal_sizes__array), $(size_t checked_signal_sizes__size)), $(bool normalized), $(bool onesided), ArrayRef<int64_t>($(int64_t* output_sizes__array), $(size_t output_sizes__size))));
   }|] >>= newForeignPtr deleteTensor


-- int64_t _cufft_get_plan_cache_size(int64_t device_index)
--
_cufft_get_plan_cache_size__6 :: Int64 -> IO (Int64)
_cufft_get_plan_cache_size__6 device_index =  
  [C.block|int64_t {
    return at::_cufft_get_plan_cache_size($(int64_t device_index));
   }|]


-- int64_t _cufft_get_plan_cache_max_size(int64_t device_index)
--
_cufft_get_plan_cache_max_size__6 :: Int64 -> IO (Int64)
_cufft_get_plan_cache_max_size__6 device_index =  
  [C.block|int64_t {
    return at::_cufft_get_plan_cache_max_size($(int64_t device_index));
   }|]


-- void _cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size)
--
_cufft_set_plan_cache_max_size__66 :: Int64 -> Int64 -> IO ()
_cufft_set_plan_cache_max_size__66 device_index max_size =  
  [C.block|void {
    return at::_cufft_set_plan_cache_max_size($(int64_t device_index), $(int64_t max_size));
   }|]


-- void _cufft_clear_plan_cache(int64_t device_index)
--
_cufft_clear_plan_cache__6 :: Int64 -> IO ()
_cufft_clear_plan_cache__6 device_index =  
  [C.block|void {
    return at::_cufft_clear_plan_cache($(int64_t device_index));
   }|]


-- Tensor index(const Tensor & self, TensorList indices)
--
index__tl :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> IO (ForeignPtr CTensor)
index__tl self indices =  V.unsafeWith indices $ \indices__array -> let indices__size = fromIntegral (V.length indices) in 
  [C.block|Tensor* {
    return new Tensor(at::index(*$fptr-ptr:(Tensor* self), pack_tensor_list($(Tensor** indices__array), $(size_t indices__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor index_copy(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source)
--
index_copy__t6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_copy__t6tt self dim index source =  
  [C.block|Tensor* {
    return new Tensor(at::index_copy(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & index_put_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate)
--
index_put___tltb :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
index_put___tltb self indices values accumulate =  V.unsafeWith indices $ \indices__array -> let indices__size = fromIntegral (V.length indices) in 
  [C.block|void {
    at::index_put_(*$fptr-ptr:(Tensor* self), pack_tensor_list($(Tensor** indices__array), $(size_t indices__size)), *$fptr-ptr:(Tensor* values), $(bool accumulate));
   }|] >> pure self


-- Tensor index_put(const Tensor & self, TensorList indices, const Tensor & values, bool accumulate)
--
index_put__tltb :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
index_put__tltb self indices values accumulate =  V.unsafeWith indices $ \indices__array -> let indices__size = fromIntegral (V.length indices) in 
  [C.block|Tensor* {
    return new Tensor(at::index_put(*$fptr-ptr:(Tensor* self), pack_tensor_list($(Tensor** indices__array), $(size_t indices__size)), *$fptr-ptr:(Tensor* values), $(bool accumulate)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & _index_put_impl_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate, bool unsafe)
--
_index_put_impl___tltbb :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor)
_index_put_impl___tltbb self indices values accumulate unsafe =  V.unsafeWith indices $ \indices__array -> let indices__size = fromIntegral (V.length indices) in 
  [C.block|void {
    at::_index_put_impl_(*$fptr-ptr:(Tensor* self), pack_tensor_list($(Tensor** indices__array), $(size_t indices__size)), *$fptr-ptr:(Tensor* values), $(bool accumulate), $(bool unsafe));
   }|] >> pure self


-- Tensor instance_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled)
--
instance_norm__tttttbddb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> CBool -> IO (ForeignPtr CTensor)
instance_norm__tttttbddb input weight bias running_mean running_var use_input_stats momentum eps cudnn_enabled =  
  [C.block|Tensor* {
    return new Tensor(at::instance_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool use_input_stats), $(double momentum), $(double eps), $(bool cudnn_enabled)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor inverse(const Tensor & self)
--
inverse__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
inverse__t self =  
  [C.block|Tensor* {
    return new Tensor(at::inverse(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & inverse_out(Tensor & out, const Tensor & self)
--
inverse_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
inverse_out__tt out self =  
  [C.block|void {
    at::inverse_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor _inverse_helper(const Tensor & self)
--
_inverse_helper__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_inverse_helper__t self =  
  [C.block|Tensor* {
    return new Tensor(at::_inverse_helper(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan)
--
isclose__ttddb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> CBool -> IO (ForeignPtr CTensor)
isclose__ttddb self other rtol atol equal_nan =  
  [C.block|Tensor* {
    return new Tensor(at::isclose(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), $(double rtol), $(double atol), $(bool equal_nan)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor isnan(const Tensor & self)
--
isnan__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
isnan__t self =  
  [C.block|Tensor* {
    return new Tensor(at::isnan(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- bool is_distributed(const Tensor & self)
--
is_distributed__t :: ForeignPtr CTensor -> IO (CBool)
is_distributed__t self =  
  [C.block|bool {
    return at::is_distributed(*$fptr-ptr:(Tensor* self));
   }|]


-- bool is_floating_point(const Tensor & self)
--
is_floating_point__t :: ForeignPtr CTensor -> IO (CBool)
is_floating_point__t self =  
  [C.block|bool {
    return at::is_floating_point(*$fptr-ptr:(Tensor* self));
   }|]


-- bool is_complex(const Tensor & self)
--
is_complex__t :: ForeignPtr CTensor -> IO (CBool)
is_complex__t self =  
  [C.block|bool {
    return at::is_complex(*$fptr-ptr:(Tensor* self));
   }|]


-- bool is_nonzero(const Tensor & self)
--
is_nonzero__t :: ForeignPtr CTensor -> IO (CBool)
is_nonzero__t self =  
  [C.block|bool {
    return at::is_nonzero(*$fptr-ptr:(Tensor* self));
   }|]


-- bool is_same_size(const Tensor & self, const Tensor & other)
--
is_same_size__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (CBool)
is_same_size__tt self other =  
  [C.block|bool {
    return at::is_same_size(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|]


-- bool is_signed(const Tensor & self)
--
is_signed__t :: ForeignPtr CTensor -> IO (CBool)
is_signed__t self =  
  [C.block|bool {
    return at::is_signed(*$fptr-ptr:(Tensor* self));
   }|]


-- Tensor kl_div(const Tensor & self, const Tensor & target, int64_t reduction)
--
kl_div__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
kl_div__tt6 self target reduction =  
  [C.block|Tensor* {
    return new Tensor(at::kl_div(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim)
--
kthvalue__t66b :: ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
kthvalue__t66b self k dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::kthvalue(*$fptr-ptr:(Tensor* self), $(int64_t k), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim)
--
kthvalue_out__ttt66b :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
kthvalue_out__ttt66b values indices self k dim keepdim =  
  [C.block|void {
    at::kthvalue_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t k), $(int64_t dim), $(bool keepdim));
   }|] >> pure (values,indices)


-- Tensor layer_norm(const Tensor & input, IntArrayRef normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable)
--
layer_norm__tattdb__1 :: ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
layer_norm__tattdb__1 input normalized_shape weight bias eps cudnn_enable =  V.unsafeWith normalized_shape $ \normalized_shape__array -> let normalized_shape__size = fromIntegral (V.length normalized_shape) in 
  [C.block|Tensor* {
    return new Tensor(at::layer_norm(*$fptr-ptr:(Tensor* input), ArrayRef<int64_t>($(int64_t* normalized_shape__array), $(size_t normalized_shape__size)), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), $(double eps), $(bool cudnn_enable)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor,Tensor> native_layer_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t M, int64_t N, double eps)
--
native_layer_norm__ttt66d :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
native_layer_norm__ttt66d input weight bias m n eps =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::native_layer_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), $(int64_t m), $(int64_t n), $(double eps)));
   }|] >>= unTupleTensorTensorTensor


-- Tensor linear(const Tensor & input, const Tensor & weight, const Tensor & bias)
--
linear__ttt__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
linear__ttt__1 input weight bias =  
  [C.block|Tensor* {
    return new Tensor(at::linear(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mkldnn_linear(const Tensor & input, const Tensor & weight, const Tensor & bias)
--
mkldnn_linear__ttt__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mkldnn_linear__ttt__1 input weight bias =  
  [C.block|Tensor* {
    return new Tensor(at::mkldnn_linear(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor fbgemm_linear_int8_weight_fp32_activation(const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, Scalar weight_scale, Scalar weight_zero_point, const Tensor & bias)
--
fbgemm_linear_int8_weight_fp32_activation__ttttsst :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fbgemm_linear_int8_weight_fp32_activation__ttttsst input weight packed col_offsets weight_scale weight_zero_point bias =  
  [C.block|Tensor* {
    return new Tensor(at::fbgemm_linear_int8_weight_fp32_activation(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* packed), *$fptr-ptr:(Tensor* col_offsets), *$fptr-ptr:(Scalar* weight_scale), *$fptr-ptr:(Scalar* weight_zero_point), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor fbgemm_linear_int8_weight(const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, Scalar weight_scale, Scalar weight_zero_point, const Tensor & bias)
--
fbgemm_linear_int8_weight__ttttsst :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fbgemm_linear_int8_weight__ttttsst input weight packed col_offsets weight_scale weight_zero_point bias =  
  [C.block|Tensor* {
    return new Tensor(at::fbgemm_linear_int8_weight(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* packed), *$fptr-ptr:(Tensor* col_offsets), *$fptr-ptr:(Scalar* weight_scale), *$fptr-ptr:(Scalar* weight_zero_point), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor,double,int64_t> fbgemm_linear_quantize_weight(const Tensor & input)
--
fbgemm_linear_quantize_weight__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor, CDouble, Int64)
fbgemm_linear_quantize_weight__t input =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,double,int64_t>(at::fbgemm_linear_quantize_weight(*$fptr-ptr:(Tensor* input)));
   }|] >>= unTupleTensorTensorDoubleInt64


-- Tensor fbgemm_pack_gemm_matrix_fp16(const Tensor & input)
--
fbgemm_pack_gemm_matrix_fp16__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fbgemm_pack_gemm_matrix_fp16__t input =  
  [C.block|Tensor* {
    return new Tensor(at::fbgemm_pack_gemm_matrix_fp16(*$fptr-ptr:(Tensor* input)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor fbgemm_linear_fp16_weight_fp32_activation(const Tensor & input, const Tensor & packed_weight, const Tensor & bias)
--
fbgemm_linear_fp16_weight_fp32_activation__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fbgemm_linear_fp16_weight_fp32_activation__ttt input packed_weight bias =  
  [C.block|Tensor* {
    return new Tensor(at::fbgemm_linear_fp16_weight_fp32_activation(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* packed_weight), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor fbgemm_linear_fp16_weight(const Tensor & input, const Tensor & packed_weight, const Tensor & bias)
--
fbgemm_linear_fp16_weight__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fbgemm_linear_fp16_weight__ttt input packed_weight bias =  
  [C.block|Tensor* {
    return new Tensor(at::fbgemm_linear_fp16_weight(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* packed_weight), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor fbgemm_pack_quantized_matrix(const Tensor & input)
--
fbgemm_pack_quantized_matrix__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fbgemm_pack_quantized_matrix__t input =  
  [C.block|Tensor* {
    return new Tensor(at::fbgemm_pack_quantized_matrix(*$fptr-ptr:(Tensor* input)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor fbgemm_pack_quantized_matrix(const Tensor & input, int64_t K, int64_t N)
--
fbgemm_pack_quantized_matrix__t66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
fbgemm_pack_quantized_matrix__t66 input k n =  
  [C.block|Tensor* {
    return new Tensor(at::fbgemm_pack_quantized_matrix(*$fptr-ptr:(Tensor* input), $(int64_t k), $(int64_t n)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor linspace(Scalar start, Scalar end, int64_t steps, const TensorOptions & options)
--
linspace__ss6o__1 :: ForeignPtr CScalar -> ForeignPtr CScalar -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
linspace__ss6o__1 start end steps options =  
  [C.block|Tensor* {
    return new Tensor(at::linspace(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), $(int64_t steps), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & linspace_out(Tensor & out, Scalar start, Scalar end, int64_t steps)
--
linspace_out__tss6 :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> Int64 -> IO (ForeignPtr CTensor)
linspace_out__tss6 out start end steps =  
  [C.block|void {
    at::linspace_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), $(int64_t steps));
   }|] >> pure (out)


-- Tensor log(const Tensor & self)
--
log__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log__t self =  
  [C.block|Tensor* {
    return new Tensor(at::log(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & log_(Tensor & self)
--
log___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log___t self =  
  [C.block|void {
    at::log_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & log_out(Tensor & out, const Tensor & self)
--
log_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log_out__tt out self =  
  [C.block|void {
    at::log_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor log10(const Tensor & self)
--
log10__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log10__t self =  
  [C.block|Tensor* {
    return new Tensor(at::log10(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & log10_(Tensor & self)
--
log10___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log10___t self =  
  [C.block|void {
    at::log10_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & log10_out(Tensor & out, const Tensor & self)
--
log10_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log10_out__tt out self =  
  [C.block|void {
    at::log10_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor log1p(const Tensor & self)
--
log1p__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log1p__t self =  
  [C.block|Tensor* {
    return new Tensor(at::log1p(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & log1p_(Tensor & self)
--
log1p___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log1p___t self =  
  [C.block|void {
    at::log1p_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & log1p_out(Tensor & out, const Tensor & self)
--
log1p_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log1p_out__tt out self =  
  [C.block|void {
    at::log1p_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor log2(const Tensor & self)
--
log2__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log2__t self =  
  [C.block|Tensor* {
    return new Tensor(at::log2(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & log2_(Tensor & self)
--
log2___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log2___t self =  
  [C.block|void {
    at::log2_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & log2_out(Tensor & out, const Tensor & self)
--
log2_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log2_out__tt out self =  
  [C.block|void {
    at::log2_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor logdet(const Tensor & self)
--
logdet__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logdet__t self =  
  [C.block|Tensor* {
    return new Tensor(at::logdet(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor logspace(Scalar start, Scalar end, int64_t steps, double base, const TensorOptions & options)
--
logspace__ss6do__1 :: ForeignPtr CScalar -> ForeignPtr CScalar -> Int64 -> CDouble -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
logspace__ss6do__1 start end steps base options =  
  [C.block|Tensor* {
    return new Tensor(at::logspace(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), $(int64_t steps), $(double base), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & logspace_out(Tensor & out, Scalar start, Scalar end, int64_t steps, double base)
--
logspace_out__tss6d :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> Int64 -> CDouble -> IO (ForeignPtr CTensor)
logspace_out__tss6d out start end steps base =  
  [C.block|void {
    at::logspace_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), $(int64_t steps), $(double base));
   }|] >> pure (out)


-- Tensor log_softmax(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype)
--
log_softmax__t6s :: ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
log_softmax__t6s self dim dtype =  
  [C.block|Tensor* {
    return new Tensor(at::log_softmax(*$fptr-ptr:(Tensor* self), $(int64_t dim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _log_softmax(const Tensor & self, int64_t dim, bool half_to_float)
--
_log_softmax__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
_log_softmax__t6b self dim half_to_float =  
  [C.block|Tensor* {
    return new Tensor(at::_log_softmax(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool half_to_float)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor logsumexp(const Tensor & self, IntArrayRef dim, bool keepdim)
--
logsumexp__tab :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
logsumexp__tab self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::logsumexp(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & logsumexp_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim)
--
logsumexp_out__ttab :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
logsumexp_out__ttab out self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|void {
    at::logsumexp_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim));
   }|] >> pure (out)


-- Tensor margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction)
--
margin_ranking_loss__tttd6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
margin_ranking_loss__tttd6 input1 input2 target margin reduction =  
  [C.block|Tensor* {
    return new Tensor(at::margin_ranking_loss(*$fptr-ptr:(Tensor* input1), *$fptr-ptr:(Tensor* input2), *$fptr-ptr:(Tensor* target), $(double margin), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor matmul(const Tensor & self, const Tensor & other)
--
matmul__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
matmul__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::matmul(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & matmul_out(Tensor & out, const Tensor & self, const Tensor & other)
--
matmul_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
matmul_out__ttt out self other =  
  [C.block|void {
    at::matmul_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor matrix_rank(const Tensor & self, double tol, bool symmetric)
--
matrix_rank__tdb :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
matrix_rank__tdb self tol symmetric =  
  [C.block|Tensor* {
    return new Tensor(at::matrix_rank(*$fptr-ptr:(Tensor* self), $(double tol), $(bool symmetric)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor matrix_rank(const Tensor & self, bool symmetric)
--
matrix_rank__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
matrix_rank__tb self symmetric =  
  [C.block|Tensor* {
    return new Tensor(at::matrix_rank(*$fptr-ptr:(Tensor* self), $(bool symmetric)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor matrix_power(const Tensor & self, int64_t n)
--
matrix_power__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
matrix_power__t6 self n =  
  [C.block|Tensor* {
    return new Tensor(at::matrix_power(*$fptr-ptr:(Tensor* self), $(int64_t n)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim)
--
max__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max__t6b self dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::max(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> max_out(Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim)
--
max_out__ttt6b :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_out__ttt6b max max_values self dim keepdim =  
  [C.block|void {
    at::max_out(*$fptr-ptr:(Tensor* max), *$fptr-ptr:(Tensor* max_values), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (max,max_values)


-- Tensor max_values(const Tensor & self, IntArrayRef dim, bool keepdim)
--
max_values__tab :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
max_values__tab self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::max_values(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> max_pool1d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool1d_with_indices__taaaab__1 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool1d_with_indices__taaaab__1 self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::max_pool1d_with_indices(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= unTupleTensorTensor


-- Tensor max_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool1d__taaaab__1 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
max_pool1d__taaaab__1 self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::max_pool1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool2d__taaaab__1 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
max_pool2d__taaaab__1 self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mkldnn_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
mkldnn_max_pool2d__taaaab__1 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
mkldnn_max_pool2d__taaaab__1 self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::mkldnn_max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor quantized_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
quantized_max_pool2d__taaaab__1 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
quantized_max_pool2d__taaaab__1 self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::quantized_max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool3d__taaaab__1 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
max_pool3d__taaaab__1 self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::max_pool3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mean(const Tensor & self, c10::optional<ScalarType> dtype)
--
mean__ts :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
mean__ts self dtype =  
  [C.block|Tensor* {
    return new Tensor(at::mean(*$fptr-ptr:(Tensor* self), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mean(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype)
--
mean__tabs :: ForeignPtr CTensor -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
mean__tabs self dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::mean(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & mean_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype)
--
mean_out__ttabs :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
mean_out__ttabs out self dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|void {
    at::mean_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype)));
   }|] >> pure (out)


-- std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim, bool keepdim)
--
median__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
median__t6b self dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::median(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim)
--
median_out__ttt6b :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
median_out__ttt6b values indices self dim keepdim =  
  [C.block|void {
    at::median_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (values,indices)


-- std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim)
--
min__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
min__t6b self dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::min(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim)
--
min_out__ttt6b :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
min_out__ttt6b min min_indices self dim keepdim =  
  [C.block|void {
    at::min_out(*$fptr-ptr:(Tensor* min), *$fptr-ptr:(Tensor* min_indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (min,min_indices)


-- Tensor min_values(const Tensor & self, IntArrayRef dim, bool keepdim)
--
min_values__tab :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
min_values__tab self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::min_values(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mkldnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
--
mkldnn_convolution__tttaaa6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
mkldnn_convolution__tttaaa6 self weight bias padding stride dilation groups =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::mkldnn_convolution(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor,Tensor> miopen_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon)
--
miopen_batch_norm__tttttbdd :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
miopen_batch_norm__tttttbdd input weight bias running_mean running_var training exponential_average_factor epsilon =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::miopen_batch_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool training), $(double exponential_average_factor), $(double epsilon)));
   }|] >>= unTupleTensorTensorTensor


-- Tensor miopen_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)
--
miopen_convolution__tttaaa6bb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
miopen_convolution__tttaaa6bb self weight bias padding stride dilation groups benchmark deterministic =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::miopen_convolution(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor miopen_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)
--
miopen_convolution_transpose__tttaaaa6bb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
miopen_convolution_transpose__tttaaaa6bb self weight bias padding output_padding stride dilation groups benchmark deterministic =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::miopen_convolution_transpose(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor miopen_depthwise_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)
--
miopen_depthwise_convolution__tttaaa6bb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
miopen_depthwise_convolution__tttaaa6bb self weight bias padding stride dilation groups benchmark deterministic =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::miopen_depthwise_convolution(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> miopen_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state)
--
miopen_rnn__tl6tt666bdbbat :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> CBool -> CDouble -> CBool -> CBool -> Vector Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
miopen_rnn__tl6tt666bdbbat input weight weight_stride0 hx cx mode hidden_size num_layers batch_first dropout train bidirectional batch_sizes dropout_state =  V.unsafeWith weight $ \weight__array -> let weight__size = fromIntegral (V.length weight) in V.unsafeWith batch_sizes $ \batch_sizes__array -> let batch_sizes__size = fromIntegral (V.length batch_sizes) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>(at::miopen_rnn(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** weight__array), $(size_t weight__size)), $(int64_t weight_stride0), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* cx), $(int64_t mode), $(int64_t hidden_size), $(int64_t num_layers), $(bool batch_first), $(double dropout), $(bool train), $(bool bidirectional), ArrayRef<int64_t>($(int64_t* batch_sizes__array), $(size_t batch_sizes__size)), *$fptr-ptr:(Tensor* dropout_state)));
   }|] >>= unTupleTensorTensorTensorTensorTensor


-- Tensor mm(const Tensor & self, const Tensor & mat2)
--
mm__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mm__tt self mat2 =  
  [C.block|Tensor* {
    return new Tensor(at::mm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & mm_out(Tensor & out, const Tensor & self, const Tensor & mat2)
--
mm_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mm_out__ttt out self mat2 =  
  [C.block|void {
    at::mm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat2));
   }|] >> pure (out)


-- Tensor _sparse_mm(const Tensor & sparse, const Tensor & dense)
--
_sparse_mm__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_sparse_mm__tt sparse dense =  
  [C.block|Tensor* {
    return new Tensor(at::_sparse_mm(*$fptr-ptr:(Tensor* sparse), *$fptr-ptr:(Tensor* dense)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim, bool keepdim)
--
mode__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
mode__t6b self dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::mode(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim)
--
mode_out__ttt6b :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
mode_out__ttt6b values indices self dim keepdim =  
  [C.block|void {
    at::mode_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (values,indices)


-- Tensor mul(const Tensor & self, const Tensor & other)
--
mul__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mul__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::mul(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & mul_out(Tensor & out, const Tensor & self, const Tensor & other)
--
mul_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mul_out__ttt out self other =  
  [C.block|void {
    at::mul_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor mul(const Tensor & self, Scalar other)
--
mul__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
mul__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::mul(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mv(const Tensor & self, const Tensor & vec)
--
mv__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mv__tt self vec =  
  [C.block|Tensor* {
    return new Tensor(at::mv(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & mv_out(Tensor & out, const Tensor & self, const Tensor & vec)
--
mv_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mv_out__ttt out self vec =  
  [C.block|void {
    at::mv_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec));
   }|] >> pure (out)


-- Tensor mvlgamma(const Tensor & self, int64_t p)
--
mvlgamma__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
mvlgamma__t6 self p =  
  [C.block|Tensor* {
    return new Tensor(at::mvlgamma(*$fptr-ptr:(Tensor* self), $(int64_t p)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length)
--
narrow__t666 :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
narrow__t666 self dim start length =  
  [C.block|Tensor* {
    return new Tensor(at::narrow(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(int64_t start), $(int64_t length)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor narrow(const Tensor & self, int64_t dim, const Tensor & start, int64_t length)
--
narrow__t6t6 :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
narrow__t6t6 self dim start length =  
  [C.block|Tensor* {
    return new Tensor(at::narrow(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* start), $(int64_t length)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps)
--
native_batch_norm__tttttbdd :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
native_batch_norm__tttttbdd input weight bias running_mean running_var training momentum eps =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::native_batch_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool training), $(double momentum), $(double eps)));
   }|] >>= unTupleTensorTensorTensor


-- std::tuple<Tensor &,Tensor &,Tensor &> native_batch_norm_out(Tensor & out, Tensor & save_mean, Tensor & save_invstd, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps)
--
native_batch_norm_out__ttttttttbdd :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
native_batch_norm_out__ttttttttbdd out save_mean save_invstd input weight bias running_mean running_var training momentum eps =  
  [C.block|void {
    at::native_batch_norm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* save_mean), *$fptr-ptr:(Tensor* save_invstd), *$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool training), $(double momentum), $(double eps));
   }|] >> pure (out,save_mean,save_invstd)


-- std::tuple<Tensor,Tensor> batch_norm_stats(const Tensor & input, double eps)
--
batch_norm_stats__td :: ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
batch_norm_stats__td input eps =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::batch_norm_stats(*$fptr-ptr:(Tensor* input), $(double eps)));
   }|] >>= unTupleTensorTensor


-- Tensor batch_norm_elemt(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & invstd, double eps)
--
batch_norm_elemt__tttttd :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor)
batch_norm_elemt__tttttd input weight bias mean invstd eps =  
  [C.block|Tensor* {
    return new Tensor(at::batch_norm_elemt(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* mean), *$fptr-ptr:(Tensor* invstd), $(double eps)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & batch_norm_elemt_out(Tensor & out, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & invstd, double eps)
--
batch_norm_elemt_out__ttttttd :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor)
batch_norm_elemt_out__ttttttd out input weight bias mean invstd eps =  
  [C.block|void {
    at::batch_norm_elemt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* mean), *$fptr-ptr:(Tensor* invstd), $(double eps));
   }|] >> pure (out)


-- std::tuple<Tensor,Tensor> batch_norm_gather_stats(const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, int64_t count)
--
batch_norm_gather_stats__tttttdd6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
batch_norm_gather_stats__tttttdd6 input mean invstd running_mean running_var momentum eps count =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::batch_norm_gather_stats(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* mean), *$fptr-ptr:(Tensor* invstd), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(double momentum), $(double eps), $(int64_t count)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> batch_norm_gather_stats_with_counts(const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, IntArrayRef counts)
--
batch_norm_gather_stats_with_counts__tttttdda :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> Vector Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
batch_norm_gather_stats_with_counts__tttttdda input mean invstd running_mean running_var momentum eps counts =  V.unsafeWith counts $ \counts__array -> let counts__size = fromIntegral (V.length counts) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::batch_norm_gather_stats_with_counts(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* mean), *$fptr-ptr:(Tensor* invstd), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(double momentum), $(double eps), ArrayRef<int64_t>($(int64_t* counts__array), $(size_t counts__size))));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> batch_norm_update_stats(const Tensor & input, const Tensor & running_mean, const Tensor & running_var, double momentum)
--
batch_norm_update_stats__tttd :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
batch_norm_update_stats__tttd input running_mean running_var momentum =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::batch_norm_update_stats(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(double momentum)));
   }|] >>= unTupleTensorTensor


-- bool _nnpack_available()
--
_nnpack_available__ :: IO (CBool)
_nnpack_available__  =  
  [C.block|bool {
    return at::_nnpack_available();
   }|]


-- Tensor _nnpack_spatial_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride)
--
_nnpack_spatial_convolution__tttaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
_nnpack_spatial_convolution__tttaa input weight bias padding stride =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in 
  [C.block|Tensor* {
    return new Tensor(at::_nnpack_spatial_convolution(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor ones(IntArrayRef size, const TensorOptions & options)
--
ones__ao__1 :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
ones__ao__1 size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::ones(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & ones_out(Tensor & out, IntArrayRef size)
--
ones_out__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
ones_out__ta out size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::ones_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure (out)


-- Tensor ones_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
ones_like__tom__1 :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
ones_like__tom__1 self options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::ones_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor pairwise_distance(const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim)
--
pairwise_distance__ttddb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> CBool -> IO (ForeignPtr CTensor)
pairwise_distance__ttddb x1 x2 p eps keepdim =  
  [C.block|Tensor* {
    return new Tensor(at::pairwise_distance(*$fptr-ptr:(Tensor* x1), *$fptr-ptr:(Tensor* x2), $(double p), $(double eps), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor cdist(const Tensor & x1, const Tensor & x2, double p, c10::optional<int64_t> compute_mode)
--
cdist__ttd6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> Maybe Int64 -> IO (ForeignPtr CTensor)
cdist__ttd6 x1 x2 p compute_mode =  let (compute_mode__is_present, compute_mode__value) = splitMaybe compute_mode 0 in 
  [C.block|Tensor* {
    return new Tensor(at::cdist(*$fptr-ptr:(Tensor* x1), *$fptr-ptr:(Tensor* x2), $(double p), ($(bool compute_mode__is_present) ? make_optional($(int64_t compute_mode__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor pdist(const Tensor & self, double p)
--
pdist__td :: ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor)
pdist__td self p =  
  [C.block|Tensor* {
    return new Tensor(at::pdist(*$fptr-ptr:(Tensor* self), $(double p)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor cosine_similarity(const Tensor & x1, const Tensor & x2, int64_t dim, double eps)
--
cosine_similarity__tt6d :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CDouble -> IO (ForeignPtr CTensor)
cosine_similarity__tt6d x1 x2 dim eps =  
  [C.block|Tensor* {
    return new Tensor(at::cosine_similarity(*$fptr-ptr:(Tensor* x1), *$fptr-ptr:(Tensor* x2), $(int64_t dim), $(double eps)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor pixel_shuffle(const Tensor & self, int64_t upscale_factor)
--
pixel_shuffle__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
pixel_shuffle__t6 self upscale_factor =  
  [C.block|Tensor* {
    return new Tensor(at::pixel_shuffle(*$fptr-ptr:(Tensor* self), $(int64_t upscale_factor)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor pinverse(const Tensor & self, double rcond)
--
pinverse__td :: ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor)
pinverse__td self rcond =  
  [C.block|Tensor* {
    return new Tensor(at::pinverse(*$fptr-ptr:(Tensor* self), $(double rcond)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor poisson_nll_loss(const Tensor & input, const Tensor & target, bool log_input, bool full, double eps, int64_t reduction)
--
poisson_nll_loss__ttbbd6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
poisson_nll_loss__ttbbd6 input target log_input full eps reduction =  
  [C.block|Tensor* {
    return new Tensor(at::poisson_nll_loss(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* target), $(bool log_input), $(bool full), $(double eps), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor scalar_tensor(Scalar s, const TensorOptions & options)
--
scalar_tensor__so__1 :: ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
scalar_tensor__so__1 s options =  
  [C.block|Tensor* {
    return new Tensor(at::scalar_tensor(*$fptr-ptr:(Scalar* s), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rand(IntArrayRef size, const TensorOptions & options)
--
rand__ao__1 :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
rand__ao__1 size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::rand(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rand(IntArrayRef size, Generator * generator, const TensorOptions & options)
--
rand__ago__1 :: Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
rand__ago__1 size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::rand(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & rand_out(Tensor & out, IntArrayRef size)
--
rand_out__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
rand_out__ta out size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::rand_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure (out)


-- Tensor & rand_out(Tensor & out, IntArrayRef size, Generator * generator)
--
rand_out__tag :: ForeignPtr CTensor -> Vector Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
rand_out__tag out size generator =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::rand_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator));
   }|] >> pure (out)


-- Tensor rand_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
rand_like__tom__1 :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
rand_like__tom__1 self options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::rand_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randint(int64_t high, IntArrayRef size, const TensorOptions & options)
--
randint__6ao__1 :: Int64 -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint__6ao__1 high size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::randint($(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randint(int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options)
--
randint__6ago__1 :: Int64 -> Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint__6ago__1 high size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::randint($(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randint(int64_t low, int64_t high, IntArrayRef size, const TensorOptions & options)
--
randint__66ao__1 :: Int64 -> Int64 -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint__66ao__1 low high size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::randint($(int64_t low), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randint(int64_t low, int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options)
--
randint__66ago__1 :: Int64 -> Int64 -> Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint__66ago__1 low high size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::randint($(int64_t low), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & randint_out(Tensor & out, int64_t high, IntArrayRef size)
--
randint_out__t6a :: ForeignPtr CTensor -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
randint_out__t6a out high size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::randint_out(*$fptr-ptr:(Tensor* out), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure (out)


-- Tensor & randint_out(Tensor & out, int64_t high, IntArrayRef size, Generator * generator)
--
randint_out__t6ag :: ForeignPtr CTensor -> Int64 -> Vector Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
randint_out__t6ag out high size generator =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::randint_out(*$fptr-ptr:(Tensor* out), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator));
   }|] >> pure (out)


-- Tensor & randint_out(Tensor & out, int64_t low, int64_t high, IntArrayRef size)
--
randint_out__t66a :: ForeignPtr CTensor -> Int64 -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
randint_out__t66a out low high size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::randint_out(*$fptr-ptr:(Tensor* out), $(int64_t low), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure (out)


-- Tensor & randint_out(Tensor & out, int64_t low, int64_t high, IntArrayRef size, Generator * generator)
--
randint_out__t66ag :: ForeignPtr CTensor -> Int64 -> Int64 -> Vector Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
randint_out__t66ag out low high size generator =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::randint_out(*$fptr-ptr:(Tensor* out), $(int64_t low), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator));
   }|] >> pure (out)


-- Tensor randint_like(const Tensor & self, int64_t high, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
randint_like__t6om__1 :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
randint_like__t6om__1 self high options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::randint_like(*$fptr-ptr:(Tensor* self), $(int64_t high), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randint_like(const Tensor & self, int64_t low, int64_t high, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
randint_like__t66om__1 :: ForeignPtr CTensor -> Int64 -> Int64 -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
randint_like__t66om__1 self low high options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::randint_like(*$fptr-ptr:(Tensor* self), $(int64_t low), $(int64_t high), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randn(IntArrayRef size, const TensorOptions & options)
--
randn__ao__1 :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randn__ao__1 size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::randn(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randn(IntArrayRef size, Generator * generator, const TensorOptions & options)
--
randn__ago__1 :: Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randn__ago__1 size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::randn(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & randn_out(Tensor & out, IntArrayRef size)
--
randn_out__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
randn_out__ta out size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::randn_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure (out)


-- Tensor & randn_out(Tensor & out, IntArrayRef size, Generator * generator)
--
randn_out__tag :: ForeignPtr CTensor -> Vector Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
randn_out__tag out size generator =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::randn_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator));
   }|] >> pure (out)


-- Tensor randn_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
randn_like__tom__1 :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
randn_like__tom__1 self options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::randn_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randperm(int64_t n, const TensorOptions & options)
--
randperm__6o__1 :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randperm__6o__1 n options =  
  [C.block|Tensor* {
    return new Tensor(at::randperm($(int64_t n), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor randperm(int64_t n, Generator * generator, const TensorOptions & options)
--
randperm__6go__1 :: Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randperm__6go__1 n generator options =  
  [C.block|Tensor* {
    return new Tensor(at::randperm($(int64_t n), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & randperm_out(Tensor & out, int64_t n)
--
randperm_out__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
randperm_out__t6 out n =  
  [C.block|void {
    at::randperm_out(*$fptr-ptr:(Tensor* out), $(int64_t n));
   }|] >> pure (out)


-- Tensor & randperm_out(Tensor & out, int64_t n, Generator * generator)
--
randperm_out__t6g :: ForeignPtr CTensor -> Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
randperm_out__t6g out n generator =  
  [C.block|void {
    at::randperm_out(*$fptr-ptr:(Tensor* out), $(int64_t n), $(Generator* generator));
   }|] >> pure (out)


-- Tensor range(Scalar start, Scalar end, Scalar step, const TensorOptions & options)
--
range__ssso__1 :: ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
range__ssso__1 start end step options =  
  [C.block|Tensor* {
    return new Tensor(at::range(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(Scalar* step), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor range(Scalar start, Scalar end, const TensorOptions & options)
--
range__sso__1 :: ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
range__sso__1 start end options =  
  [C.block|Tensor* {
    return new Tensor(at::range(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & range_out(Tensor & out, Scalar start, Scalar end, Scalar step)
--
range_out__tsss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
range_out__tsss out start end step =  
  [C.block|void {
    at::range_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(Scalar* step));
   }|] >> pure (out)


-- Tensor reciprocal(const Tensor & self)
--
reciprocal__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
reciprocal__t self =  
  [C.block|Tensor* {
    return new Tensor(at::reciprocal(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & reciprocal_(Tensor & self)
--
reciprocal___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
reciprocal___t self =  
  [C.block|void {
    at::reciprocal_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & reciprocal_out(Tensor & out, const Tensor & self)
--
reciprocal_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
reciprocal_out__tt out self =  
  [C.block|void {
    at::reciprocal_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor neg(const Tensor & self)
--
neg__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
neg__t self =  
  [C.block|Tensor* {
    return new Tensor(at::neg(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & neg_(Tensor & self)
--
neg___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
neg___t self =  
  [C.block|void {
    at::neg_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & neg_out(Tensor & out, const Tensor & self)
--
neg_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
neg_out__tt out self =  
  [C.block|void {
    at::neg_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor repeat_interleave(const Tensor & repeats)
--
repeat_interleave__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
repeat_interleave__t repeats =  
  [C.block|Tensor* {
    return new Tensor(at::repeat_interleave(*$fptr-ptr:(Tensor* repeats)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor repeat_interleave(const Tensor & self, const Tensor & repeats, c10::optional<int64_t> dim)
--
repeat_interleave__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Maybe Int64 -> IO (ForeignPtr CTensor)
repeat_interleave__tt6 self repeats dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in 
  [C.block|Tensor* {
    return new Tensor(at::repeat_interleave(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* repeats), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor repeat_interleave(const Tensor & self, int64_t repeats, c10::optional<int64_t> dim)
--
repeat_interleave__t66 :: ForeignPtr CTensor -> Int64 -> Maybe Int64 -> IO (ForeignPtr CTensor)
repeat_interleave__t66 self repeats dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in 
  [C.block|Tensor* {
    return new Tensor(at::repeat_interleave(*$fptr-ptr:(Tensor* self), $(int64_t repeats), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor reshape(const Tensor & self, IntArrayRef shape)
--
reshape__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
reshape__ta self shape =  V.unsafeWith shape $ \shape__array -> let shape__size = fromIntegral (V.length shape) in 
  [C.block|Tensor* {
    return new Tensor(at::reshape(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* shape__array), $(size_t shape__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _mkldnn_reshape(const Tensor & self, IntArrayRef shape)
--
_mkldnn_reshape__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
_mkldnn_reshape__ta self shape =  V.unsafeWith shape $ \shape__array -> let shape__size = fromIntegral (V.length shape) in 
  [C.block|Tensor* {
    return new Tensor(at::_mkldnn_reshape(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* shape__array), $(size_t shape__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor round(const Tensor & self)
--
round__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
round__t self =  
  [C.block|Tensor* {
    return new Tensor(at::round(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & round_(Tensor & self)
--
round___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
round___t self =  
  [C.block|void {
    at::round_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & round_out(Tensor & out, const Tensor & self)
--
round_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
round_out__tt out self =  
  [C.block|void {
    at::round_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator)
--
rrelu__tssbg :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
rrelu__tssbg self lower upper training generator =  
  [C.block|Tensor* {
    return new Tensor(at::rrelu(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* lower), *$fptr-ptr:(Scalar* upper), $(bool training), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator)
--
rrelu___tssbg :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
rrelu___tssbg self lower upper training generator =  
  [C.block|void {
    at::rrelu_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* lower), *$fptr-ptr:(Scalar* upper), $(bool training), $(Generator* generator));
   }|] >> pure self


-- Tensor relu(const Tensor & self)
--
relu__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
relu__t self =  
  [C.block|Tensor* {
    return new Tensor(at::relu(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & relu_(Tensor & self)
--
relu___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
relu___t self =  
  [C.block|void {
    at::relu_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor prelu(const Tensor & self, const Tensor & weight)
--
prelu__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
prelu__tt self weight =  
  [C.block|Tensor* {
    return new Tensor(at::prelu(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor gelu(const Tensor & self)
--
gelu__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
gelu__t self =  
  [C.block|Tensor* {
    return new Tensor(at::gelu(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hardshrink(const Tensor & self, Scalar lambd)
--
hardshrink__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
hardshrink__ts self lambd =  
  [C.block|Tensor* {
    return new Tensor(at::hardshrink(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* lambd)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rsqrt(const Tensor & self)
--
rsqrt__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rsqrt__t self =  
  [C.block|Tensor* {
    return new Tensor(at::rsqrt(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & rsqrt_(Tensor & self)
--
rsqrt___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rsqrt___t self =  
  [C.block|void {
    at::rsqrt_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & rsqrt_out(Tensor & out, const Tensor & self)
--
rsqrt_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rsqrt_out__tt out self =  
  [C.block|void {
    at::rsqrt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor select(const Tensor & self, int64_t dim, int64_t index)
--
select__t66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
select__t66 self dim index =  
  [C.block|Tensor* {
    return new Tensor(at::select(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(int64_t index)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor selu(const Tensor & self)
--
selu__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
selu__t self =  
  [C.block|Tensor* {
    return new Tensor(at::selu(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & selu_(Tensor & self)
--
selu___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
selu___t self =  
  [C.block|void {
    at::selu_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor celu(const Tensor & self, Scalar alpha)
--
celu__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
celu__ts self alpha =  
  [C.block|Tensor* {
    return new Tensor(at::celu(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & celu_(Tensor & self, Scalar alpha)
--
celu___ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
celu___ts self alpha =  
  [C.block|void {
    at::celu_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor sigmoid(const Tensor & self)
--
sigmoid__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sigmoid__t self =  
  [C.block|Tensor* {
    return new Tensor(at::sigmoid(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sigmoid_(Tensor & self)
--
sigmoid___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sigmoid___t self =  
  [C.block|void {
    at::sigmoid_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & sigmoid_out(Tensor & out, const Tensor & self)
--
sigmoid_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sigmoid_out__tt out self =  
  [C.block|void {
    at::sigmoid_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor sin(const Tensor & self)
--
sin__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sin__t self =  
  [C.block|Tensor* {
    return new Tensor(at::sin(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sin_(Tensor & self)
--
sin___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sin___t self =  
  [C.block|void {
    at::sin_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & sin_out(Tensor & out, const Tensor & self)
--
sin_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sin_out__tt out self =  
  [C.block|void {
    at::sin_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor sinh(const Tensor & self)
--
sinh__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sinh__t self =  
  [C.block|Tensor* {
    return new Tensor(at::sinh(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sinh_(Tensor & self)
--
sinh___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sinh___t self =  
  [C.block|void {
    at::sinh_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & sinh_out(Tensor & out, const Tensor & self)
--
sinh_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sinh_out__tt out self =  
  [C.block|void {
    at::sinh_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor detach(const Tensor & self)
--
detach__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
detach__t self =  
  [C.block|Tensor* {
    return new Tensor(at::detach(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & detach_(Tensor & self)
--
detach___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
detach___t self =  
  [C.block|void {
    at::detach_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- int64_t size(const Tensor & self, int64_t dim)
--
size__t6 :: ForeignPtr CTensor -> Int64 -> IO (Int64)
size__t6 self dim =  
  [C.block|int64_t {
    return at::size(*$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|]


-- Tensor slice(const Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step)
--
slice__t6666 :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
slice__t6666 self dim start end step =  
  [C.block|Tensor* {
    return new Tensor(at::slice(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(int64_t start), $(int64_t end), $(int64_t step)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> slogdet(const Tensor & self)
--
slogdet__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
slogdet__t self =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::slogdet(*$fptr-ptr:(Tensor* self)));
   }|] >>= unTupleTensorTensor


-- Tensor smm(const Tensor & self, const Tensor & mat2)
--
smm__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
smm__tt self mat2 =  
  [C.block|Tensor* {
    return new Tensor(at::smm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor softmax(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype)
--
softmax__t6s :: ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
softmax__t6s self dim dtype =  
  [C.block|Tensor* {
    return new Tensor(at::softmax(*$fptr-ptr:(Tensor* self), $(int64_t dim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _softmax(const Tensor & self, int64_t dim, bool half_to_float)
--
_softmax__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
_softmax__t6b self dim half_to_float =  
  [C.block|Tensor* {
    return new Tensor(at::_softmax(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool half_to_float)));
   }|] >>= newForeignPtr deleteTensor


-- std::vector<Tensor> split(const Tensor & self, int64_t split_size, int64_t dim)
--
split__t66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (Vector (Ptr CTensor))
split__t66 self split_size dim =  
  [C.block|void* {
    return (void*)new std::vector<Tensor>(at::split(*$fptr-ptr:(Tensor* self), $(int64_t split_size), $(int64_t dim)));
   }|] >>= unVectorTensor


-- std::vector<Tensor> split_with_sizes(const Tensor & self, IntArrayRef split_sizes, int64_t dim)
--
split_with_sizes__ta6 :: ForeignPtr CTensor -> Vector Int64 -> Int64 -> IO (Vector (Ptr CTensor))
split_with_sizes__ta6 self split_sizes dim =  V.unsafeWith split_sizes $ \split_sizes__array -> let split_sizes__size = fromIntegral (V.length split_sizes) in 
  [C.block|void* {
    return (void*)new std::vector<Tensor>(at::split_with_sizes(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* split_sizes__array), $(size_t split_sizes__size)), $(int64_t dim)));
   }|] >>= unVectorTensor


-- Tensor squeeze(const Tensor & self)
--
squeeze__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
squeeze__t self =  
  [C.block|Tensor* {
    return new Tensor(at::squeeze(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor squeeze(const Tensor & self, int64_t dim)
--
squeeze__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
squeeze__t6 self dim =  
  [C.block|Tensor* {
    return new Tensor(at::squeeze(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha)
--
sspaddmm__tttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sspaddmm__tttss self mat1 mat2 beta alpha =  
  [C.block|Tensor* {
    return new Tensor(at::sspaddmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sspaddmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha)
--
sspaddmm_out__ttttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sspaddmm_out__ttttss out self mat1 mat2 beta alpha =  
  [C.block|void {
    at::sspaddmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


-- Tensor stack(TensorList tensors, int64_t dim)
--
stack__l6 :: Vector (Ptr CTensor) -> Int64 -> IO (ForeignPtr CTensor)
stack__l6 tensors dim =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in 
  [C.block|Tensor* {
    return new Tensor(at::stack(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size)), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & stack_out(Tensor & out, TensorList tensors, int64_t dim)
--
stack_out__tl6 :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Int64 -> IO (ForeignPtr CTensor)
stack_out__tl6 out tensors dim =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in 
  [C.block|void {
    at::stack_out(*$fptr-ptr:(Tensor* out), pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size)), $(int64_t dim));
   }|] >> pure (out)


-- Tensor stft(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided)
--
stft__t666tbb__1 :: ForeignPtr CTensor -> Int64 -> Maybe Int64 -> Maybe Int64 -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor)
stft__t666tbb__1 self n_fft hop_length win_length window normalized onesided =  let (hop_length__is_present, hop_length__value) = splitMaybe hop_length 0 in let (win_length__is_present, win_length__value) = splitMaybe win_length 0 in 
  [C.block|Tensor* {
    return new Tensor(at::stft(*$fptr-ptr:(Tensor* self), $(int64_t n_fft), ($(bool hop_length__is_present) ? make_optional($(int64_t hop_length__value)) : c10::nullopt), ($(bool win_length__is_present) ? make_optional($(int64_t win_length__value)) : c10::nullopt), *$fptr-ptr:(Tensor* window), $(bool normalized), $(bool onesided)));
   }|] >>= newForeignPtr deleteTensor


-- int64_t stride(const Tensor & self, int64_t dim)
--
stride__t6 :: ForeignPtr CTensor -> Int64 -> IO (Int64)
stride__t6 self dim =  
  [C.block|int64_t {
    return at::stride(*$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|]


-- Tensor sum(const Tensor & self, c10::optional<ScalarType> dtype)
--
sum__ts :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
sum__ts self dtype =  
  [C.block|Tensor* {
    return new Tensor(at::sum(*$fptr-ptr:(Tensor* self), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor sum(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype)
--
sum__tabs :: ForeignPtr CTensor -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
sum__tabs self dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::sum(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sum_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype)
--
sum_out__ttabs :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
sum_out__ttabs out self dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|void {
    at::sum_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype)));
   }|] >> pure (out)


-- Tensor sqrt(const Tensor & self)
--
sqrt__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sqrt__t self =  
  [C.block|Tensor* {
    return new Tensor(at::sqrt(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sqrt_(Tensor & self)
--
sqrt___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sqrt___t self =  
  [C.block|void {
    at::sqrt_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & sqrt_out(Tensor & out, const Tensor & self)
--
sqrt_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sqrt_out__tt out self =  
  [C.block|void {
    at::sqrt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor square(const Tensor & self)
--
square__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
square__t self =  
  [C.block|Tensor* {
    return new Tensor(at::square(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & square_(Tensor & self)
--
square___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
square___t self =  
  [C.block|void {
    at::square_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor std(const Tensor & self, bool unbiased)
--
std__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
std__tb self unbiased =  
  [C.block|Tensor* {
    return new Tensor(at::std(*$fptr-ptr:(Tensor* self), $(bool unbiased)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor std(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim)
--
std__tabb :: ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
std__tabb self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::std(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> std_mean(const Tensor & self, bool unbiased)
--
std_mean__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
std_mean__tb self unbiased =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::std_mean(*$fptr-ptr:(Tensor* self), $(bool unbiased)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> std_mean(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim)
--
std_mean__tabb :: ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
std_mean__tabb self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::std_mean(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- Tensor & std_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim)
--
std_out__ttabb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
std_out__ttabb out self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|void {
    at::std_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim));
   }|] >> pure (out)


-- Tensor prod(const Tensor & self, c10::optional<ScalarType> dtype)
--
prod__ts :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
prod__ts self dtype =  
  [C.block|Tensor* {
    return new Tensor(at::prod(*$fptr-ptr:(Tensor* self), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor prod(const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype)
--
prod__t6bs :: ForeignPtr CTensor -> Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
prod__t6bs self dim keepdim dtype =  
  [C.block|Tensor* {
    return new Tensor(at::prod(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & prod_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype)
--
prod_out__tt6bs :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
prod_out__tt6bs out self dim keepdim dtype =  
  [C.block|void {
    at::prod_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype)));
   }|] >> pure (out)


-- Tensor t(const Tensor & self)
--
t__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
t__t self =  
  [C.block|Tensor* {
    return new Tensor(at::t(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor tan(const Tensor & self)
--
tan__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tan__t self =  
  [C.block|Tensor* {
    return new Tensor(at::tan(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & tan_(Tensor & self)
--
tan___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tan___t self =  
  [C.block|void {
    at::tan_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & tan_out(Tensor & out, const Tensor & self)
--
tan_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tan_out__tt out self =  
  [C.block|void {
    at::tan_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor tanh(const Tensor & self)
--
tanh__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tanh__t self =  
  [C.block|Tensor* {
    return new Tensor(at::tanh(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & tanh_(Tensor & self)
--
tanh___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tanh___t self =  
  [C.block|void {
    at::tanh_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & tanh_out(Tensor & out, const Tensor & self)
--
tanh_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tanh_out__tt out self =  
  [C.block|void {
    at::tanh_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor tensordot(const Tensor & self, const Tensor & other, IntArrayRef dims_self, IntArrayRef dims_other)
--
tensordot__ttaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
tensordot__ttaa self other dims_self dims_other =  V.unsafeWith dims_self $ \dims_self__array -> let dims_self__size = fromIntegral (V.length dims_self) in V.unsafeWith dims_other $ \dims_other__array -> let dims_other__size = fromIntegral (V.length dims_other) in 
  [C.block|Tensor* {
    return new Tensor(at::tensordot(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), ArrayRef<int64_t>($(int64_t* dims_self__array), $(size_t dims_self__size)), ArrayRef<int64_t>($(int64_t* dims_other__array), $(size_t dims_other__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor threshold(const Tensor & self, Scalar threshold, Scalar value)
--
threshold__tss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
threshold__tss self threshold value =  
  [C.block|Tensor* {
    return new Tensor(at::threshold(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* threshold), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & threshold_(Tensor & self, Scalar threshold, Scalar value)
--
threshold___tss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
threshold___tss self threshold value =  
  [C.block|void {
    at::threshold_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* threshold), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


-- Tensor & threshold_out(Tensor & out, const Tensor & self, Scalar threshold, Scalar value)
--
threshold_out__ttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
threshold_out__ttss out self threshold value =  
  [C.block|void {
    at::threshold_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* threshold), *$fptr-ptr:(Scalar* value));
   }|] >> pure (out)


-- Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1)
--
transpose__t66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
transpose__t66 self dim0 dim1 =  
  [C.block|Tensor* {
    return new Tensor(at::transpose(*$fptr-ptr:(Tensor* self), $(int64_t dim0), $(int64_t dim1)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _mkldnn_transpose(const Tensor & self, int64_t dim0, int64_t dim1)
--
_mkldnn_transpose__t66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
_mkldnn_transpose__t66 self dim0 dim1 =  
  [C.block|Tensor* {
    return new Tensor(at::_mkldnn_transpose(*$fptr-ptr:(Tensor* self), $(int64_t dim0), $(int64_t dim1)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & _mkldnn_transpose_(Tensor & self, int64_t dim0, int64_t dim1)
--
_mkldnn_transpose___t66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
_mkldnn_transpose___t66 self dim0 dim1 =  
  [C.block|void {
    at::_mkldnn_transpose_(*$fptr-ptr:(Tensor* self), $(int64_t dim0), $(int64_t dim1));
   }|] >> pure self


-- Tensor one_hot(const Tensor & self, int64_t num_classes)
--
one_hot__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
one_hot__t6 self num_classes =  
  [C.block|Tensor* {
    return new Tensor(at::one_hot(*$fptr-ptr:(Tensor* self), $(int64_t num_classes)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor flip(const Tensor & self, IntArrayRef dims)
--
flip__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
flip__ta self dims =  V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in 
  [C.block|Tensor* {
    return new Tensor(at::flip(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor roll(const Tensor & self, IntArrayRef shifts, IntArrayRef dims)
--
roll__taa__1 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
roll__taa__1 self shifts dims =  V.unsafeWith shifts $ \shifts__array -> let shifts__size = fromIntegral (V.length shifts) in V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in 
  [C.block|Tensor* {
    return new Tensor(at::roll(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* shifts__array), $(size_t shifts__size)), ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rot90(const Tensor & self, int64_t k, IntArrayRef dims)
--
rot90__t6a__1 :: ForeignPtr CTensor -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
rot90__t6a__1 self k dims =  V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in 
  [C.block|Tensor* {
    return new Tensor(at::rot90(*$fptr-ptr:(Tensor* self), $(int64_t k), ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor trapz(const Tensor & y, const Tensor & x, int64_t dim)
--
trapz__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
trapz__tt6 y x dim =  
  [C.block|Tensor* {
    return new Tensor(at::trapz(*$fptr-ptr:(Tensor* y), *$fptr-ptr:(Tensor* x), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor trapz(const Tensor & y, double dx, int64_t dim)
--
trapz__td6 :: ForeignPtr CTensor -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
trapz__td6 y dx dim =  
  [C.block|Tensor* {
    return new Tensor(at::trapz(*$fptr-ptr:(Tensor* y), $(double dx), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _trilinear(const Tensor & i1, const Tensor & i2, const Tensor & i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim)
--
_trilinear__tttaaaa6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
_trilinear__tttaaaa6 i1 i2 i3 expand1 expand2 expand3 sumdim unroll_dim =  V.unsafeWith expand1 $ \expand1__array -> let expand1__size = fromIntegral (V.length expand1) in V.unsafeWith expand2 $ \expand2__array -> let expand2__size = fromIntegral (V.length expand2) in V.unsafeWith expand3 $ \expand3__array -> let expand3__size = fromIntegral (V.length expand3) in V.unsafeWith sumdim $ \sumdim__array -> let sumdim__size = fromIntegral (V.length sumdim) in 
  [C.block|Tensor* {
    return new Tensor(at::_trilinear(*$fptr-ptr:(Tensor* i1), *$fptr-ptr:(Tensor* i2), *$fptr-ptr:(Tensor* i3), ArrayRef<int64_t>($(int64_t* expand1__array), $(size_t expand1__size)), ArrayRef<int64_t>($(int64_t* expand2__array), $(size_t expand2__size)), ArrayRef<int64_t>($(int64_t* expand3__array), $(size_t expand3__size)), ArrayRef<int64_t>($(int64_t* sumdim__array), $(size_t sumdim__size)), $(int64_t unroll_dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction)
--
triplet_margin_loss__tttdddb6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> CDouble -> CBool -> Int64 -> IO (ForeignPtr CTensor)
triplet_margin_loss__tttdddb6 anchor positive negative margin p eps swap reduction =  
  [C.block|Tensor* {
    return new Tensor(at::triplet_margin_loss(*$fptr-ptr:(Tensor* anchor), *$fptr-ptr:(Tensor* positive), *$fptr-ptr:(Tensor* negative), $(double margin), $(double p), $(double eps), $(bool swap), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor true_divide(const Tensor & self, const Tensor & other)
--
true_divide__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
true_divide__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::true_divide(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & true_divide_out(Tensor & out, const Tensor & self, const Tensor & other)
--
true_divide_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
true_divide_out__ttt out self other =  
  [C.block|void {
    at::true_divide_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor true_divide(const Tensor & self, Scalar other)
--
true_divide__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
true_divide__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::true_divide(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor trunc(const Tensor & self)
--
trunc__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
trunc__t self =  
  [C.block|Tensor* {
    return new Tensor(at::trunc(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & trunc_(Tensor & self)
--
trunc___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
trunc___t self =  
  [C.block|void {
    at::trunc_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & trunc_out(Tensor & out, const Tensor & self)
--
trunc_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
trunc_out__tt out self =  
  [C.block|void {
    at::trunc_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- bool _has_compatible_shallow_copy_type(const Tensor & self, const Tensor & from)
--
_has_compatible_shallow_copy_type__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (CBool)
_has_compatible_shallow_copy_type__tt self from =  
  [C.block|bool {
    return at::_has_compatible_shallow_copy_type(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* from));
   }|]


-- std::tuple<Tensor,Tensor> _unique(const Tensor & self, bool sorted, bool return_inverse)
--
_unique__tbb :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_unique__tbb self sorted return_inverse =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_unique(*$fptr-ptr:(Tensor* self), $(bool sorted), $(bool return_inverse)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor,Tensor> unique_dim(const Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts)
--
unique_dim__t6bbb :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
unique_dim__t6bbb self dim sorted return_inverse return_counts =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::unique_dim(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool sorted), $(bool return_inverse), $(bool return_counts)));
   }|] >>= unTupleTensorTensorTensor


-- std::tuple<Tensor,Tensor,Tensor> unique_consecutive(const Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim)
--
unique_consecutive__tbb6 :: ForeignPtr CTensor -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
unique_consecutive__tbb6 self return_inverse return_counts dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::unique_consecutive(*$fptr-ptr:(Tensor* self), $(bool return_inverse), $(bool return_counts), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt)));
   }|] >>= unTupleTensorTensorTensor


-- std::tuple<Tensor,Tensor,Tensor> unique_dim_consecutive(const Tensor & self, int64_t dim, bool return_inverse, bool return_counts)
--
unique_dim_consecutive__t6bb :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
unique_dim_consecutive__t6bb self dim return_inverse return_counts =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::unique_dim_consecutive(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool return_inverse), $(bool return_counts)));
   }|] >>= unTupleTensorTensorTensor


-- std::tuple<Tensor,Tensor,Tensor> _unique2(const Tensor & self, bool sorted, bool return_inverse, bool return_counts)
--
_unique2__tbbb :: ForeignPtr CTensor -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_unique2__tbbb self sorted return_inverse return_counts =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::_unique2(*$fptr-ptr:(Tensor* self), $(bool sorted), $(bool return_inverse), $(bool return_counts)));
   }|] >>= unTupleTensorTensorTensor


-- Tensor _unsafe_view(const Tensor & self, IntArrayRef size)
--
_unsafe_view__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
_unsafe_view__ta self size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::_unsafe_view(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor unsqueeze(const Tensor & self, int64_t dim)
--
unsqueeze__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
unsqueeze__t6 self dim =  
  [C.block|Tensor* {
    return new Tensor(at::unsqueeze(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor var(const Tensor & self, bool unbiased)
--
var__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
var__tb self unbiased =  
  [C.block|Tensor* {
    return new Tensor(at::var(*$fptr-ptr:(Tensor* self), $(bool unbiased)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor var(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim)
--
var__tabb :: ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
var__tabb self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::var(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & var_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim)
--
var_out__ttabb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
var_out__ttabb out self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|void {
    at::var_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim));
   }|] >> pure (out)


-- std::tuple<Tensor,Tensor> var_mean(const Tensor & self, bool unbiased)
--
var_mean__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
var_mean__tb self unbiased =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::var_mean(*$fptr-ptr:(Tensor* self), $(bool unbiased)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> var_mean(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim)
--
var_mean__tabb :: ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
var_mean__tabb self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::var_mean(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- Tensor where(const Tensor & condition, const Tensor & self, const Tensor & other)
--
where__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
where__ttt condition self other =  
  [C.block|Tensor* {
    return new Tensor(at::where(*$fptr-ptr:(Tensor* condition), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- std::vector<Tensor> where(const Tensor & condition)
--
where__t :: ForeignPtr CTensor -> IO (Vector (Ptr CTensor))
where__t condition =  
  [C.block|void* {
    return (void*)new std::vector<Tensor>(at::where(*$fptr-ptr:(Tensor* condition)));
   }|] >>= unVectorTensor


-- Tensor _s_where(const Tensor & condition, const Tensor & self, const Tensor & other)
--
_s_where__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_s_where__ttt condition self other =  
  [C.block|Tensor* {
    return new Tensor(at::_s_where(*$fptr-ptr:(Tensor* condition), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor norm_except_dim(const Tensor & v, int64_t pow, int64_t dim)
--
norm_except_dim__t66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
norm_except_dim__t66 v pow dim =  
  [C.block|Tensor* {
    return new Tensor(at::norm_except_dim(*$fptr-ptr:(Tensor* v), $(int64_t pow), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _weight_norm(const Tensor & v, const Tensor & g, int64_t dim)
--
_weight_norm__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_weight_norm__tt6 v g dim =  
  [C.block|Tensor* {
    return new Tensor(at::_weight_norm(*$fptr-ptr:(Tensor* v), *$fptr-ptr:(Tensor* g), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> _weight_norm_cuda_interface(const Tensor & v, const Tensor & g, int64_t dim)
--
_weight_norm_cuda_interface__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_weight_norm_cuda_interface__tt6 v g dim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_weight_norm_cuda_interface(*$fptr-ptr:(Tensor* v), *$fptr-ptr:(Tensor* g), $(int64_t dim)));
   }|] >>= unTupleTensorTensor


-- Tensor zeros(IntArrayRef size, const TensorOptions & options)
--
zeros__ao__1 :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
zeros__ao__1 size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::zeros(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & zeros_out(Tensor & out, IntArrayRef size)
--
zeros_out__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
zeros_out__ta out size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::zeros_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure (out)


-- Tensor zeros_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format)
--
zeros_like__tom__1 :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
zeros_like__tom__1 self options memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::zeros_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _standard_gamma_grad(const Tensor & self, const Tensor & output)
--
_standard_gamma_grad__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_standard_gamma_grad__tt self output =  
  [C.block|Tensor* {
    return new Tensor(at::_standard_gamma_grad(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* output)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _standard_gamma(const Tensor & self, Generator * generator)
--
_standard_gamma__tg :: ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
_standard_gamma__tg self generator =  
  [C.block|Tensor* {
    return new Tensor(at::_standard_gamma(*$fptr-ptr:(Tensor* self), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total)
--
_dirichlet_grad__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_dirichlet_grad__ttt x alpha total =  
  [C.block|Tensor* {
    return new Tensor(at::_dirichlet_grad(*$fptr-ptr:(Tensor* x), *$fptr-ptr:(Tensor* alpha), *$fptr-ptr:(Tensor* total)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _sample_dirichlet(const Tensor & self, Generator * generator)
--
_sample_dirichlet__tg :: ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
_sample_dirichlet__tg self generator =  
  [C.block|Tensor* {
    return new Tensor(at::_sample_dirichlet(*$fptr-ptr:(Tensor* self), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor poisson(const Tensor & self, Generator * generator)
--
poisson__tg :: ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
poisson__tg self generator =  
  [C.block|Tensor* {
    return new Tensor(at::poisson(*$fptr-ptr:(Tensor* self), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor native_norm(const Tensor & self, Scalar p)
--
native_norm__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
native_norm__ts self p =  
  [C.block|Tensor* {
    return new Tensor(at::native_norm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _sparse_sum(const Tensor & self)
--
_sparse_sum__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_sparse_sum__t self =  
  [C.block|Tensor* {
    return new Tensor(at::_sparse_sum(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _sparse_sum(const Tensor & self, ScalarType dtype)
--
_sparse_sum__ts :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
_sparse_sum__ts self dtype =  
  [C.block|Tensor* {
    return new Tensor(at::_sparse_sum(*$fptr-ptr:(Tensor* self), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _sparse_sum(const Tensor & self, IntArrayRef dim)
--
_sparse_sum__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
_sparse_sum__ta self dim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::_sparse_sum(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _sparse_sum(const Tensor & self, IntArrayRef dim, ScalarType dtype)
--
_sparse_sum__tas :: ForeignPtr CTensor -> Vector Int64 -> Int8 -> IO (ForeignPtr CTensor)
_sparse_sum__tas self dim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::_sparse_sum(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor norm(const Tensor & self, c10::optional<Scalar> p, ScalarType dtype)
--
norm__tss :: ForeignPtr CTensor -> ForeignPtr CScalar -> Int8 -> IO (ForeignPtr CTensor)
norm__tss self p dtype =  
  [C.block|Tensor* {
    return new Tensor(at::norm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor norm(const Tensor & self, Scalar p)
--
norm__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
norm__ts self p =  
  [C.block|Tensor* {
    return new Tensor(at::norm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor norm(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype)
--
norm__tsabs :: ForeignPtr CTensor -> ForeignPtr CScalar -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
norm__tsabs self p dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::norm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor norm(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim)
--
norm__tsab :: ForeignPtr CTensor -> ForeignPtr CScalar -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
norm__tsab self p dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::norm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype)
--
norm_out__ttsabs :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
norm_out__ttsabs out self p dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|void {
    at::norm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype)));
   }|] >> pure (out)


-- Tensor & norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim)
--
norm_out__ttsab :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
norm_out__ttsab out self p dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|void {
    at::norm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim));
   }|] >> pure (out)


-- Tensor frobenius_norm(const Tensor & self)
--
frobenius_norm__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
frobenius_norm__t self =  
  [C.block|Tensor* {
    return new Tensor(at::frobenius_norm(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor frobenius_norm(const Tensor & self, IntArrayRef dim, bool keepdim)
--
frobenius_norm__tab :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
frobenius_norm__tab self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::frobenius_norm(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & frobenius_norm_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim)
--
frobenius_norm_out__ttab :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
frobenius_norm_out__ttab out self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|void {
    at::frobenius_norm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim));
   }|] >> pure (out)


-- Tensor nuclear_norm(const Tensor & self, bool keepdim)
--
nuclear_norm__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
nuclear_norm__tb self keepdim =  
  [C.block|Tensor* {
    return new Tensor(at::nuclear_norm(*$fptr-ptr:(Tensor* self), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & nuclear_norm_out(Tensor & out, const Tensor & self, bool keepdim)
--
nuclear_norm_out__ttb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
nuclear_norm_out__ttb out self keepdim =  
  [C.block|void {
    at::nuclear_norm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(bool keepdim));
   }|] >> pure (out)


-- Tensor nuclear_norm(const Tensor & self, IntArrayRef dim, bool keepdim)
--
nuclear_norm__tab :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
nuclear_norm__tab self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor(at::nuclear_norm(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & nuclear_norm_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim)
--
nuclear_norm_out__ttab :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
nuclear_norm_out__ttab out self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|void {
    at::nuclear_norm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim));
   }|] >> pure (out)


-- Tensor clone(const Tensor & self, c10::optional<MemoryFormat> memory_format)
--
clone__tm :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
clone__tm self memory_format =  
  [C.block|Tensor* {
    return new Tensor(at::clone(*$fptr-ptr:(Tensor* self), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & resize_as_(Tensor & self, const Tensor & the_template, c10::optional<MemoryFormat> memory_format)
--
resize_as___ttm :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
resize_as___ttm self the_template memory_format =  
  [C.block|void {
    at::resize_as_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* the_template), static_cast<MemoryFormat>($(int8_t memory_format)));
   }|] >> pure self


-- Tensor & pow_out(Tensor & out, const Tensor & self, Scalar exponent)
--
pow_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
pow_out__tts out self exponent =  
  [C.block|void {
    at::pow_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* exponent));
   }|] >> pure (out)


-- Tensor pow(const Tensor & self, Scalar exponent)
--
pow__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
pow__ts self exponent =  
  [C.block|Tensor* {
    return new Tensor(at::pow(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* exponent)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & zero_(Tensor & self)
--
zero___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
zero___t self =  
  [C.block|void {
    at::zero_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & sub_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha)
--
sub_out__ttts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sub_out__ttts out self other alpha =  
  [C.block|void {
    at::sub_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


-- Tensor sub(const Tensor & self, const Tensor & other, Scalar alpha)
--
sub__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sub__tts self other alpha =  
  [C.block|Tensor* {
    return new Tensor(at::sub(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor sub(const Tensor & self, Scalar other, Scalar alpha)
--
sub__tss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sub__tss self other alpha =  
  [C.block|Tensor* {
    return new Tensor(at::sub(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rsub(const Tensor & self, const Tensor & other, Scalar alpha)
--
rsub__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
rsub__tts self other alpha =  
  [C.block|Tensor* {
    return new Tensor(at::rsub(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rsub(const Tensor & self, Scalar other, Scalar alpha)
--
rsub__tss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
rsub__tss self other alpha =  
  [C.block|Tensor* {
    return new Tensor(at::rsub(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _sparse_addmm(const Tensor & self, const Tensor & sparse, const Tensor & dense, Scalar beta, Scalar alpha)
--
_sparse_addmm__tttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_sparse_addmm__tttss self sparse dense beta alpha =  
  [C.block|Tensor* {
    return new Tensor(at::_sparse_addmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* sparse), *$fptr-ptr:(Tensor* dense), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & addmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha)
--
addmm_out__ttttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmm_out__ttttss out self mat1 mat2 beta alpha =  
  [C.block|void {
    at::addmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


-- Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha)
--
addmm__tttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmm__tttss self mat1 mat2 beta alpha =  
  [C.block|Tensor* {
    return new Tensor(at::addmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor sparse_coo_tensor(IntArrayRef size, const TensorOptions & options)
--
sparse_coo_tensor__ao :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
sparse_coo_tensor__ao size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::sparse_coo_tensor(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values, const TensorOptions & options)
--
sparse_coo_tensor__tto__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
sparse_coo_tensor__tto__1 indices values options =  
  [C.block|Tensor* {
    return new Tensor(at::sparse_coo_tensor(*$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* values), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options)
--
sparse_coo_tensor__ttao__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
sparse_coo_tensor__ttao__1 indices values size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::sparse_coo_tensor(*$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* values), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _sparse_coo_tensor_unsafe(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options)
--
_sparse_coo_tensor_unsafe__ttao__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
_sparse_coo_tensor_unsafe__ttao__1 indices values size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::_sparse_coo_tensor_unsafe(*$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* values), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const TensorOptions & options)
--
_sparse_coo_tensor_with_dims__66ao :: Int64 -> Int64 -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
_sparse_coo_tensor_with_dims__66ao sparse_dim dense_dim size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::_sparse_coo_tensor_with_dims($(int64_t sparse_dim), $(int64_t dense_dim), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, const TensorOptions & options)
--
_sparse_coo_tensor_with_dims_and_tensors__66atto :: Int64 -> Int64 -> Vector Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
_sparse_coo_tensor_with_dims_and_tensors__66atto sparse_dim dense_dim size indices values options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::_sparse_coo_tensor_with_dims_and_tensors($(int64_t sparse_dim), $(int64_t dense_dim), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* values), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & hspmm_out(Tensor & out, const Tensor & mat1, const Tensor & mat2)
--
hspmm_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
hspmm_out__ttt out mat1 mat2 =  
  [C.block|void {
    at::hspmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2));
   }|] >> pure (out)


-- Tensor hspmm(const Tensor & mat1, const Tensor & mat2)
--
hspmm__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
hspmm__tt mat1 mat2 =  
  [C.block|Tensor* {
    return new Tensor(at::hspmm(*$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & copy_sparse_to_sparse_(Tensor & self, const Tensor & src, bool non_blocking)
--
copy_sparse_to_sparse___ttb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
copy_sparse_to_sparse___ttb self src non_blocking =  
  [C.block|void {
    at::copy_sparse_to_sparse_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* src), $(bool non_blocking));
   }|] >> pure self


-- std::vector<Tensor> unbind(const Tensor & self, int64_t dim)
--
unbind__t6 :: ForeignPtr CTensor -> Int64 -> IO (Vector (Ptr CTensor))
unbind__t6 self dim =  
  [C.block|void* {
    return (void*)new std::vector<Tensor>(at::unbind(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= unVectorTensor


-- Tensor mkldnn_reorder_conv2d_weight(const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
--
mkldnn_reorder_conv2d_weight__taaa6 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
mkldnn_reorder_conv2d_weight__taaa6 self padding stride dilation groups =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::mkldnn_reorder_conv2d_weight(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor quantize_per_tensor(const Tensor & self, double scale, int64_t zero_point, ScalarType dtype)
--
quantize_per_tensor__td6s :: ForeignPtr CTensor -> CDouble -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
quantize_per_tensor__td6s self scale zero_point dtype =  
  [C.block|Tensor* {
    return new Tensor(at::quantize_per_tensor(*$fptr-ptr:(Tensor* self), $(double scale), $(int64_t zero_point), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor quantize_per_channel(const Tensor & self, const Tensor & scales, const Tensor & zero_points, int64_t axis, ScalarType dtype)
--
quantize_per_channel__ttt6s :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
quantize_per_channel__ttt6s self scales zero_points axis dtype =  
  [C.block|Tensor* {
    return new Tensor(at::quantize_per_channel(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* scales), *$fptr-ptr:(Tensor* zero_points), $(int64_t axis), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor dequantize(const Tensor & self)
--
dequantize__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
dequantize__t self =  
  [C.block|Tensor* {
    return new Tensor(at::dequantize(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- double q_scale(const Tensor & self)
--
q_scale__t :: ForeignPtr CTensor -> IO (CDouble)
q_scale__t self =  
  [C.block|double {
    return at::q_scale(*$fptr-ptr:(Tensor* self));
   }|]


-- int64_t q_zero_point(const Tensor & self)
--
q_zero_point__t :: ForeignPtr CTensor -> IO (Int64)
q_zero_point__t self =  
  [C.block|int64_t {
    return at::q_zero_point(*$fptr-ptr:(Tensor* self));
   }|]


-- Tensor q_per_channel_scales(const Tensor & self)
--
q_per_channel_scales__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
q_per_channel_scales__t self =  
  [C.block|Tensor* {
    return new Tensor(at::q_per_channel_scales(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor q_per_channel_zero_points(const Tensor & self)
--
q_per_channel_zero_points__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
q_per_channel_zero_points__t self =  
  [C.block|Tensor* {
    return new Tensor(at::q_per_channel_zero_points(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- int64_t q_per_channel_axis(const Tensor & self)
--
q_per_channel_axis__t :: ForeignPtr CTensor -> IO (Int64)
q_per_channel_axis__t self =  
  [C.block|int64_t {
    return at::q_per_channel_axis(*$fptr-ptr:(Tensor* self));
   }|]


-- Tensor int_repr(const Tensor & self)
--
int_repr__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
int_repr__t self =  
  [C.block|Tensor* {
    return new Tensor(at::int_repr(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _make_per_tensor_quantized_tensor(const Tensor & self, double scale, int64_t zero_point)
--
_make_per_tensor_quantized_tensor__td6 :: ForeignPtr CTensor -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
_make_per_tensor_quantized_tensor__td6 self scale zero_point =  
  [C.block|Tensor* {
    return new Tensor(at::_make_per_tensor_quantized_tensor(*$fptr-ptr:(Tensor* self), $(double scale), $(int64_t zero_point)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _make_per_channel_quantized_tensor(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis)
--
_make_per_channel_quantized_tensor__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_make_per_channel_quantized_tensor__ttt6 self scale zero_point axis =  
  [C.block|Tensor* {
    return new Tensor(at::_make_per_channel_quantized_tensor(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* scale), *$fptr-ptr:(Tensor* zero_point), $(int64_t axis)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor fake_quantize_per_tensor_affine(const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max)
--
fake_quantize_per_tensor_affine__td666 :: ForeignPtr CTensor -> CDouble -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
fake_quantize_per_tensor_affine__td666 self scale zero_point quant_min quant_max =  
  [C.block|Tensor* {
    return new Tensor(at::fake_quantize_per_tensor_affine(*$fptr-ptr:(Tensor* self), $(double scale), $(int64_t zero_point), $(int64_t quant_min), $(int64_t quant_max)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor fake_quantize_per_channel_affine(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max)
--
fake_quantize_per_channel_affine__ttt666 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
fake_quantize_per_channel_affine__ttt666 self scale zero_point axis quant_min quant_max =  
  [C.block|Tensor* {
    return new Tensor(at::fake_quantize_per_channel_affine(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* scale), *$fptr-ptr:(Tensor* zero_point), $(int64_t axis), $(int64_t quant_min), $(int64_t quant_max)));
   }|] >>= newForeignPtr deleteTensor


-- std::vector<Tensor> meshgrid(TensorList tensors)
--
meshgrid__l :: Vector (Ptr CTensor) -> IO (Vector (Ptr CTensor))
meshgrid__l tensors =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in 
  [C.block|void* {
    return (void*)new std::vector<Tensor>(at::meshgrid(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size))));
   }|] >>= unVectorTensor


-- Tensor cartesian_prod(TensorList tensors)
--
cartesian_prod__l :: Vector (Ptr CTensor) -> IO (ForeignPtr CTensor)
cartesian_prod__l tensors =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in 
  [C.block|Tensor* {
    return new Tensor(at::cartesian_prod(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor combinations(const Tensor & self, int64_t r, bool with_replacement)
--
combinations__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
combinations__t6b self r with_replacement =  
  [C.block|Tensor* {
    return new Tensor(at::combinations(*$fptr-ptr:(Tensor* self), $(int64_t r), $(bool with_replacement)));
   }|] >>= newForeignPtr deleteTensor


-- ScalarType result_type(const Tensor & tensor, const Tensor & other)
--
result_type__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (Int8)
result_type__tt tensor other =  
  [C.block|int8_t {
    return static_cast<int8_t>(at::result_type(*$fptr-ptr:(Tensor* tensor), *$fptr-ptr:(Tensor* other)));
   }|]


-- ScalarType result_type(const Tensor & tensor, Scalar other)
--
result_type__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (Int8)
result_type__ts tensor other =  
  [C.block|int8_t {
    return static_cast<int8_t>(at::result_type(*$fptr-ptr:(Tensor* tensor), *$fptr-ptr:(Scalar* other)));
   }|]


-- ScalarType result_type(Scalar scalar, const Tensor & tensor)
--
result_type__st :: ForeignPtr CScalar -> ForeignPtr CTensor -> IO (Int8)
result_type__st scalar tensor =  
  [C.block|int8_t {
    return static_cast<int8_t>(at::result_type(*$fptr-ptr:(Scalar* scalar), *$fptr-ptr:(Tensor* tensor)));
   }|]


-- ScalarType result_type(Scalar scalar1, Scalar scalar2)
--
result_type__ss :: ForeignPtr CScalar -> ForeignPtr CScalar -> IO (Int8)
result_type__ss scalar1 scalar2 =  
  [C.block|int8_t {
    return static_cast<int8_t>(at::result_type(*$fptr-ptr:(Scalar* scalar1), *$fptr-ptr:(Scalar* scalar2)));
   }|]


-- bool can_cast(ScalarType from, ScalarType to)
--
can_cast__ss :: Int8 -> Int8 -> IO (CBool)
can_cast__ss from to =  
  [C.block|bool {
    return at::can_cast(static_cast<ScalarType>($(int8_t from)), static_cast<ScalarType>($(int8_t to)));
   }|]


-- ScalarType promote_types(ScalarType type1, ScalarType type2)
--
promote_types__ss :: Int8 -> Int8 -> IO (Int8)
promote_types__ss type1 type2 =  
  [C.block|int8_t {
    return static_cast<int8_t>(at::promote_types(static_cast<ScalarType>($(int8_t type1)), static_cast<ScalarType>($(int8_t type2))));
   }|]


-- Scalar _local_scalar_dense(const Tensor & self)
--
_local_scalar_dense__t :: ForeignPtr CTensor -> IO (ForeignPtr CScalar)
_local_scalar_dense__t self =  
  [C.block|Scalar* {
    return new Scalar(at::_local_scalar_dense(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteScalar'


-- std::tuple<Tensor,Tensor,Tensor> _thnn_fused_lstm_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & cx, const Tensor & input_bias, const Tensor & hidden_bias)
--
_thnn_fused_lstm_cell__ttttt__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_thnn_fused_lstm_cell__ttttt__1 input_gates hidden_gates cx input_bias hidden_bias =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::_thnn_fused_lstm_cell(*$fptr-ptr:(Tensor* input_gates), *$fptr-ptr:(Tensor* hidden_gates), *$fptr-ptr:(Tensor* cx), *$fptr-ptr:(Tensor* input_bias), *$fptr-ptr:(Tensor* hidden_bias)));
   }|] >>= unTupleTensorTensorTensor


-- std::tuple<Tensor,Tensor> _thnn_fused_gru_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const Tensor & input_bias, const Tensor & hidden_bias)
--
_thnn_fused_gru_cell__ttttt__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_thnn_fused_gru_cell__ttttt__1 input_gates hidden_gates hx input_bias hidden_bias =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_thnn_fused_gru_cell(*$fptr-ptr:(Tensor* input_gates), *$fptr-ptr:(Tensor* hidden_gates), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* input_bias), *$fptr-ptr:(Tensor* hidden_bias)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor,Tensor> lstm(const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)
--
lstm__tllb6dbbb :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
lstm__tllb6dbbb input hx params has_biases num_layers dropout train bidirectional batch_first =  V.unsafeWith hx $ \hx__array -> let hx__size = fromIntegral (V.length hx) in V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::lstm(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** hx__array), $(size_t hx__size)), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), $(bool batch_first)));
   }|] >>= unTupleTensorTensorTensor


-- std::tuple<Tensor,Tensor,Tensor> lstm(const Tensor & data, const Tensor & batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)
--
lstm__ttllb6dbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
lstm__ttllb6dbb dataX batch_sizes hx params has_biases num_layers dropout train bidirectional =  V.unsafeWith hx $ \hx__array -> let hx__size = fromIntegral (V.length hx) in V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::lstm(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), pack_tensor_list($(Tensor** hx__array), $(size_t hx__size)), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional)));
   }|] >>= unTupleTensorTensorTensor


-- std::tuple<Tensor,Tensor> gru(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)
--
gru__ttlb6dbbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
gru__ttlb6dbbb input hx params has_biases num_layers dropout train bidirectional batch_first =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::gru(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), $(bool batch_first)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> gru(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)
--
gru__tttlb6dbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
gru__tttlb6dbb dataX batch_sizes hx params has_biases num_layers dropout train bidirectional =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::gru(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> rnn_tanh(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)
--
rnn_tanh__ttlb6dbbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
rnn_tanh__ttlb6dbbb input hx params has_biases num_layers dropout train bidirectional batch_first =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::rnn_tanh(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), $(bool batch_first)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> rnn_tanh(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)
--
rnn_tanh__tttlb6dbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
rnn_tanh__tttlb6dbb dataX batch_sizes hx params has_biases num_layers dropout train bidirectional =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::rnn_tanh(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> rnn_relu(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)
--
rnn_relu__ttlb6dbbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
rnn_relu__ttlb6dbbb input hx params has_biases num_layers dropout train bidirectional batch_first =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::rnn_relu(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), $(bool batch_first)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> rnn_relu(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)
--
rnn_relu__tttlb6dbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
rnn_relu__tttlb6dbb dataX batch_sizes hx params has_biases num_layers dropout train bidirectional =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::rnn_relu(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh)
--
lstm_cell__tltttt__1 :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
lstm_cell__tltttt__1 input hx w_ih w_hh b_ih b_hh =  V.unsafeWith hx $ \hx__array -> let hx__size = fromIntegral (V.length hx) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::lstm_cell(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** hx__array), $(size_t hx__size)), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh)));
   }|] >>= unTupleTensorTensor


-- Tensor gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh)
--
gru_cell__tttttt__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
gru_cell__tttttt__1 input hx w_ih w_hh b_ih b_hh =  
  [C.block|Tensor* {
    return new Tensor(at::gru_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh)
--
rnn_tanh_cell__tttttt__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rnn_tanh_cell__tttttt__1 input hx w_ih w_hh b_ih b_hh =  
  [C.block|Tensor* {
    return new Tensor(at::rnn_tanh_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh)
--
rnn_relu_cell__tttttt__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rnn_relu_cell__tttttt__1 input hx w_ih w_hh b_ih b_hh =  
  [C.block|Tensor* {
    return new Tensor(at::rnn_relu_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor,Tensor> quantized_lstm(const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, c10::optional<ScalarType> dtype, bool use_dynamic)
--
quantized_lstm__tllb6dbbbsb :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> Int8 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
quantized_lstm__tllb6dbbbsb input hx params has_biases num_layers dropout train bidirectional batch_first dtype use_dynamic =  V.unsafeWith hx $ \hx__array -> let hx__size = fromIntegral (V.length hx) in V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::quantized_lstm(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** hx__array), $(size_t hx__size)), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), $(bool batch_first), static_cast<ScalarType>($(int8_t dtype)), $(bool use_dynamic)));
   }|] >>= unTupleTensorTensorTensor


-- std::tuple<Tensor,Tensor,Tensor> quantized_lstm(const Tensor & data, const Tensor & batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, c10::optional<ScalarType> dtype, bool use_dynamic)
--
quantized_lstm__ttllb6dbbsb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> Int8 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
quantized_lstm__ttllb6dbbsb dataX batch_sizes hx params has_biases num_layers dropout train bidirectional dtype use_dynamic =  V.unsafeWith hx $ \hx__array -> let hx__size = fromIntegral (V.length hx) in V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::quantized_lstm(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), pack_tensor_list($(Tensor** hx__array), $(size_t hx__size)), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), static_cast<ScalarType>($(int8_t dtype)), $(bool use_dynamic)));
   }|] >>= unTupleTensorTensorTensor


-- std::tuple<Tensor,Tensor> quantized_gru(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)
--
quantized_gru__ttlb6dbbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
quantized_gru__ttlb6dbbb input hx params has_biases num_layers dropout train bidirectional batch_first =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::quantized_gru(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), $(bool batch_first)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> quantized_gru(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)
--
quantized_gru__tttlb6dbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
quantized_gru__tttlb6dbb dataX batch_sizes hx params has_biases num_layers dropout train bidirectional =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::quantized_gru(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> quantized_lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh)
--
quantized_lstm_cell__tlttttttttssss :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
quantized_lstm_cell__tlttttttttssss input hx w_ih w_hh b_ih b_hh packed_ih packed_hh col_offsets_ih col_offsets_hh scale_ih scale_hh zero_point_ih zero_point_hh =  V.unsafeWith hx $ \hx__array -> let hx__size = fromIntegral (V.length hx) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::quantized_lstm_cell(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** hx__array), $(size_t hx__size)), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh), *$fptr-ptr:(Tensor* packed_ih), *$fptr-ptr:(Tensor* packed_hh), *$fptr-ptr:(Tensor* col_offsets_ih), *$fptr-ptr:(Tensor* col_offsets_hh), *$fptr-ptr:(Scalar* scale_ih), *$fptr-ptr:(Scalar* scale_hh), *$fptr-ptr:(Scalar* zero_point_ih), *$fptr-ptr:(Scalar* zero_point_hh)));
   }|] >>= unTupleTensorTensor


-- Tensor quantized_gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh)
--
quantized_gru_cell__ttttttttttssss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
quantized_gru_cell__ttttttttttssss input hx w_ih w_hh b_ih b_hh packed_ih packed_hh col_offsets_ih col_offsets_hh scale_ih scale_hh zero_point_ih zero_point_hh =  
  [C.block|Tensor* {
    return new Tensor(at::quantized_gru_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh), *$fptr-ptr:(Tensor* packed_ih), *$fptr-ptr:(Tensor* packed_hh), *$fptr-ptr:(Tensor* col_offsets_ih), *$fptr-ptr:(Tensor* col_offsets_hh), *$fptr-ptr:(Scalar* scale_ih), *$fptr-ptr:(Scalar* scale_hh), *$fptr-ptr:(Scalar* zero_point_ih), *$fptr-ptr:(Scalar* zero_point_hh)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor quantized_rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh)
--
quantized_rnn_relu_cell__ttttttttttssss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
quantized_rnn_relu_cell__ttttttttttssss input hx w_ih w_hh b_ih b_hh packed_ih packed_hh col_offsets_ih col_offsets_hh scale_ih scale_hh zero_point_ih zero_point_hh =  
  [C.block|Tensor* {
    return new Tensor(at::quantized_rnn_relu_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh), *$fptr-ptr:(Tensor* packed_ih), *$fptr-ptr:(Tensor* packed_hh), *$fptr-ptr:(Tensor* col_offsets_ih), *$fptr-ptr:(Tensor* col_offsets_hh), *$fptr-ptr:(Scalar* scale_ih), *$fptr-ptr:(Scalar* scale_hh), *$fptr-ptr:(Scalar* zero_point_ih), *$fptr-ptr:(Scalar* zero_point_hh)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor quantized_rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh)
--
quantized_rnn_tanh_cell__ttttttttttssss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
quantized_rnn_tanh_cell__ttttttttttssss input hx w_ih w_hh b_ih b_hh packed_ih packed_hh col_offsets_ih col_offsets_hh scale_ih scale_hh zero_point_ih zero_point_hh =  
  [C.block|Tensor* {
    return new Tensor(at::quantized_rnn_tanh_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh), *$fptr-ptr:(Tensor* packed_ih), *$fptr-ptr:(Tensor* packed_hh), *$fptr-ptr:(Tensor* col_offsets_ih), *$fptr-ptr:(Tensor* col_offsets_hh), *$fptr-ptr:(Scalar* scale_ih), *$fptr-ptr:(Scalar* scale_hh), *$fptr-ptr:(Scalar* zero_point_ih), *$fptr-ptr:(Scalar* zero_point_hh)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> _pack_padded_sequence(const Tensor & input, const Tensor & lengths, bool batch_first)
--
_pack_padded_sequence__ttb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_pack_padded_sequence__ttb input lengths batch_first =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_pack_padded_sequence(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* lengths), $(bool batch_first)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> _pad_packed_sequence(const Tensor & data, const Tensor & batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length)
--
_pad_packed_sequence__ttbs6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> ForeignPtr CScalar -> Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_pad_packed_sequence__ttbs6 dataX batch_sizes batch_first padding_value total_length =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_pad_packed_sequence(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), $(bool batch_first), *$fptr-ptr:(Scalar* padding_value), $(int64_t total_length)));
   }|] >>= unTupleTensorTensor


-- Tensor masked_fill(const Tensor & self, const Tensor & mask, Scalar value)
--
masked_fill__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
masked_fill__tts self mask value =  
  [C.block|Tensor* {
    return new Tensor(at::masked_fill(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor masked_fill(const Tensor & self, const Tensor & mask, const Tensor & value)
--
masked_fill__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_fill__ttt self mask value =  
  [C.block|Tensor* {
    return new Tensor(at::masked_fill(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Tensor* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor masked_scatter(const Tensor & self, const Tensor & mask, const Tensor & source)
--
masked_scatter__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_scatter__ttt self mask source =  
  [C.block|Tensor* {
    return new Tensor(at::masked_scatter(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Tensor* source)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor index_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source)
--
index_add__t6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_add__t6tt self dim index source =  
  [C.block|Tensor* {
    return new Tensor(at::index_add(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, Scalar value)
--
index_fill__t6ts :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
index_fill__t6ts self dim index value =  
  [C.block|Tensor* {
    return new Tensor(at::index_fill(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & value)
--
index_fill__t6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_fill__t6tt self dim index value =  
  [C.block|Tensor* {
    return new Tensor(at::index_fill(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src)
--
scatter__t6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
scatter__t6tt self dim index src =  
  [C.block|Tensor* {
    return new Tensor(at::scatter(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* src)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar value)
--
scatter__t6ts :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
scatter__t6ts self dim index value =  
  [C.block|Tensor* {
    return new Tensor(at::scatter(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src)
--
scatter_add__t6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
scatter_add__t6tt self dim index src =  
  [C.block|Tensor* {
    return new Tensor(at::scatter_add(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* src)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & bitwise_and_out(Tensor & out, const Tensor & self, const Tensor & other)
--
bitwise_and_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_and_out__ttt out self other =  
  [C.block|void {
    at::bitwise_and_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor & bitwise_and_out(Tensor & out, const Tensor & self, Scalar other)
--
bitwise_and_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
bitwise_and_out__tts out self other =  
  [C.block|void {
    at::bitwise_and_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


-- Tensor bitwise_and(const Tensor & self, Scalar other)
--
bitwise_and__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
bitwise_and__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::bitwise_and(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bitwise_and(const Tensor & self, const Tensor & other)
--
bitwise_and__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_and__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::bitwise_and(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor __and__(const Tensor & self, Scalar other)
--
__and____ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__and____ts self other =  
  [C.block|void {
    at::__and__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor __and__(const Tensor & self, const Tensor & other)
--
__and____tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__and____tt self other =  
  [C.block|void {
    at::__and__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & bitwise_or_out(Tensor & out, const Tensor & self, const Tensor & other)
--
bitwise_or_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_or_out__ttt out self other =  
  [C.block|void {
    at::bitwise_or_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor & bitwise_or_out(Tensor & out, const Tensor & self, Scalar other)
--
bitwise_or_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
bitwise_or_out__tts out self other =  
  [C.block|void {
    at::bitwise_or_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


-- Tensor bitwise_or(const Tensor & self, Scalar other)
--
bitwise_or__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
bitwise_or__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::bitwise_or(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bitwise_or(const Tensor & self, const Tensor & other)
--
bitwise_or__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_or__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::bitwise_or(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor __or__(const Tensor & self, Scalar other)
--
__or____ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__or____ts self other =  
  [C.block|void {
    at::__or__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor __or__(const Tensor & self, const Tensor & other)
--
__or____tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__or____tt self other =  
  [C.block|void {
    at::__or__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & bitwise_xor_out(Tensor & out, const Tensor & self, const Tensor & other)
--
bitwise_xor_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_xor_out__ttt out self other =  
  [C.block|void {
    at::bitwise_xor_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor & bitwise_xor_out(Tensor & out, const Tensor & self, Scalar other)
--
bitwise_xor_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
bitwise_xor_out__tts out self other =  
  [C.block|void {
    at::bitwise_xor_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


-- Tensor bitwise_xor(const Tensor & self, Scalar other)
--
bitwise_xor__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
bitwise_xor__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::bitwise_xor(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bitwise_xor(const Tensor & self, const Tensor & other)
--
bitwise_xor__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_xor__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::bitwise_xor(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor __xor__(const Tensor & self, Scalar other)
--
__xor____ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__xor____ts self other =  
  [C.block|void {
    at::__xor__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor __xor__(const Tensor & self, const Tensor & other)
--
__xor____tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__xor____tt self other =  
  [C.block|void {
    at::__xor__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor __lshift__(const Tensor & self, Scalar other)
--
__lshift____ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__lshift____ts self other =  
  [C.block|void {
    at::__lshift__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor __lshift__(const Tensor & self, const Tensor & other)
--
__lshift____tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__lshift____tt self other =  
  [C.block|void {
    at::__lshift__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor __rshift__(const Tensor & self, Scalar other)
--
__rshift____ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__rshift____ts self other =  
  [C.block|void {
    at::__rshift__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor __rshift__(const Tensor & self, const Tensor & other)
--
__rshift____tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__rshift____tt self other =  
  [C.block|void {
    at::__rshift__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & addbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha)
--
addbmm_out__ttttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addbmm_out__ttttss out self batch1 batch2 beta alpha =  
  [C.block|void {
    at::addbmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


-- Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha)
--
addbmm__tttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addbmm__tttss self batch1 batch2 beta alpha =  
  [C.block|Tensor* {
    return new Tensor(at::addbmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & diag_out(Tensor & out, const Tensor & self, int64_t diagonal)
--
diag_out__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
diag_out__tt6 out self diagonal =  
  [C.block|void {
    at::diag_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t diagonal));
   }|] >> pure (out)


-- Tensor diag(const Tensor & self, int64_t diagonal)
--
diag__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
diag__t6 self diagonal =  
  [C.block|Tensor* {
    return new Tensor(at::diag(*$fptr-ptr:(Tensor* self), $(int64_t diagonal)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & cross_out(Tensor & out, const Tensor & self, const Tensor & other, c10::optional<int64_t> dim)
--
cross_out__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Maybe Int64 -> IO (ForeignPtr CTensor)
cross_out__ttt6 out self other dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in 
  [C.block|void {
    at::cross_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt));
   }|] >> pure (out)


-- Tensor cross(const Tensor & self, const Tensor & other, c10::optional<int64_t> dim)
--
cross__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Maybe Int64 -> IO (ForeignPtr CTensor)
cross__tt6 self other dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in 
  [C.block|Tensor* {
    return new Tensor(at::cross(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & triu_out(Tensor & out, const Tensor & self, int64_t diagonal)
--
triu_out__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
triu_out__tt6 out self diagonal =  
  [C.block|void {
    at::triu_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t diagonal));
   }|] >> pure (out)


-- Tensor triu(const Tensor & self, int64_t diagonal)
--
triu__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
triu__t6 self diagonal =  
  [C.block|Tensor* {
    return new Tensor(at::triu(*$fptr-ptr:(Tensor* self), $(int64_t diagonal)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & tril_out(Tensor & out, const Tensor & self, int64_t diagonal)
--
tril_out__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
tril_out__tt6 out self diagonal =  
  [C.block|void {
    at::tril_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t diagonal));
   }|] >> pure (out)


-- Tensor tril(const Tensor & self, int64_t diagonal)
--
tril__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
tril__t6 self diagonal =  
  [C.block|Tensor* {
    return new Tensor(at::tril(*$fptr-ptr:(Tensor* self), $(int64_t diagonal)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor tril_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options)
--
tril_indices__666o :: Int64 -> Int64 -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
tril_indices__666o row col offset options =  
  [C.block|Tensor* {
    return new Tensor(at::tril_indices($(int64_t row), $(int64_t col), $(int64_t offset), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor triu_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options)
--
triu_indices__666o :: Int64 -> Int64 -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
triu_indices__666o row col offset options =  
  [C.block|Tensor* {
    return new Tensor(at::triu_indices($(int64_t row), $(int64_t col), $(int64_t offset), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor trace(const Tensor & self)
--
trace__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
trace__t self =  
  [C.block|Tensor* {
    return new Tensor(at::trace(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & ne_out(Tensor & out, const Tensor & self, Scalar other)
--
ne_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ne_out__tts out self other =  
  [C.block|void {
    at::ne_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


-- Tensor ne(const Tensor & self, Scalar other)
--
ne__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ne__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::ne(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & ne_out(Tensor & out, const Tensor & self, const Tensor & other)
--
ne_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ne_out__ttt out self other =  
  [C.block|void {
    at::ne_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor ne(const Tensor & self, const Tensor & other)
--
ne__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ne__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::ne(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & eq_out(Tensor & out, const Tensor & self, Scalar other)
--
eq_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
eq_out__tts out self other =  
  [C.block|void {
    at::eq_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


-- Tensor eq(const Tensor & self, Scalar other)
--
eq__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
eq__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::eq(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & eq_out(Tensor & out, const Tensor & self, const Tensor & other)
--
eq_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
eq_out__ttt out self other =  
  [C.block|void {
    at::eq_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor eq(const Tensor & self, const Tensor & other)
--
eq__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
eq__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::eq(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & ge_out(Tensor & out, const Tensor & self, Scalar other)
--
ge_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ge_out__tts out self other =  
  [C.block|void {
    at::ge_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


-- Tensor ge(const Tensor & self, Scalar other)
--
ge__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ge__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::ge(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & ge_out(Tensor & out, const Tensor & self, const Tensor & other)
--
ge_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ge_out__ttt out self other =  
  [C.block|void {
    at::ge_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor ge(const Tensor & self, const Tensor & other)
--
ge__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ge__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::ge(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & le_out(Tensor & out, const Tensor & self, Scalar other)
--
le_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
le_out__tts out self other =  
  [C.block|void {
    at::le_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


-- Tensor le(const Tensor & self, Scalar other)
--
le__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
le__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::le(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & le_out(Tensor & out, const Tensor & self, const Tensor & other)
--
le_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
le_out__ttt out self other =  
  [C.block|void {
    at::le_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor le(const Tensor & self, const Tensor & other)
--
le__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
le__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::le(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & gt_out(Tensor & out, const Tensor & self, Scalar other)
--
gt_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
gt_out__tts out self other =  
  [C.block|void {
    at::gt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


-- Tensor gt(const Tensor & self, Scalar other)
--
gt__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
gt__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::gt(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & gt_out(Tensor & out, const Tensor & self, const Tensor & other)
--
gt_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
gt_out__ttt out self other =  
  [C.block|void {
    at::gt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor gt(const Tensor & self, const Tensor & other)
--
gt__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
gt__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::gt(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & lt_out(Tensor & out, const Tensor & self, Scalar other)
--
lt_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lt_out__tts out self other =  
  [C.block|void {
    at::lt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


-- Tensor lt(const Tensor & self, Scalar other)
--
lt__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lt__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::lt(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & lt_out(Tensor & out, const Tensor & self, const Tensor & other)
--
lt_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lt_out__ttt out self other =  
  [C.block|void {
    at::lt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor lt(const Tensor & self, const Tensor & other)
--
lt__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lt__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::lt(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & take_out(Tensor & out, const Tensor & self, const Tensor & index)
--
take_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
take_out__ttt out self index =  
  [C.block|void {
    at::take_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* index));
   }|] >> pure (out)


-- Tensor take(const Tensor & self, const Tensor & index)
--
take__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
take__tt self index =  
  [C.block|Tensor* {
    return new Tensor(at::take(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* index)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & index_select_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index)
--
index_select_out__tt6t :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_select_out__tt6t out self dim index =  
  [C.block|void {
    at::index_select_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index));
   }|] >> pure (out)


-- Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index)
--
index_select__t6t :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_select__t6t self dim index =  
  [C.block|Tensor* {
    return new Tensor(at::index_select(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & masked_select_out(Tensor & out, const Tensor & self, const Tensor & mask)
--
masked_select_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_select_out__ttt out self mask =  
  [C.block|void {
    at::masked_select_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask));
   }|] >> pure (out)


-- Tensor masked_select(const Tensor & self, const Tensor & mask)
--
masked_select__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_select__tt self mask =  
  [C.block|Tensor* {
    return new Tensor(at::masked_select(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & nonzero_out(Tensor & out, const Tensor & self)
--
nonzero_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
nonzero_out__tt out self =  
  [C.block|void {
    at::nonzero_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor nonzero(const Tensor & self)
--
nonzero__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
nonzero__t self =  
  [C.block|Tensor* {
    return new Tensor(at::nonzero(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- std::vector<Tensor> nonzero_numpy(const Tensor & self)
--
nonzero_numpy__t :: ForeignPtr CTensor -> IO (Vector (Ptr CTensor))
nonzero_numpy__t self =  
  [C.block|void* {
    return (void*)new std::vector<Tensor>(at::nonzero_numpy(*$fptr-ptr:(Tensor* self)));
   }|] >>= unVectorTensor


-- Tensor & gather_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad)
--
gather_out__tt6tb :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
gather_out__tt6tb out self dim index sparse_grad =  
  [C.block|void {
    at::gather_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), $(bool sparse_grad));
   }|] >> pure (out)


-- Tensor gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad)
--
gather__t6tb :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
gather__t6tb self dim index sparse_grad =  
  [C.block|Tensor* {
    return new Tensor(at::gather(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), $(bool sparse_grad)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & addcmul_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value)
--
addcmul_out__tttts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcmul_out__tttts out self tensor1 tensor2 value =  
  [C.block|void {
    at::addcmul_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value));
   }|] >> pure (out)


-- Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value)
--
addcmul__ttts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcmul__ttts self tensor1 tensor2 value =  
  [C.block|Tensor* {
    return new Tensor(at::addcmul(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & addcdiv_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value)
--
addcdiv_out__tttts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcdiv_out__tttts out self tensor1 tensor2 value =  
  [C.block|void {
    at::addcdiv_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value));
   }|] >> pure (out)


-- Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value)
--
addcdiv__ttts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcdiv__ttts self tensor1 tensor2 value =  
  [C.block|Tensor* {
    return new Tensor(at::addcdiv(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor &,Tensor &> lstsq_out(Tensor & X, Tensor & qr, const Tensor & self, const Tensor & A)
--
lstsq_out__tttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
lstsq_out__tttt x qr self a =  
  [C.block|void {
    at::lstsq_out(*$fptr-ptr:(Tensor* x), *$fptr-ptr:(Tensor* qr), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a));
   }|] >> pure (x,qr)


-- std::tuple<Tensor,Tensor> lstsq(const Tensor & self, const Tensor & A)
--
lstsq__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
lstsq__tt self a =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::lstsq(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> triangular_solve_out(Tensor & X, Tensor & M, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular)
--
triangular_solve_out__ttttbbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
triangular_solve_out__ttttbbb x m self a upper transpose unitriangular =  
  [C.block|void {
    at::triangular_solve_out(*$fptr-ptr:(Tensor* x), *$fptr-ptr:(Tensor* m), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a), $(bool upper), $(bool transpose), $(bool unitriangular));
   }|] >> pure (x,m)


-- std::tuple<Tensor,Tensor> triangular_solve(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular)
--
triangular_solve__ttbbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
triangular_solve__ttbbb self a upper transpose unitriangular =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::triangular_solve(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a), $(bool upper), $(bool transpose), $(bool unitriangular)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> _triangular_solve_helper(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular)
--
_triangular_solve_helper__ttbbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_triangular_solve_helper__ttbbb self a upper transpose unitriangular =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_triangular_solve_helper(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a), $(bool upper), $(bool transpose), $(bool unitriangular)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> symeig_out(Tensor & e, Tensor & V, const Tensor & self, bool eigenvectors, bool upper)
--
symeig_out__tttbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
symeig_out__tttbb e v self eigenvectors upper =  
  [C.block|void {
    at::symeig_out(*$fptr-ptr:(Tensor* e), *$fptr-ptr:(Tensor* v), *$fptr-ptr:(Tensor* self), $(bool eigenvectors), $(bool upper));
   }|] >> pure (e,v)


-- std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper)
--
symeig__tbb :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
symeig__tbb self eigenvectors upper =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::symeig(*$fptr-ptr:(Tensor* self), $(bool eigenvectors), $(bool upper)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> _symeig_helper(const Tensor & self, bool eigenvectors, bool upper)
--
_symeig_helper__tbb :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_symeig_helper__tbb self eigenvectors upper =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_symeig_helper(*$fptr-ptr:(Tensor* self), $(bool eigenvectors), $(bool upper)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> eig_out(Tensor & e, Tensor & v, const Tensor & self, bool eigenvectors)
--
eig_out__tttb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
eig_out__tttb e v self eigenvectors =  
  [C.block|void {
    at::eig_out(*$fptr-ptr:(Tensor* e), *$fptr-ptr:(Tensor* v), *$fptr-ptr:(Tensor* self), $(bool eigenvectors));
   }|] >> pure (e,v)


-- std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors)
--
eig__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
eig__tb self eigenvectors =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::eig(*$fptr-ptr:(Tensor* self), $(bool eigenvectors)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &,Tensor &> svd_out(Tensor & U, Tensor & S, Tensor & V, const Tensor & self, bool some, bool compute_uv)
--
svd_out__ttttbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
svd_out__ttttbb u s v self some compute_uv =  
  [C.block|void {
    at::svd_out(*$fptr-ptr:(Tensor* u), *$fptr-ptr:(Tensor* s), *$fptr-ptr:(Tensor* v), *$fptr-ptr:(Tensor* self), $(bool some), $(bool compute_uv));
   }|] >> pure (u,s,v)


-- std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some, bool compute_uv)
--
svd__tbb :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
svd__tbb self some compute_uv =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::svd(*$fptr-ptr:(Tensor* self), $(bool some), $(bool compute_uv)));
   }|] >>= unTupleTensorTensorTensor


-- std::tuple<Tensor,Tensor,Tensor> _svd_helper(const Tensor & self, bool some, bool compute_uv)
--
_svd_helper__tbb :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_svd_helper__tbb self some compute_uv =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::_svd_helper(*$fptr-ptr:(Tensor* self), $(bool some), $(bool compute_uv)));
   }|] >>= unTupleTensorTensorTensor


-- Tensor & cholesky_out(Tensor & out, const Tensor & self, bool upper)
--
cholesky_out__ttb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_out__ttb out self upper =  
  [C.block|void {
    at::cholesky_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(bool upper));
   }|] >> pure (out)


-- Tensor cholesky(const Tensor & self, bool upper)
--
cholesky__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky__tb self upper =  
  [C.block|Tensor* {
    return new Tensor(at::cholesky(*$fptr-ptr:(Tensor* self), $(bool upper)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _cholesky_helper(const Tensor & self, bool upper)
--
_cholesky_helper__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cholesky_helper__tb self upper =  
  [C.block|Tensor* {
    return new Tensor(at::_cholesky_helper(*$fptr-ptr:(Tensor* self), $(bool upper)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & cholesky_solve_out(Tensor & out, const Tensor & self, const Tensor & input2, bool upper)
--
cholesky_solve_out__tttb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_solve_out__tttb out self input2 upper =  
  [C.block|void {
    at::cholesky_solve_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* input2), $(bool upper));
   }|] >> pure (out)


-- Tensor cholesky_solve(const Tensor & self, const Tensor & input2, bool upper)
--
cholesky_solve__ttb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_solve__ttb self input2 upper =  
  [C.block|Tensor* {
    return new Tensor(at::cholesky_solve(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* input2), $(bool upper)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _cholesky_solve_helper(const Tensor & self, const Tensor & A, bool upper)
--
_cholesky_solve_helper__ttb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cholesky_solve_helper__ttb self a upper =  
  [C.block|Tensor* {
    return new Tensor(at::_cholesky_solve_helper(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a), $(bool upper)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> solve(const Tensor & self, const Tensor & A)
--
solve__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
solve__tt self a =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::solve(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> solve_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A)
--
solve_out__tttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
solve_out__tttt solution lu self a =  
  [C.block|void {
    at::solve_out(*$fptr-ptr:(Tensor* solution), *$fptr-ptr:(Tensor* lu), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a));
   }|] >> pure (solution,lu)


-- std::tuple<Tensor,Tensor> _solve_helper(const Tensor & self, const Tensor & A)
--
_solve_helper__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_solve_helper__tt self a =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_solve_helper(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a)));
   }|] >>= unTupleTensorTensor


-- Tensor & cholesky_inverse_out(Tensor & out, const Tensor & self, bool upper)
--
cholesky_inverse_out__ttb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_inverse_out__ttb out self upper =  
  [C.block|void {
    at::cholesky_inverse_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(bool upper));
   }|] >> pure (out)


-- Tensor cholesky_inverse(const Tensor & self, bool upper)
--
cholesky_inverse__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_inverse__tb self upper =  
  [C.block|Tensor* {
    return new Tensor(at::cholesky_inverse(*$fptr-ptr:(Tensor* self), $(bool upper)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor &,Tensor &> qr_out(Tensor & Q, Tensor & R, const Tensor & self, bool some)
--
qr_out__tttb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
qr_out__tttb q r self some =  
  [C.block|void {
    at::qr_out(*$fptr-ptr:(Tensor* q), *$fptr-ptr:(Tensor* r), *$fptr-ptr:(Tensor* self), $(bool some));
   }|] >> pure (q,r)


-- std::tuple<Tensor,Tensor> qr(const Tensor & self, bool some)
--
qr__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
qr__tb self some =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::qr(*$fptr-ptr:(Tensor* self), $(bool some)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> _qr_helper(const Tensor & self, bool some)
--
_qr_helper__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_qr_helper__tb self some =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_qr_helper(*$fptr-ptr:(Tensor* self), $(bool some)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> geqrf_out(Tensor & a, Tensor & tau, const Tensor & self)
--
geqrf_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
geqrf_out__ttt a tau self =  
  [C.block|void {
    at::geqrf_out(*$fptr-ptr:(Tensor* a), *$fptr-ptr:(Tensor* tau), *$fptr-ptr:(Tensor* self));
   }|] >> pure (a,tau)


-- std::tuple<Tensor,Tensor> geqrf(const Tensor & self)
--
geqrf__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
geqrf__t self =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::geqrf(*$fptr-ptr:(Tensor* self)));
   }|] >>= unTupleTensorTensor


-- Tensor & orgqr_out(Tensor & out, const Tensor & self, const Tensor & input2)
--
orgqr_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
orgqr_out__ttt out self input2 =  
  [C.block|void {
    at::orgqr_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* input2));
   }|] >> pure (out)


-- Tensor orgqr(const Tensor & self, const Tensor & input2)
--
orgqr__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
orgqr__tt self input2 =  
  [C.block|Tensor* {
    return new Tensor(at::orgqr(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* input2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & ormqr_out(Tensor & out, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose)
--
ormqr_out__ttttbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor)
ormqr_out__ttttbb out self input2 input3 left transpose =  
  [C.block|void {
    at::ormqr_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* input2), *$fptr-ptr:(Tensor* input3), $(bool left), $(bool transpose));
   }|] >> pure (out)


-- Tensor ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose)
--
ormqr__tttbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor)
ormqr__tttbb self input2 input3 left transpose =  
  [C.block|Tensor* {
    return new Tensor(at::ormqr(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* input2), *$fptr-ptr:(Tensor* input3), $(bool left), $(bool transpose)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor,Tensor> _lu_with_info(const Tensor & self, bool pivot, bool check_errors)
--
_lu_with_info__tbb :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_lu_with_info__tbb self pivot check_errors =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(at::_lu_with_info(*$fptr-ptr:(Tensor* self), $(bool pivot), $(bool check_errors)));
   }|] >>= unTupleTensorTensorTensor


-- Tensor & lu_solve_out(Tensor & out, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots)
--
lu_solve_out__tttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lu_solve_out__tttt out self lu_data lu_pivots =  
  [C.block|void {
    at::lu_solve_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* lu_data), *$fptr-ptr:(Tensor* lu_pivots));
   }|] >> pure (out)


-- Tensor lu_solve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots)
--
lu_solve__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lu_solve__ttt self lu_data lu_pivots =  
  [C.block|Tensor* {
    return new Tensor(at::lu_solve(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* lu_data), *$fptr-ptr:(Tensor* lu_pivots)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _lu_solve_helper(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots)
--
_lu_solve_helper__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_lu_solve_helper__ttt self lu_data lu_pivots =  
  [C.block|Tensor* {
    return new Tensor(at::_lu_solve_helper(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* lu_data), *$fptr-ptr:(Tensor* lu_pivots)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & multinomial_out(Tensor & out, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator)
--
multinomial_out__tt6bg :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
multinomial_out__tt6bg out self num_samples replacement generator =  
  [C.block|void {
    at::multinomial_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t num_samples), $(bool replacement), $(Generator* generator));
   }|] >> pure (out)


-- Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator)
--
multinomial__t6bg :: ForeignPtr CTensor -> Int64 -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
multinomial__t6bg self num_samples replacement generator =  
  [C.block|Tensor* {
    return new Tensor(at::multinomial(*$fptr-ptr:(Tensor* self), $(int64_t num_samples), $(bool replacement), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> _multinomial_alias_setup(const Tensor & probs)
--
_multinomial_alias_setup__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_multinomial_alias_setup__t probs =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_multinomial_alias_setup(*$fptr-ptr:(Tensor* probs)));
   }|] >>= unTupleTensorTensor


-- Tensor _multinomial_alias_draw(const Tensor & J, const Tensor & q, int64_t num_samples, Generator * generator)
--
_multinomial_alias_draw__tt6g :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
_multinomial_alias_draw__tt6g j q num_samples generator =  
  [C.block|Tensor* {
    return new Tensor(at::_multinomial_alias_draw(*$fptr-ptr:(Tensor* j), *$fptr-ptr:(Tensor* q), $(int64_t num_samples), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & lgamma_out(Tensor & out, const Tensor & self)
--
lgamma_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lgamma_out__tt out self =  
  [C.block|void {
    at::lgamma_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor lgamma(const Tensor & self)
--
lgamma__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lgamma__t self =  
  [C.block|Tensor* {
    return new Tensor(at::lgamma(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & digamma_out(Tensor & out, const Tensor & self)
--
digamma_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
digamma_out__tt out self =  
  [C.block|void {
    at::digamma_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor digamma(const Tensor & self)
--
digamma__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
digamma__t self =  
  [C.block|Tensor* {
    return new Tensor(at::digamma(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & polygamma_out(Tensor & out, int64_t n, const Tensor & self)
--
polygamma_out__t6t :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
polygamma_out__t6t out n self =  
  [C.block|void {
    at::polygamma_out(*$fptr-ptr:(Tensor* out), $(int64_t n), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor polygamma(int64_t n, const Tensor & self)
--
polygamma__6t :: Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
polygamma__6t n self =  
  [C.block|Tensor* {
    return new Tensor(at::polygamma($(int64_t n), *$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor erfinv(const Tensor & self)
--
erfinv__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfinv__t self =  
  [C.block|Tensor* {
    return new Tensor(at::erfinv(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & erfinv_out(Tensor & out, const Tensor & self)
--
erfinv_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfinv_out__tt out self =  
  [C.block|void {
    at::erfinv_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor sign(const Tensor & self)
--
sign__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sign__t self =  
  [C.block|Tensor* {
    return new Tensor(at::sign(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sign_out(Tensor & out, const Tensor & self)
--
sign_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sign_out__tt out self =  
  [C.block|void {
    at::sign_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor dist(const Tensor & self, const Tensor & other, Scalar p)
--
dist__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
dist__tts self other p =  
  [C.block|Tensor* {
    return new Tensor(at::dist(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* p)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & atan2_out(Tensor & out, const Tensor & self, const Tensor & other)
--
atan2_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan2_out__ttt out self other =  
  [C.block|void {
    at::atan2_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor atan2(const Tensor & self, const Tensor & other)
--
atan2__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan2__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::atan2(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & lerp_out(Tensor & out, const Tensor & self, const Tensor & end, Scalar weight)
--
lerp_out__ttts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lerp_out__ttts out self end weight =  
  [C.block|void {
    at::lerp_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* end), *$fptr-ptr:(Scalar* weight));
   }|] >> pure (out)


-- Tensor & lerp_out(Tensor & out, const Tensor & self, const Tensor & end, const Tensor & weight)
--
lerp_out__tttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lerp_out__tttt out self end weight =  
  [C.block|void {
    at::lerp_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* end), *$fptr-ptr:(Tensor* weight));
   }|] >> pure (out)


-- Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight)
--
lerp__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lerp__tts self end weight =  
  [C.block|Tensor* {
    return new Tensor(at::lerp(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* end), *$fptr-ptr:(Scalar* weight)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor lerp(const Tensor & self, const Tensor & end, const Tensor & weight)
--
lerp__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lerp__ttt self end weight =  
  [C.block|Tensor* {
    return new Tensor(at::lerp(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* end), *$fptr-ptr:(Tensor* weight)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & histc_out(Tensor & out, const Tensor & self, int64_t bins, Scalar min, Scalar max)
--
histc_out__tt6ss :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
histc_out__tt6ss out self bins min max =  
  [C.block|void {
    at::histc_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t bins), *$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max));
   }|] >> pure (out)


-- Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max)
--
histc__t6ss :: ForeignPtr CTensor -> Int64 -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
histc__t6ss self bins min max =  
  [C.block|Tensor* {
    return new Tensor(at::histc(*$fptr-ptr:(Tensor* self), $(int64_t bins), *$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & fmod_out(Tensor & out, const Tensor & self, Scalar other)
--
fmod_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
fmod_out__tts out self other =  
  [C.block|void {
    at::fmod_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


-- Tensor fmod(const Tensor & self, Scalar other)
--
fmod__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
fmod__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::fmod(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & fmod_out(Tensor & out, const Tensor & self, const Tensor & other)
--
fmod_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fmod_out__ttt out self other =  
  [C.block|void {
    at::fmod_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor fmod(const Tensor & self, const Tensor & other)
--
fmod__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fmod__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::fmod(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & remainder_out(Tensor & out, const Tensor & self, Scalar other)
--
remainder_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
remainder_out__tts out self other =  
  [C.block|void {
    at::remainder_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


-- Tensor remainder(const Tensor & self, Scalar other)
--
remainder__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
remainder__ts self other =  
  [C.block|Tensor* {
    return new Tensor(at::remainder(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & remainder_out(Tensor & out, const Tensor & self, const Tensor & other)
--
remainder_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
remainder_out__ttt out self other =  
  [C.block|void {
    at::remainder_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor remainder(const Tensor & self, const Tensor & other)
--
remainder__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
remainder__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::remainder(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & min_out(Tensor & out, const Tensor & self, const Tensor & other)
--
min_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
min_out__ttt out self other =  
  [C.block|void {
    at::min_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor min(const Tensor & self, const Tensor & other)
--
min__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
min__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::min(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor min(const Tensor & self)
--
min__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
min__t self =  
  [C.block|Tensor* {
    return new Tensor(at::min(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & max_out(Tensor & out, const Tensor & self, const Tensor & other)
--
max_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
max_out__ttt out self other =  
  [C.block|void {
    at::max_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


-- Tensor max(const Tensor & self, const Tensor & other)
--
max__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
max__tt self other =  
  [C.block|Tensor* {
    return new Tensor(at::max(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor max(const Tensor & self)
--
max__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
max__t self =  
  [C.block|Tensor* {
    return new Tensor(at::max(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor median(const Tensor & self)
--
median__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
median__t self =  
  [C.block|Tensor* {
    return new Tensor(at::median(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor &,Tensor &> sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending)
--
sort_out__ttt6b :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
sort_out__ttt6b values indices self dim descending =  
  [C.block|void {
    at::sort_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool descending));
   }|] >> pure (values,indices)


-- std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending)
--
sort__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
sort__t6b self dim descending =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::sort(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool descending)));
   }|] >>= unTupleTensorTensor


-- Tensor argsort(const Tensor & self, int64_t dim, bool descending)
--
argsort__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
argsort__t6b self dim descending =  
  [C.block|Tensor* {
    return new Tensor(at::argsort(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool descending)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor &,Tensor &> topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted)
--
topk_out__ttt66bb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
topk_out__ttt66bb values indices self k dim largest sorted =  
  [C.block|void {
    at::topk_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t k), $(int64_t dim), $(bool largest), $(bool sorted));
   }|] >> pure (values,indices)


-- std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted)
--
topk__t66bb :: ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
topk__t66bb self k dim largest sorted =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::topk(*$fptr-ptr:(Tensor* self), $(int64_t k), $(int64_t dim), $(bool largest), $(bool sorted)));
   }|] >>= unTupleTensorTensor


-- Tensor all(const Tensor & self)
--
all__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
all__t self =  
  [C.block|Tensor* {
    return new Tensor(at::all(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor any(const Tensor & self)
--
any__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
any__t self =  
  [C.block|Tensor* {
    return new Tensor(at::any(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & renorm_out(Tensor & out, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm)
--
renorm_out__tts6s :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> Int64 -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
renorm_out__tts6s out self p dim maxnorm =  
  [C.block|void {
    at::renorm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), $(int64_t dim), *$fptr-ptr:(Scalar* maxnorm));
   }|] >> pure (out)


-- Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm)
--
renorm__ts6s :: ForeignPtr CTensor -> ForeignPtr CScalar -> Int64 -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
renorm__ts6s self p dim maxnorm =  
  [C.block|Tensor* {
    return new Tensor(at::renorm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), $(int64_t dim), *$fptr-ptr:(Scalar* maxnorm)));
   }|] >>= newForeignPtr deleteTensor


-- bool equal(const Tensor & self, const Tensor & other)
--
equal__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (CBool)
equal__tt self other =  
  [C.block|bool {
    return at::equal(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|]


-- Tensor & pow_out(Tensor & out, const Tensor & self, const Tensor & exponent)
--
pow_out__ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pow_out__ttt out self exponent =  
  [C.block|void {
    at::pow_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* exponent));
   }|] >> pure (out)


-- Tensor pow(const Tensor & self, const Tensor & exponent)
--
pow__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pow__tt self exponent =  
  [C.block|Tensor* {
    return new Tensor(at::pow(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* exponent)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & pow_out(Tensor & out, Scalar self, const Tensor & exponent)
--
pow_out__tst :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pow_out__tst out self exponent =  
  [C.block|void {
    at::pow_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Scalar* self), *$fptr-ptr:(Tensor* exponent));
   }|] >> pure (out)


-- Tensor pow(Scalar self, const Tensor & exponent)
--
pow__st :: ForeignPtr CScalar -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pow__st self exponent =  
  [C.block|Tensor* {
    return new Tensor(at::pow(*$fptr-ptr:(Scalar* self), *$fptr-ptr:(Tensor* exponent)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & normal_out(Tensor & out, const Tensor & mean, double std, Generator * generator)
--
normal_out__ttdg :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal_out__ttdg out mean std generator =  
  [C.block|void {
    at::normal_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* mean), $(double std), $(Generator* generator));
   }|] >> pure (out)


-- Tensor normal(const Tensor & mean, double std, Generator * generator)
--
normal__tdg :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal__tdg mean std generator =  
  [C.block|Tensor* {
    return new Tensor(at::normal(*$fptr-ptr:(Tensor* mean), $(double std), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & normal_out(Tensor & out, double mean, const Tensor & std, Generator * generator)
--
normal_out__tdtg :: ForeignPtr CTensor -> CDouble -> ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal_out__tdtg out mean std generator =  
  [C.block|void {
    at::normal_out(*$fptr-ptr:(Tensor* out), $(double mean), *$fptr-ptr:(Tensor* std), $(Generator* generator));
   }|] >> pure (out)


-- Tensor normal(double mean, const Tensor & std, Generator * generator)
--
normal__dtg :: CDouble -> ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal__dtg mean std generator =  
  [C.block|Tensor* {
    return new Tensor(at::normal($(double mean), *$fptr-ptr:(Tensor* std), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & normal_out(Tensor & out, const Tensor & mean, const Tensor & std, Generator * generator)
--
normal_out__tttg :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal_out__tttg out mean std generator =  
  [C.block|void {
    at::normal_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* mean), *$fptr-ptr:(Tensor* std), $(Generator* generator));
   }|] >> pure (out)


-- Tensor normal(const Tensor & mean, const Tensor & std, Generator * generator)
--
normal__ttg :: ForeignPtr CTensor -> ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal__ttg mean std generator =  
  [C.block|Tensor* {
    return new Tensor(at::normal(*$fptr-ptr:(Tensor* mean), *$fptr-ptr:(Tensor* std), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor normal(double mean, double std, IntArrayRef size, Generator * generator, const TensorOptions & options)
--
normal__ddago__1 :: CDouble -> CDouble -> Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
normal__ddago__1 mean std size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor(at::normal($(double mean), $(double std), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & normal_out(Tensor & out, double mean, double std, IntArrayRef size, Generator * generator)
--
normal_out__tddag :: ForeignPtr CTensor -> CDouble -> CDouble -> Vector Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal_out__tddag out mean std size generator =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    at::normal_out(*$fptr-ptr:(Tensor* out), $(double mean), $(double std), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator));
   }|] >> pure (out)


-- Tensor alias(const Tensor & self)
--
alias__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
alias__t self =  
  [C.block|Tensor* {
    return new Tensor(at::alias(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha)
--
_addr__tttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_addr__tttss self vec1 vec2 beta alpha =  
  [C.block|Tensor* {
    return new Tensor(at::_addr(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & _addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha)
--
_addr___tttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_addr___tttss self vec1 vec2 beta alpha =  
  [C.block|void {
    at::_addr_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor & _addr_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha)
--
_addr_out__ttttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_addr_out__ttttss out self vec1 vec2 beta alpha =  
  [C.block|void {
    at::_addr_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


-- Tensor & _index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source)
--
_index_copy___t6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_index_copy___t6tt self dim index source =  
  [C.block|void {
    at::_index_copy_(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source));
   }|] >> pure self


-- Tensor _cumsum(const Tensor & self, int64_t dim)
--
_cumsum__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_cumsum__t6 self dim =  
  [C.block|Tensor* {
    return new Tensor(at::_cumsum(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & _cumsum_out(Tensor & out, const Tensor & self, int64_t dim)
--
_cumsum_out__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_cumsum_out__tt6 out self dim =  
  [C.block|void {
    at::_cumsum_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|] >> pure (out)


-- Tensor _cumprod(const Tensor & self, int64_t dim)
--
_cumprod__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_cumprod__t6 self dim =  
  [C.block|Tensor* {
    return new Tensor(at::_cumprod(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & _cumprod_out(Tensor & out, const Tensor & self, int64_t dim)
--
_cumprod_out__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_cumprod_out__tt6 out self dim =  
  [C.block|void {
    at::_cumprod_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|] >> pure (out)


-- Tensor _var(const Tensor & self, bool unbiased)
--
_var__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_var__tb self unbiased =  
  [C.block|Tensor* {
    return new Tensor(at::_var(*$fptr-ptr:(Tensor* self), $(bool unbiased)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _std(const Tensor & self, bool unbiased)
--
_std__tb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_std__tb self unbiased =  
  [C.block|Tensor* {
    return new Tensor(at::_std(*$fptr-ptr:(Tensor* self), $(bool unbiased)));
   }|] >>= newForeignPtr deleteTensor


-- void _amp_non_finite_check_and_unscale_(Tensor & self, Tensor & found_inf, const Tensor & inv_scale)
--
_amp_non_finite_check_and_unscale___ttt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO ()
_amp_non_finite_check_and_unscale___ttt self found_inf inv_scale =  
  [C.block|void {
    at::_amp_non_finite_check_and_unscale_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* found_inf), *$fptr-ptr:(Tensor* inv_scale));
   }|]


-- Tensor _amp_update_scale(Tensor & growth_tracker, const Tensor & current_scale, const Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval)
--
_amp_update_scale__tttdd6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
_amp_update_scale__tttdd6 growth_tracker current_scale found_inf scale_growth_factor scale_backoff_factor growth_interval =  
  [C.block|Tensor* {
    return new Tensor(at::_amp_update_scale(*$fptr-ptr:(Tensor* growth_tracker), *$fptr-ptr:(Tensor* current_scale), *$fptr-ptr:(Tensor* found_inf), $(double scale_growth_factor), $(double scale_backoff_factor), $(int64_t growth_interval)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _cat(TensorList tensors, int64_t dim)
--
_cat__l6 :: Vector (Ptr CTensor) -> Int64 -> IO (ForeignPtr CTensor)
_cat__l6 tensors dim =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in 
  [C.block|Tensor* {
    return new Tensor(at::_cat(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size)), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & _cat_out(Tensor & out, TensorList tensors, int64_t dim)
--
_cat_out__tl6 :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Int64 -> IO (ForeignPtr CTensor)
_cat_out__tl6 out tensors dim =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in 
  [C.block|void {
    at::_cat_out(*$fptr-ptr:(Tensor* out), pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size)), $(int64_t dim));
   }|] >> pure (out)


-- std::tuple<Tensor,Tensor> _mode(const Tensor & self, int64_t dim, bool keepdim)
--
_mode__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_mode__t6b self dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_mode(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> _mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim)
--
_mode_out__ttt6b :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_mode_out__ttt6b values indices self dim keepdim =  
  [C.block|void {
    at::_mode_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (values,indices)


-- std::tuple<Tensor,Tensor> _max(const Tensor & self, int64_t dim, bool keepdim)
--
_max__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_max__t6b self dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_max(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> _max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim)
--
_max_out__ttt6b :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_max_out__ttt6b max max_indices self dim keepdim =  
  [C.block|void {
    at::_max_out(*$fptr-ptr:(Tensor* max), *$fptr-ptr:(Tensor* max_indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (max,max_indices)


-- std::tuple<Tensor,Tensor> _min(const Tensor & self, int64_t dim, bool keepdim)
--
_min__t6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_min__t6b self dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::_min(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> _min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim)
--
_min_out__ttt6b :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_min_out__ttt6b min min_indices self dim keepdim =  
  [C.block|void {
    at::_min_out(*$fptr-ptr:(Tensor* min), *$fptr-ptr:(Tensor* min_indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (min,min_indices)


-- Tensor & mse_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction)
--
mse_loss_out__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
mse_loss_out__ttt6 out self target reduction =  
  [C.block|void {
    at::mse_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction));
   }|] >> pure (out)


-- Tensor mse_loss(const Tensor & self, const Tensor & target, int64_t reduction)
--
mse_loss__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
mse_loss__tt6 self target reduction =  
  [C.block|Tensor* {
    return new Tensor(at::mse_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & l1_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction)
--
l1_loss_out__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
l1_loss_out__ttt6 out self target reduction =  
  [C.block|void {
    at::l1_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction));
   }|] >> pure (out)


-- Tensor l1_loss(const Tensor & self, const Tensor & target, int64_t reduction)
--
l1_loss__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
l1_loss__tt6 self target reduction =  
  [C.block|Tensor* {
    return new Tensor(at::l1_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & multi_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction)
--
multi_margin_loss_out__tttsst6__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
multi_margin_loss_out__tttsst6__1 out self target p margin weight reduction =  
  [C.block|void {
    at::multi_margin_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Scalar* p), *$fptr-ptr:(Scalar* margin), *$fptr-ptr:(Tensor* weight), $(int64_t reduction));
   }|] >> pure (out)


-- Tensor multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction)
--
multi_margin_loss__ttsst6__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
multi_margin_loss__ttsst6__1 self target p margin weight reduction =  
  [C.block|Tensor* {
    return new Tensor(at::multi_margin_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Scalar* p), *$fptr-ptr:(Scalar* margin), *$fptr-ptr:(Tensor* weight), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & multilabel_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction)
--
multilabel_margin_loss_out__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
multilabel_margin_loss_out__ttt6 out self target reduction =  
  [C.block|void {
    at::multilabel_margin_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction));
   }|] >> pure (out)


-- Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction)
--
multilabel_margin_loss__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
multilabel_margin_loss__tt6 self target reduction =  
  [C.block|Tensor* {
    return new Tensor(at::multilabel_margin_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & nll_loss_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index)
--
nll_loss_out__tttt66__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
nll_loss_out__tttt66__1 out self target weight reduction ignore_index =  
  [C.block|void {
    at::nll_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction), $(int64_t ignore_index));
   }|] >> pure (out)


-- Tensor nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index)
--
nll_loss__ttt66__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
nll_loss__ttt66__1 self target weight reduction ignore_index =  
  [C.block|Tensor* {
    return new Tensor(at::nll_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction), $(int64_t ignore_index)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & nll_loss2d_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index)
--
nll_loss2d_out__tttt66__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
nll_loss2d_out__tttt66__1 out self target weight reduction ignore_index =  
  [C.block|void {
    at::nll_loss2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction), $(int64_t ignore_index));
   }|] >> pure (out)


-- Tensor nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index)
--
nll_loss2d__ttt66__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
nll_loss2d__ttt66__1 self target weight reduction ignore_index =  
  [C.block|Tensor* {
    return new Tensor(at::nll_loss2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction), $(int64_t ignore_index)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & smooth_l1_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction)
--
smooth_l1_loss_out__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
smooth_l1_loss_out__ttt6 out self target reduction =  
  [C.block|void {
    at::smooth_l1_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction));
   }|] >> pure (out)


-- Tensor smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction)
--
smooth_l1_loss__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
smooth_l1_loss__tt6 self target reduction =  
  [C.block|Tensor* {
    return new Tensor(at::smooth_l1_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & soft_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction)
--
soft_margin_loss_out__ttt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
soft_margin_loss_out__ttt6 out self target reduction =  
  [C.block|void {
    at::soft_margin_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction));
   }|] >> pure (out)


-- Tensor soft_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction)
--
soft_margin_loss__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
soft_margin_loss__tt6 self target reduction =  
  [C.block|Tensor* {
    return new Tensor(at::soft_margin_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & elu_out(Tensor & out, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale)
--
elu_out__ttsss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
elu_out__ttsss out self alpha scale input_scale =  
  [C.block|void {
    at::elu_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* alpha), *$fptr-ptr:(Scalar* scale), *$fptr-ptr:(Scalar* input_scale));
   }|] >> pure (out)


-- Tensor elu(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale)
--
elu__tsss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
elu__tsss self alpha scale input_scale =  
  [C.block|Tensor* {
    return new Tensor(at::elu(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* alpha), *$fptr-ptr:(Scalar* scale), *$fptr-ptr:(Scalar* input_scale)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & elu_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale)
--
elu___tsss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
elu___tsss self alpha scale input_scale =  
  [C.block|void {
    at::elu_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* alpha), *$fptr-ptr:(Scalar* scale), *$fptr-ptr:(Scalar* input_scale));
   }|] >> pure self


-- Tensor & glu_out(Tensor & out, const Tensor & self, int64_t dim)
--
glu_out__tt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
glu_out__tt6 out self dim =  
  [C.block|void {
    at::glu_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|] >> pure (out)


-- Tensor glu(const Tensor & self, int64_t dim)
--
glu__t6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
glu__t6 self dim =  
  [C.block|Tensor* {
    return new Tensor(at::glu(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & hardsigmoid_out(Tensor & out, const Tensor & self)
--
hardsigmoid_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
hardsigmoid_out__tt out self =  
  [C.block|void {
    at::hardsigmoid_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor hardsigmoid(const Tensor & self)
--
hardsigmoid__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
hardsigmoid__t self =  
  [C.block|Tensor* {
    return new Tensor(at::hardsigmoid(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & hardsigmoid_(Tensor & self)
--
hardsigmoid___t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
hardsigmoid___t self =  
  [C.block|void {
    at::hardsigmoid_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


-- Tensor & hardtanh_out(Tensor & out, const Tensor & self, Scalar min_val, Scalar max_val)
--
hardtanh_out__ttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
hardtanh_out__ttss out self min_val max_val =  
  [C.block|void {
    at::hardtanh_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min_val), *$fptr-ptr:(Scalar* max_val));
   }|] >> pure (out)


-- Tensor hardtanh(const Tensor & self, Scalar min_val, Scalar max_val)
--
hardtanh__tss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
hardtanh__tss self min_val max_val =  
  [C.block|Tensor* {
    return new Tensor(at::hardtanh(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min_val), *$fptr-ptr:(Scalar* max_val)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & hardtanh_(Tensor & self, Scalar min_val, Scalar max_val)
--
hardtanh___tss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
hardtanh___tss self min_val max_val =  
  [C.block|void {
    at::hardtanh_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min_val), *$fptr-ptr:(Scalar* max_val));
   }|] >> pure self


-- Tensor & leaky_relu_out(Tensor & out, const Tensor & self, Scalar negative_slope)
--
leaky_relu_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
leaky_relu_out__tts out self negative_slope =  
  [C.block|void {
    at::leaky_relu_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* negative_slope));
   }|] >> pure (out)


-- Tensor leaky_relu(const Tensor & self, Scalar negative_slope)
--
leaky_relu__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
leaky_relu__ts self negative_slope =  
  [C.block|Tensor* {
    return new Tensor(at::leaky_relu(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* negative_slope)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & leaky_relu_(Tensor & self, Scalar negative_slope)
--
leaky_relu___ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
leaky_relu___ts self negative_slope =  
  [C.block|void {
    at::leaky_relu_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* negative_slope));
   }|] >> pure self


-- Tensor & log_sigmoid_out(Tensor & out, const Tensor & self)
--
log_sigmoid_out__tt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log_sigmoid_out__tt out self =  
  [C.block|void {
    at::log_sigmoid_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


-- Tensor log_sigmoid(const Tensor & self)
--
log_sigmoid__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log_sigmoid__t self =  
  [C.block|Tensor* {
    return new Tensor(at::log_sigmoid(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & rrelu_with_noise_out(Tensor & out, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator)
--
rrelu_with_noise_out__tttssbg :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
rrelu_with_noise_out__tttssbg out self noise lower upper training generator =  
  [C.block|void {
    at::rrelu_with_noise_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* noise), *$fptr-ptr:(Scalar* lower), *$fptr-ptr:(Scalar* upper), $(bool training), $(Generator* generator));
   }|] >> pure (out)


-- Tensor rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator)
--
rrelu_with_noise__ttssbg :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
rrelu_with_noise__ttssbg self noise lower upper training generator =  
  [C.block|Tensor* {
    return new Tensor(at::rrelu_with_noise(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* noise), *$fptr-ptr:(Scalar* lower), *$fptr-ptr:(Scalar* upper), $(bool training), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator)
--
rrelu_with_noise___ttssbg :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
rrelu_with_noise___ttssbg self noise lower upper training generator =  
  [C.block|void {
    at::rrelu_with_noise_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* noise), *$fptr-ptr:(Scalar* lower), *$fptr-ptr:(Scalar* upper), $(bool training), $(Generator* generator));
   }|] >> pure self


-- Tensor & softplus_out(Tensor & out, const Tensor & self, Scalar beta, Scalar threshold)
--
softplus_out__ttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
softplus_out__ttss out self beta threshold =  
  [C.block|void {
    at::softplus_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* threshold));
   }|] >> pure (out)


-- Tensor softplus(const Tensor & self, Scalar beta, Scalar threshold)
--
softplus__tss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
softplus__tss self beta threshold =  
  [C.block|Tensor* {
    return new Tensor(at::softplus(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* threshold)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & softshrink_out(Tensor & out, const Tensor & self, Scalar lambd)
--
softshrink_out__tts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
softshrink_out__tts out self lambd =  
  [C.block|void {
    at::softshrink_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* lambd));
   }|] >> pure (out)


-- Tensor softshrink(const Tensor & self, Scalar lambd)
--
softshrink__ts :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
softshrink__ts self lambd =  
  [C.block|Tensor* {
    return new Tensor(at::softshrink(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* lambd)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & adaptive_avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size)
--
adaptive_avg_pool2d_out__tta :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
adaptive_avg_pool2d_out__tta out self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|void {
    at::adaptive_avg_pool2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out)


-- Tensor adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size)
--
adaptive_avg_pool2d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
adaptive_avg_pool2d__ta self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|Tensor* {
    return new Tensor(at::adaptive_avg_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mkldnn_adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size)
--
mkldnn_adaptive_avg_pool2d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
mkldnn_adaptive_avg_pool2d__ta self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|Tensor* {
    return new Tensor(at::mkldnn_adaptive_avg_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor _adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size)
--
_adaptive_avg_pool2d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
_adaptive_avg_pool2d__ta self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|Tensor* {
    return new Tensor(at::_adaptive_avg_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & adaptive_avg_pool3d_out(Tensor & out, const Tensor & self, IntArrayRef output_size)
--
adaptive_avg_pool3d_out__tta :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
adaptive_avg_pool3d_out__tta out self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|void {
    at::adaptive_avg_pool3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out)


-- Tensor adaptive_avg_pool3d(const Tensor & self, IntArrayRef output_size)
--
adaptive_avg_pool3d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
adaptive_avg_pool3d__ta self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|Tensor* {
    return new Tensor(at::adaptive_avg_pool3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size)
--
adaptive_max_pool2d_out__ttta :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
adaptive_max_pool2d_out__ttta out indices self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|void {
    at::adaptive_max_pool2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out,indices)


-- std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & self, IntArrayRef output_size)
--
adaptive_max_pool2d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
adaptive_max_pool2d__ta self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::adaptive_max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size)
--
adaptive_max_pool3d_out__ttta :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
adaptive_max_pool3d_out__ttta out indices self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|void {
    at::adaptive_max_pool3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out,indices)


-- std::tuple<Tensor,Tensor> adaptive_max_pool3d(const Tensor & self, IntArrayRef output_size)
--
adaptive_max_pool3d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
adaptive_max_pool3d__ta self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::adaptive_max_pool3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= unTupleTensorTensor


-- Tensor & avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override)
--
avg_pool2d_out__ttaaabb6__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor)
avg_pool2d_out__ttaaabb6__1 out self kernel_size stride padding ceil_mode count_include_pad divisor_override =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in let (divisor_override__is_present, divisor_override__value) = splitMaybe divisor_override 0 in 
  [C.block|void {
    at::avg_pool2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad), ($(bool divisor_override__is_present) ? make_optional($(int64_t divisor_override__value)) : c10::nullopt));
   }|] >> pure (out)


-- Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override)
--
avg_pool2d__taaabb6__1 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor)
avg_pool2d__taaabb6__1 self kernel_size stride padding ceil_mode count_include_pad divisor_override =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in let (divisor_override__is_present, divisor_override__value) = splitMaybe divisor_override 0 in 
  [C.block|Tensor* {
    return new Tensor(at::avg_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad), ($(bool divisor_override__is_present) ? make_optional($(int64_t divisor_override__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & avg_pool3d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override)
--
avg_pool3d_out__ttaaabb6__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor)
avg_pool3d_out__ttaaabb6__1 out self kernel_size stride padding ceil_mode count_include_pad divisor_override =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in let (divisor_override__is_present, divisor_override__value) = splitMaybe divisor_override 0 in 
  [C.block|void {
    at::avg_pool3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad), ($(bool divisor_override__is_present) ? make_optional($(int64_t divisor_override__value)) : c10::nullopt));
   }|] >> pure (out)


-- Tensor avg_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override)
--
avg_pool3d__taaabb6__1 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor)
avg_pool3d__taaabb6__1 self kernel_size stride padding ceil_mode count_include_pad divisor_override =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in let (divisor_override__is_present, divisor_override__value) = splitMaybe divisor_override 0 in 
  [C.block|Tensor* {
    return new Tensor(at::avg_pool3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad), ($(bool divisor_override__is_present) ? make_optional($(int64_t divisor_override__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor &,Tensor &> fractional_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples)
--
fractional_max_pool2d_out__tttaat :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
fractional_max_pool2d_out__tttaat output indices self kernel_size output_size random_samples =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|void {
    at::fractional_max_pool2d_out(*$fptr-ptr:(Tensor* output), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), *$fptr-ptr:(Tensor* random_samples));
   }|] >> pure (output,indices)


-- std::tuple<Tensor,Tensor> fractional_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples)
--
fractional_max_pool2d__taat :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
fractional_max_pool2d__taat self kernel_size output_size random_samples =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::fractional_max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), *$fptr-ptr:(Tensor* random_samples)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> fractional_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples)
--
fractional_max_pool3d_out__tttaat :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
fractional_max_pool3d_out__tttaat output indices self kernel_size output_size random_samples =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|void {
    at::fractional_max_pool3d_out(*$fptr-ptr:(Tensor* output), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), *$fptr-ptr:(Tensor* random_samples));
   }|] >> pure (output,indices)


-- std::tuple<Tensor,Tensor> fractional_max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples)
--
fractional_max_pool3d__taat :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
fractional_max_pool3d__taat self kernel_size output_size random_samples =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::fractional_max_pool3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), *$fptr-ptr:(Tensor* random_samples)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool2d_with_indices_out__tttaaaab__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool2d_with_indices_out__tttaaaab__1 out indices self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void {
    at::max_pool2d_with_indices_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode));
   }|] >> pure (out,indices)


-- std::tuple<Tensor,Tensor> max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool2d_with_indices__taaaab__1 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool2d_with_indices__taaaab__1 self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::max_pool2d_with_indices(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor &,Tensor &> max_pool3d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool3d_with_indices_out__tttaaaab__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool3d_with_indices_out__tttaaaab__1 out indices self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void {
    at::max_pool3d_with_indices_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode));
   }|] >> pure (out,indices)


-- std::tuple<Tensor,Tensor> max_pool3d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
--
max_pool3d_with_indices__taaaab__1 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool3d_with_indices__taaaab__1 self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(at::max_pool3d_with_indices(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= unTupleTensorTensor


-- Tensor & max_unpool2d_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size)
--
max_unpool2d_out__ttta :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
max_unpool2d_out__ttta out self indices output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|void {
    at::max_unpool2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* indices), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out)


-- Tensor max_unpool2d(const Tensor & self, const Tensor & indices, IntArrayRef output_size)
--
max_unpool2d__tta :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
max_unpool2d__tta self indices output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in 
  [C.block|Tensor* {
    return new Tensor(at::max_unpool2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* indices), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & max_unpool3d_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding)
--
max_unpool3d_out__tttaaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
max_unpool3d_out__tttaaa out self indices output_size stride padding =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|void {
    at::max_unpool3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* indices), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


-- Tensor max_unpool3d(const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding)
--
max_unpool3d__ttaaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
max_unpool3d__ttaaa self indices output_size stride padding =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|Tensor* {
    return new Tensor(at::max_unpool3d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* indices), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & reflection_pad1d_out(Tensor & out, const Tensor & self, IntArrayRef padding)
--
reflection_pad1d_out__tta :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
reflection_pad1d_out__tta out self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|void {
    at::reflection_pad1d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


-- Tensor reflection_pad1d(const Tensor & self, IntArrayRef padding)
--
reflection_pad1d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
reflection_pad1d__ta self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|Tensor* {
    return new Tensor(at::reflection_pad1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & reflection_pad2d_out(Tensor & out, const Tensor & self, IntArrayRef padding)
--
reflection_pad2d_out__tta :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
reflection_pad2d_out__tta out self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|void {
    at::reflection_pad2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


-- Tensor reflection_pad2d(const Tensor & self, IntArrayRef padding)
--
reflection_pad2d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
reflection_pad2d__ta self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|Tensor* {
    return new Tensor(at::reflection_pad2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & replication_pad1d_out(Tensor & out, const Tensor & self, IntArrayRef padding)
--
replication_pad1d_out__tta :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
replication_pad1d_out__tta out self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|void {
    at::replication_pad1d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


-- Tensor replication_pad1d(const Tensor & self, IntArrayRef padding)
--
replication_pad1d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
replication_pad1d__ta self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|Tensor* {
    return new Tensor(at::replication_pad1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & replication_pad2d_out(Tensor & out, const Tensor & self, IntArrayRef padding)
--
replication_pad2d_out__tta :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
replication_pad2d_out__tta out self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|void {
    at::replication_pad2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


-- Tensor replication_pad2d(const Tensor & self, IntArrayRef padding)
--
replication_pad2d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
replication_pad2d__ta self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|Tensor* {
    return new Tensor(at::replication_pad2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & replication_pad3d_out(Tensor & out, const Tensor & self, IntArrayRef padding)
--
replication_pad3d_out__tta :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
replication_pad3d_out__tta out self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|void {
    at::replication_pad3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


-- Tensor replication_pad3d(const Tensor & self, IntArrayRef padding)
--
replication_pad3d__ta :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
replication_pad3d__ta self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|Tensor* {
    return new Tensor(at::replication_pad3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & upsample_linear1d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales)
--
upsample_linear1d_out__ttabd :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_linear1d_out__ttabd out self output_size align_corners scales =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales__is_present, scales__value) = splitMaybe scales 0 in 
  [C.block|void {
    at::upsample_linear1d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners), ($(bool scales__is_present) ? make_optional($(double scales__value)) : c10::nullopt));
   }|] >> pure (out)


-- Tensor upsample_linear1d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales)
--
upsample_linear1d__tabd :: ForeignPtr CTensor -> Vector Int64 -> CBool -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_linear1d__tabd self output_size align_corners scales =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales__is_present, scales__value) = splitMaybe scales 0 in 
  [C.block|Tensor* {
    return new Tensor(at::upsample_linear1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners), ($(bool scales__is_present) ? make_optional($(double scales__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & upsample_bilinear2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w)
--
upsample_bilinear2d_out__ttabdd :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> Maybe CDouble -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_bilinear2d_out__ttabdd out self output_size align_corners scales_h scales_w =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales_h__is_present, scales_h__value) = splitMaybe scales_h 0 in let (scales_w__is_present, scales_w__value) = splitMaybe scales_w 0 in 
  [C.block|void {
    at::upsample_bilinear2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners), ($(bool scales_h__is_present) ? make_optional($(double scales_h__value)) : c10::nullopt), ($(bool scales_w__is_present) ? make_optional($(double scales_w__value)) : c10::nullopt));
   }|] >> pure (out)


-- Tensor upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w)
--
upsample_bilinear2d__tabdd :: ForeignPtr CTensor -> Vector Int64 -> CBool -> Maybe CDouble -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_bilinear2d__tabdd self output_size align_corners scales_h scales_w =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales_h__is_present, scales_h__value) = splitMaybe scales_h 0 in let (scales_w__is_present, scales_w__value) = splitMaybe scales_w 0 in 
  [C.block|Tensor* {
    return new Tensor(at::upsample_bilinear2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners), ($(bool scales_h__is_present) ? make_optional($(double scales_h__value)) : c10::nullopt), ($(bool scales_w__is_present) ? make_optional($(double scales_w__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & upsample_bicubic2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w)
--
upsample_bicubic2d_out__ttabdd :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> Maybe CDouble -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_bicubic2d_out__ttabdd out self output_size align_corners scales_h scales_w =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales_h__is_present, scales_h__value) = splitMaybe scales_h 0 in let (scales_w__is_present, scales_w__value) = splitMaybe scales_w 0 in 
  [C.block|void {
    at::upsample_bicubic2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners), ($(bool scales_h__is_present) ? make_optional($(double scales_h__value)) : c10::nullopt), ($(bool scales_w__is_present) ? make_optional($(double scales_w__value)) : c10::nullopt));
   }|] >> pure (out)


-- Tensor upsample_bicubic2d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w)
--
upsample_bicubic2d__tabdd :: ForeignPtr CTensor -> Vector Int64 -> CBool -> Maybe CDouble -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_bicubic2d__tabdd self output_size align_corners scales_h scales_w =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales_h__is_present, scales_h__value) = splitMaybe scales_h 0 in let (scales_w__is_present, scales_w__value) = splitMaybe scales_w 0 in 
  [C.block|Tensor* {
    return new Tensor(at::upsample_bicubic2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners), ($(bool scales_h__is_present) ? make_optional($(double scales_h__value)) : c10::nullopt), ($(bool scales_w__is_present) ? make_optional($(double scales_w__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & upsample_trilinear3d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w)
--
upsample_trilinear3d_out__ttabddd :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> Maybe CDouble -> Maybe CDouble -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_trilinear3d_out__ttabddd out self output_size align_corners scales_d scales_h scales_w =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales_d__is_present, scales_d__value) = splitMaybe scales_d 0 in let (scales_h__is_present, scales_h__value) = splitMaybe scales_h 0 in let (scales_w__is_present, scales_w__value) = splitMaybe scales_w 0 in 
  [C.block|void {
    at::upsample_trilinear3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners), ($(bool scales_d__is_present) ? make_optional($(double scales_d__value)) : c10::nullopt), ($(bool scales_h__is_present) ? make_optional($(double scales_h__value)) : c10::nullopt), ($(bool scales_w__is_present) ? make_optional($(double scales_w__value)) : c10::nullopt));
   }|] >> pure (out)


-- Tensor upsample_trilinear3d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w)
--
upsample_trilinear3d__tabddd :: ForeignPtr CTensor -> Vector Int64 -> CBool -> Maybe CDouble -> Maybe CDouble -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_trilinear3d__tabddd self output_size align_corners scales_d scales_h scales_w =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales_d__is_present, scales_d__value) = splitMaybe scales_d 0 in let (scales_h__is_present, scales_h__value) = splitMaybe scales_h 0 in let (scales_w__is_present, scales_w__value) = splitMaybe scales_w 0 in 
  [C.block|Tensor* {
    return new Tensor(at::upsample_trilinear3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners), ($(bool scales_d__is_present) ? make_optional($(double scales_d__value)) : c10::nullopt), ($(bool scales_h__is_present) ? make_optional($(double scales_h__value)) : c10::nullopt), ($(bool scales_w__is_present) ? make_optional($(double scales_w__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & upsample_nearest1d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales)
--
upsample_nearest1d_out__ttad :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_nearest1d_out__ttad out self output_size scales =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales__is_present, scales__value) = splitMaybe scales 0 in 
  [C.block|void {
    at::upsample_nearest1d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ($(bool scales__is_present) ? make_optional($(double scales__value)) : c10::nullopt));
   }|] >> pure (out)


-- Tensor upsample_nearest1d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales)
--
upsample_nearest1d__tad :: ForeignPtr CTensor -> Vector Int64 -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_nearest1d__tad self output_size scales =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales__is_present, scales__value) = splitMaybe scales 0 in 
  [C.block|Tensor* {
    return new Tensor(at::upsample_nearest1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ($(bool scales__is_present) ? make_optional($(double scales__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & upsample_nearest2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w)
--
upsample_nearest2d_out__ttadd :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Maybe CDouble -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_nearest2d_out__ttadd out self output_size scales_h scales_w =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales_h__is_present, scales_h__value) = splitMaybe scales_h 0 in let (scales_w__is_present, scales_w__value) = splitMaybe scales_w 0 in 
  [C.block|void {
    at::upsample_nearest2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ($(bool scales_h__is_present) ? make_optional($(double scales_h__value)) : c10::nullopt), ($(bool scales_w__is_present) ? make_optional($(double scales_w__value)) : c10::nullopt));
   }|] >> pure (out)


-- Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w)
--
upsample_nearest2d__tadd :: ForeignPtr CTensor -> Vector Int64 -> Maybe CDouble -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_nearest2d__tadd self output_size scales_h scales_w =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales_h__is_present, scales_h__value) = splitMaybe scales_h 0 in let (scales_w__is_present, scales_w__value) = splitMaybe scales_w 0 in 
  [C.block|Tensor* {
    return new Tensor(at::upsample_nearest2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ($(bool scales_h__is_present) ? make_optional($(double scales_h__value)) : c10::nullopt), ($(bool scales_w__is_present) ? make_optional($(double scales_w__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & upsample_nearest3d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w)
--
upsample_nearest3d_out__ttaddd :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Maybe CDouble -> Maybe CDouble -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_nearest3d_out__ttaddd out self output_size scales_d scales_h scales_w =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales_d__is_present, scales_d__value) = splitMaybe scales_d 0 in let (scales_h__is_present, scales_h__value) = splitMaybe scales_h 0 in let (scales_w__is_present, scales_w__value) = splitMaybe scales_w 0 in 
  [C.block|void {
    at::upsample_nearest3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ($(bool scales_d__is_present) ? make_optional($(double scales_d__value)) : c10::nullopt), ($(bool scales_h__is_present) ? make_optional($(double scales_h__value)) : c10::nullopt), ($(bool scales_w__is_present) ? make_optional($(double scales_w__value)) : c10::nullopt));
   }|] >> pure (out)


-- Tensor upsample_nearest3d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w)
--
upsample_nearest3d__taddd :: ForeignPtr CTensor -> Vector Int64 -> Maybe CDouble -> Maybe CDouble -> Maybe CDouble -> IO (ForeignPtr CTensor)
upsample_nearest3d__taddd self output_size scales_d scales_h scales_w =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in let (scales_d__is_present, scales_d__value) = splitMaybe scales_d 0 in let (scales_h__is_present, scales_h__value) = splitMaybe scales_h 0 in let (scales_w__is_present, scales_w__value) = splitMaybe scales_w 0 in 
  [C.block|Tensor* {
    return new Tensor(at::upsample_nearest3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ($(bool scales_d__is_present) ? make_optional($(double scales_d__value)) : c10::nullopt), ($(bool scales_h__is_present) ? make_optional($(double scales_h__value)) : c10::nullopt), ($(bool scales_w__is_present) ? make_optional($(double scales_w__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & slow_conv_transpose2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation)
--
slow_conv_transpose2d_out__tttataaaa__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv_transpose2d_out__tttataaaa__1 out self weight kernel_size bias stride padding output_padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void {
    at::slow_conv_transpose2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)));
   }|] >> pure (out)


-- Tensor slow_conv_transpose2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation)
--
slow_conv_transpose2d__ttataaaa__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv_transpose2d__ttataaaa__1 self weight kernel_size bias stride padding output_padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::slow_conv_transpose2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & slow_conv_transpose3d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation)
--
slow_conv_transpose3d_out__tttataaaa__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv_transpose3d_out__tttataaaa__1 out self weight kernel_size bias stride padding output_padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void {
    at::slow_conv_transpose3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)));
   }|] >> pure (out)


-- Tensor slow_conv_transpose3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation)
--
slow_conv_transpose3d__ttataaaa__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv_transpose3d__ttataaaa__1 self weight kernel_size bias stride padding output_padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::slow_conv_transpose3d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & thnn_conv2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding)
--
thnn_conv2d_out__tttataa__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv2d_out__tttataa__1 out self weight kernel_size bias stride padding =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|void {
    at::thnn_conv2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


-- Tensor thnn_conv2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding)
--
thnn_conv2d__ttataa__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv2d__ttataa__1 self weight kernel_size bias stride padding =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|Tensor* {
    return new Tensor(at::thnn_conv2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & thnn_conv_depthwise2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation)
--
thnn_conv_depthwise2d_out__tttataaa__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv_depthwise2d_out__tttataaa__1 out self weight kernel_size bias stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|void {
    at::thnn_conv_depthwise2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)));
   }|] >> pure (out)


-- Tensor thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation)
--
thnn_conv_depthwise2d__ttataaa__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv_depthwise2d__ttataaa__1 self weight kernel_size bias stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::thnn_conv_depthwise2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & slow_conv3d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding)
--
slow_conv3d_out__tttataa__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv3d_out__tttataa__1 out self weight kernel_size bias stride padding =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|void {
    at::slow_conv3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


-- Tensor slow_conv3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding)
--
slow_conv3d__ttataa__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv3d__ttataa__1 self weight kernel_size bias stride padding =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in 
  [C.block|Tensor* {
    return new Tensor(at::slow_conv3d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor slow_conv_dilated2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation)
--
slow_conv_dilated2d__ttataaa__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv_dilated2d__ttataaa__1 self weight kernel_size bias stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::slow_conv_dilated2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor slow_conv_dilated3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation)
--
slow_conv_dilated3d__ttataaa__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
slow_conv_dilated3d__ttataaa__1 self weight kernel_size bias stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in 
  [C.block|Tensor* {
    return new Tensor(at::slow_conv_dilated3d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & col2im_out(Tensor & out, const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride)
--
col2im_out__ttaaaaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
col2im_out__ttaaaaa out self output_size kernel_size dilation padding stride =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in 
  [C.block|void {
    at::col2im_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)));
   }|] >> pure (out)


-- Tensor col2im(const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride)
--
col2im__taaaaa :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
col2im__taaaaa self output_size kernel_size dilation padding stride =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in 
  [C.block|Tensor* {
    return new Tensor(at::col2im(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & im2col_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride)
--
im2col_out__ttaaaa :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
im2col_out__ttaaaa out self kernel_size dilation padding stride =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in 
  [C.block|void {
    at::im2col_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)));
   }|] >> pure (out)


-- Tensor im2col(const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride)
--
im2col__taaaa :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
im2col__taaaa self kernel_size dilation padding stride =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in 
  [C.block|Tensor* {
    return new Tensor(at::im2col(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor isfinite(const Tensor & self)
--
isfinite__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
isfinite__t self =  
  [C.block|Tensor* {
    return new Tensor(at::isfinite(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor isinf(const Tensor & self)
--
isinf__t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
isinf__t self =  
  [C.block|Tensor* {
    return new Tensor(at::isinf(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor



-- void backward(const Tensor & gradient, bool keep_graph, bool create_graph)
--
backward_mtbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> IO ()
backward_mtbb self gradient keep_graph create_graph =  
  [C.block|void {
    return $fptr-ptr:(Tensor *self)->backward(*$fptr-ptr:(Tensor* gradient), $(bool keep_graph), $(bool create_graph));
   }|]


-- void set_data(const Tensor & new_data)
--
set_data_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO ()
set_data_mt self new_data =  
  [C.block|void {
    return $fptr-ptr:(Tensor *self)->set_data(*$fptr-ptr:(Tensor* new_data));
   }|]


-- Tensor data()
--
data_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
data_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->data());
   }|] >>= newForeignPtr deleteTensor


-- bool is_leaf()
--
is_leaf_m :: ForeignPtr CTensor -> IO (CBool)
is_leaf_m self =  
  [C.block|bool {
    return $fptr-ptr:(Tensor *self)->is_leaf();
   }|]


-- int64_t output_nr()
--
output_nr_m :: ForeignPtr CTensor -> IO (Int64)
output_nr_m self =  
  [C.block|int64_t {
    return $fptr-ptr:(Tensor *self)->output_nr();
   }|]


-- int64_t _version()
--
_version_m :: ForeignPtr CTensor -> IO (Int64)
_version_m self =  
  [C.block|int64_t {
    return $fptr-ptr:(Tensor *self)->_version();
   }|]


-- Tensor & requires_grad_(bool _requires_grad)
--
requires_grad__mb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
requires_grad__mb self _requires_grad =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->requires_grad_($(bool _requires_grad));
   }|] >> pure self


-- void retain_grad()
--
retain_grad_m :: ForeignPtr CTensor -> IO ()
retain_grad_m self =  
  [C.block|void {
    return $fptr-ptr:(Tensor *self)->retain_grad();
   }|]


-- Tensor align_as(const Tensor & other)
--
align_as_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
align_as_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->align_as(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor abs()
--
abs_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
abs_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->abs());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & abs_()
--
abs__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
abs__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->abs_();
   }|] >> pure self


-- Tensor angle()
--
angle_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
angle_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->angle());
   }|] >>= newForeignPtr deleteTensor


-- Tensor conj()
--
conj_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
conj_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->conj());
   }|] >>= newForeignPtr deleteTensor


-- Tensor acos()
--
acos_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
acos_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->acos());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & acos_()
--
acos__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
acos__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->acos_();
   }|] >> pure self


-- Tensor add(const Tensor & other, Scalar alpha)
--
add_mts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
add_mts self other alpha =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->add(*$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & add_(const Tensor & other, Scalar alpha)
--
add__mts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
add__mts self other alpha =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->add_(*$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor add(Scalar other, Scalar alpha)
--
add_mss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
add_mss self other alpha =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->add(*$fptr-ptr:(Scalar* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & add_(Scalar other, Scalar alpha)
--
add__mss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
add__mss self other alpha =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->add_(*$fptr-ptr:(Scalar* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor addmv(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha)
--
addmv_mttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmv_mttss self mat vec beta alpha =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->addmv(*$fptr-ptr:(Tensor* mat), *$fptr-ptr:(Tensor* vec), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & addmv_(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha)
--
addmv__mttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmv__mttss self mat vec beta alpha =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->addmv_(*$fptr-ptr:(Tensor* mat), *$fptr-ptr:(Tensor* vec), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor addr(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha)
--
addr_mttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addr_mttss self vec1 vec2 beta alpha =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->addr(*$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha)
--
addr__mttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addr__mttss self vec1 vec2 beta alpha =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->addr_(*$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor all(int64_t dim, bool keepdim)
--
all_m6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
all_m6b self dim keepdim =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->all($(int64_t dim), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- bool allclose(const Tensor & other, double rtol, double atol, bool equal_nan)
--
allclose_mtddb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> CBool -> IO (CBool)
allclose_mtddb self other rtol atol equal_nan =  
  [C.block|bool {
    return $fptr-ptr:(Tensor *self)->allclose(*$fptr-ptr:(Tensor* other), $(double rtol), $(double atol), $(bool equal_nan));
   }|]


-- Tensor any(int64_t dim, bool keepdim)
--
any_m6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
any_m6b self dim keepdim =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->any($(int64_t dim), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor argmax(c10::optional<int64_t> dim, bool keepdim)
--
argmax_m6b :: ForeignPtr CTensor -> Maybe Int64 -> CBool -> IO (ForeignPtr CTensor)
argmax_m6b self dim keepdim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->argmax(($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor argmin(c10::optional<int64_t> dim, bool keepdim)
--
argmin_m6b :: ForeignPtr CTensor -> Maybe Int64 -> CBool -> IO (ForeignPtr CTensor)
argmin_m6b self dim keepdim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->argmin(($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor as_strided(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset)
--
as_strided_maa6 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Maybe Int64 -> IO (ForeignPtr CTensor)
as_strided_maa6 self size stride storage_offset =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in let (storage_offset__is_present, storage_offset__value) = splitMaybe storage_offset 0 in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->as_strided(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ($(bool storage_offset__is_present) ? make_optional($(int64_t storage_offset__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & as_strided_(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset)
--
as_strided__maa6 :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Maybe Int64 -> IO (ForeignPtr CTensor)
as_strided__maa6 self size stride storage_offset =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in let (storage_offset__is_present, storage_offset__value) = splitMaybe storage_offset 0 in 
  [C.block|void {
    $fptr-ptr:(Tensor *self)->as_strided_(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ($(bool storage_offset__is_present) ? make_optional($(int64_t storage_offset__value)) : c10::nullopt));
   }|] >> pure self


-- Tensor asin()
--
asin_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
asin_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->asin());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & asin_()
--
asin__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
asin__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->asin_();
   }|] >> pure self


-- Tensor atan()
--
atan_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->atan());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & atan_()
--
atan__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->atan_();
   }|] >> pure self


-- Tensor baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha)
--
baddbmm_mttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
baddbmm_mttss self batch1 batch2 beta alpha =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->baddbmm(*$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & baddbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha)
--
baddbmm__mttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
baddbmm__mttss self batch1 batch2 beta alpha =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->baddbmm_(*$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor bernoulli(Generator * generator)
--
bernoulli_mg :: ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
bernoulli_mg self generator =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->bernoulli($(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & bernoulli_(const Tensor & p, Generator * generator)
--
bernoulli__mtg :: ForeignPtr CTensor -> ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
bernoulli__mtg self p generator =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->bernoulli_(*$fptr-ptr:(Tensor* p), $(Generator* generator));
   }|] >> pure self


-- Tensor & bernoulli_(double p, Generator * generator)
--
bernoulli__mdg :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
bernoulli__mdg self p generator =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->bernoulli_($(double p), $(Generator* generator));
   }|] >> pure self


-- Tensor bernoulli(double p, Generator * generator)
--
bernoulli_mdg :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
bernoulli_mdg self p generator =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->bernoulli($(double p), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bincount(const Tensor & weights, int64_t minlength)
--
bincount_mt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
bincount_mt6 self weights minlength =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->bincount(*$fptr-ptr:(Tensor* weights), $(int64_t minlength)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bitwise_not()
--
bitwise_not_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_not_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->bitwise_not());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & bitwise_not_()
--
bitwise_not__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_not__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->bitwise_not_();
   }|] >> pure self


-- Tensor logical_not()
--
logical_not_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_not_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->logical_not());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & logical_not_()
--
logical_not__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_not__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->logical_not_();
   }|] >> pure self


-- Tensor logical_xor(const Tensor & other)
--
logical_xor_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_xor_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->logical_xor(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & logical_xor_(const Tensor & other)
--
logical_xor__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_xor__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->logical_xor_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor logical_and(const Tensor & other)
--
logical_and_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_and_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->logical_and(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & logical_and_(const Tensor & other)
--
logical_and__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_and__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->logical_and_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor logical_or(const Tensor & other)
--
logical_or_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_or_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->logical_or(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & logical_or_(const Tensor & other)
--
logical_or__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logical_or__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->logical_or_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor bmm(const Tensor & mat2)
--
bmm_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bmm_mt self mat2 =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->bmm(*$fptr-ptr:(Tensor* mat2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor ceil()
--
ceil_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ceil_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->ceil());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & ceil_()
--
ceil__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ceil__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->ceil_();
   }|] >> pure self


-- std::vector<Tensor> chunk(int64_t chunks, int64_t dim)
--
chunk_m66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (Vector (Ptr CTensor))
chunk_m66 self chunks dim =  
  [C.block|void* {
    return (void*)new std::vector<Tensor>($fptr-ptr:(Tensor *self)->chunk($(int64_t chunks), $(int64_t dim)));
   }|] >>= unVectorTensor


-- Tensor clamp(c10::optional<Scalar> min, c10::optional<Scalar> max)
--
clamp_mss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_mss self min max =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->clamp(*$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & clamp_(c10::optional<Scalar> min, c10::optional<Scalar> max)
--
clamp__mss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp__mss self min max =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->clamp_(*$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max));
   }|] >> pure self


-- Tensor clamp_max(Scalar max)
--
clamp_max_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_max_ms self max =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->clamp_max(*$fptr-ptr:(Scalar* max)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & clamp_max_(Scalar max)
--
clamp_max__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_max__ms self max =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->clamp_max_(*$fptr-ptr:(Scalar* max));
   }|] >> pure self


-- Tensor clamp_min(Scalar min)
--
clamp_min_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_min_ms self min =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->clamp_min(*$fptr-ptr:(Scalar* min)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & clamp_min_(Scalar min)
--
clamp_min__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_min__ms self min =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->clamp_min_(*$fptr-ptr:(Scalar* min));
   }|] >> pure self


-- Tensor contiguous(MemoryFormat memory_format)
--
contiguous_mm :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
contiguous_mm self memory_format =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->contiguous(static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & copy_(const Tensor & src, bool non_blocking)
--
copy__mtb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
copy__mtb self src non_blocking =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->copy_(*$fptr-ptr:(Tensor* src), $(bool non_blocking));
   }|] >> pure self


-- Tensor cos()
--
cos_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cos_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->cos());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & cos_()
--
cos__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cos__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->cos_();
   }|] >> pure self


-- Tensor cosh()
--
cosh_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cosh_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->cosh());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & cosh_()
--
cosh__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cosh__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->cosh_();
   }|] >> pure self


-- std::tuple<Tensor,Tensor> cummax(int64_t dim)
--
cummax_m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
cummax_m6 self dim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->cummax($(int64_t dim)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> cummin(int64_t dim)
--
cummin_m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
cummin_m6 self dim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->cummin($(int64_t dim)));
   }|] >>= unTupleTensorTensor


-- Tensor cumprod(int64_t dim, c10::optional<ScalarType> dtype)
--
cumprod_m6s :: ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
cumprod_m6s self dim dtype =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->cumprod($(int64_t dim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor cumsum(int64_t dim, c10::optional<ScalarType> dtype)
--
cumsum_m6s :: ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
cumsum_m6s self dim dtype =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->cumsum($(int64_t dim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor det()
--
det_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
det_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->det());
   }|] >>= newForeignPtr deleteTensor


-- Tensor diag_embed(int64_t offset, int64_t dim1, int64_t dim2)
--
diag_embed_m666 :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
diag_embed_m666 self offset dim1 dim2 =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->diag_embed($(int64_t offset), $(int64_t dim1), $(int64_t dim2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor diagflat(int64_t offset)
--
diagflat_m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
diagflat_m6 self offset =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->diagflat($(int64_t offset)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor diagonal(int64_t offset, int64_t dim1, int64_t dim2)
--
diagonal_m666 :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
diagonal_m666 self offset dim1 dim2 =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->diagonal($(int64_t offset), $(int64_t dim1), $(int64_t dim2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & fill_diagonal_(Scalar fill_value, bool wrap)
--
fill_diagonal__msb :: ForeignPtr CTensor -> ForeignPtr CScalar -> CBool -> IO (ForeignPtr CTensor)
fill_diagonal__msb self fill_value wrap =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->fill_diagonal_(*$fptr-ptr:(Scalar* fill_value), $(bool wrap));
   }|] >> pure self


-- Tensor div(const Tensor & other)
--
div_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
div_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->div(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & div_(const Tensor & other)
--
div__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
div__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->div_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor div(Scalar other)
--
div_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
div_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->div(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & div_(Scalar other)
--
div__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
div__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->div_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor dot(const Tensor & tensor)
--
dot_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
dot_mt self tensor =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->dot(*$fptr-ptr:(Tensor* tensor)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor new_empty(IntArrayRef size, const TensorOptions & options)
--
new_empty_mao :: ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
new_empty_mao self size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->new_empty(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor new_full(IntArrayRef size, Scalar fill_value, const TensorOptions & options)
--
new_full_maso :: ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
new_full_maso self size fill_value options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->new_full(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(Scalar* fill_value), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor new_zeros(IntArrayRef size, const TensorOptions & options)
--
new_zeros_mao :: ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
new_zeros_mao self size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->new_zeros(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & resize_(IntArrayRef size, c10::optional<MemoryFormat> memory_format)
--
resize__mam :: ForeignPtr CTensor -> Vector Int64 -> Int8 -> IO (ForeignPtr CTensor)
resize__mam self size memory_format =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    $fptr-ptr:(Tensor *self)->resize_(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), static_cast<MemoryFormat>($(int8_t memory_format)));
   }|] >> pure self


-- Tensor erf()
--
erf_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erf_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->erf());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & erf_()
--
erf__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erf__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->erf_();
   }|] >> pure self


-- Tensor erfc()
--
erfc_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfc_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->erfc());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & erfc_()
--
erfc__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfc__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->erfc_();
   }|] >> pure self


-- Tensor exp()
--
exp_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
exp_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->exp());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & exp_()
--
exp__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
exp__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->exp_();
   }|] >> pure self


-- Tensor expm1()
--
expm1_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
expm1_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->expm1());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & expm1_()
--
expm1__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
expm1__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->expm1_();
   }|] >> pure self


-- Tensor expand(IntArrayRef size, bool implicit)
--
expand_mab :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
expand_mab self size implicit =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->expand(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(bool implicit)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor expand_as(const Tensor & other)
--
expand_as_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
expand_as_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->expand_as(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor flatten(int64_t start_dim, int64_t end_dim)
--
flatten_m66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
flatten_m66 self start_dim end_dim =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->flatten($(int64_t start_dim), $(int64_t end_dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & fill_(Scalar value)
--
fill__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
fill__ms self value =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->fill_(*$fptr-ptr:(Scalar* value));
   }|] >> pure self


-- Tensor & fill_(const Tensor & value)
--
fill__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fill__mt self value =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->fill_(*$fptr-ptr:(Tensor* value));
   }|] >> pure self


-- Tensor floor()
--
floor_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
floor_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->floor());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & floor_()
--
floor__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
floor__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->floor_();
   }|] >> pure self


-- Tensor floor_divide(const Tensor & other)
--
floor_divide_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
floor_divide_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->floor_divide(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & floor_divide_(const Tensor & other)
--
floor_divide__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
floor_divide__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->floor_divide_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor floor_divide(Scalar other)
--
floor_divide_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
floor_divide_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->floor_divide(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & floor_divide_(Scalar other)
--
floor_divide__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
floor_divide__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->floor_divide_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor frac()
--
frac_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
frac_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->frac());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & frac_()
--
frac__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
frac__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->frac_();
   }|] >> pure self


-- Tensor ger(const Tensor & vec2)
--
ger_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ger_mt self vec2 =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->ger(*$fptr-ptr:(Tensor* vec2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor fft(int64_t signal_ndim, bool normalized)
--
fft_m6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
fft_m6b self signal_ndim normalized =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->fft($(int64_t signal_ndim), $(bool normalized)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor ifft(int64_t signal_ndim, bool normalized)
--
ifft_m6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
ifft_m6b self signal_ndim normalized =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->ifft($(int64_t signal_ndim), $(bool normalized)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rfft(int64_t signal_ndim, bool normalized, bool onesided)
--
rfft_m6bb :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
rfft_m6bb self signal_ndim normalized onesided =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->rfft($(int64_t signal_ndim), $(bool normalized), $(bool onesided)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor irfft(int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes)
--
irfft_m6bba :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> Vector Int64 -> IO (ForeignPtr CTensor)
irfft_m6bba self signal_ndim normalized onesided signal_sizes =  V.unsafeWith signal_sizes $ \signal_sizes__array -> let signal_sizes__size = fromIntegral (V.length signal_sizes) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->irfft($(int64_t signal_ndim), $(bool normalized), $(bool onesided), ArrayRef<int64_t>($(int64_t* signal_sizes__array), $(size_t signal_sizes__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor index(TensorList indices)
--
index_ml :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> IO (ForeignPtr CTensor)
index_ml self indices =  V.unsafeWith indices $ \indices__array -> let indices__size = fromIntegral (V.length indices) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->index(pack_tensor_list($(Tensor** indices__array), $(size_t indices__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & index_copy_(int64_t dim, const Tensor & index, const Tensor & source)
--
index_copy__m6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_copy__m6tt self dim index source =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->index_copy_($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source));
   }|] >> pure self


-- Tensor index_copy(int64_t dim, const Tensor & index, const Tensor & source)
--
index_copy_m6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_copy_m6tt self dim index source =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->index_copy($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & index_put_(TensorList indices, const Tensor & values, bool accumulate)
--
index_put__mltb :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
index_put__mltb self indices values accumulate =  V.unsafeWith indices $ \indices__array -> let indices__size = fromIntegral (V.length indices) in 
  [C.block|void {
    $fptr-ptr:(Tensor *self)->index_put_(pack_tensor_list($(Tensor** indices__array), $(size_t indices__size)), *$fptr-ptr:(Tensor* values), $(bool accumulate));
   }|] >> pure self


-- Tensor index_put(TensorList indices, const Tensor & values, bool accumulate)
--
index_put_mltb :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
index_put_mltb self indices values accumulate =  V.unsafeWith indices $ \indices__array -> let indices__size = fromIntegral (V.length indices) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->index_put(pack_tensor_list($(Tensor** indices__array), $(size_t indices__size)), *$fptr-ptr:(Tensor* values), $(bool accumulate)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor inverse()
--
inverse_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
inverse_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->inverse());
   }|] >>= newForeignPtr deleteTensor


-- Tensor isclose(const Tensor & other, double rtol, double atol, bool equal_nan)
--
isclose_mtddb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> CBool -> IO (ForeignPtr CTensor)
isclose_mtddb self other rtol atol equal_nan =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->isclose(*$fptr-ptr:(Tensor* other), $(double rtol), $(double atol), $(bool equal_nan)));
   }|] >>= newForeignPtr deleteTensor


-- bool is_distributed()
--
is_distributed_m :: ForeignPtr CTensor -> IO (CBool)
is_distributed_m self =  
  [C.block|bool {
    return $fptr-ptr:(Tensor *self)->is_distributed();
   }|]


-- bool is_floating_point()
--
is_floating_point_m :: ForeignPtr CTensor -> IO (CBool)
is_floating_point_m self =  
  [C.block|bool {
    return $fptr-ptr:(Tensor *self)->is_floating_point();
   }|]


-- bool is_complex()
--
is_complex_m :: ForeignPtr CTensor -> IO (CBool)
is_complex_m self =  
  [C.block|bool {
    return $fptr-ptr:(Tensor *self)->is_complex();
   }|]


-- bool is_nonzero()
--
is_nonzero_m :: ForeignPtr CTensor -> IO (CBool)
is_nonzero_m self =  
  [C.block|bool {
    return $fptr-ptr:(Tensor *self)->is_nonzero();
   }|]


-- bool is_same_size(const Tensor & other)
--
is_same_size_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (CBool)
is_same_size_mt self other =  
  [C.block|bool {
    return $fptr-ptr:(Tensor *self)->is_same_size(*$fptr-ptr:(Tensor* other));
   }|]


-- bool is_signed()
--
is_signed_m :: ForeignPtr CTensor -> IO (CBool)
is_signed_m self =  
  [C.block|bool {
    return $fptr-ptr:(Tensor *self)->is_signed();
   }|]


-- std::tuple<Tensor,Tensor> kthvalue(int64_t k, int64_t dim, bool keepdim)
--
kthvalue_m66b :: ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
kthvalue_m66b self k dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->kthvalue($(int64_t k), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- Tensor log()
--
log_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->log());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & log_()
--
log__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->log_();
   }|] >> pure self


-- Tensor log10()
--
log10_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log10_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->log10());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & log10_()
--
log10__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log10__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->log10_();
   }|] >> pure self


-- Tensor log1p()
--
log1p_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log1p_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->log1p());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & log1p_()
--
log1p__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log1p__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->log1p_();
   }|] >> pure self


-- Tensor log2()
--
log2_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log2_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->log2());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & log2_()
--
log2__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log2__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->log2_();
   }|] >> pure self


-- Tensor logdet()
--
logdet_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logdet_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->logdet());
   }|] >>= newForeignPtr deleteTensor


-- Tensor log_softmax(int64_t dim, c10::optional<ScalarType> dtype)
--
log_softmax_m6s :: ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
log_softmax_m6s self dim dtype =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->log_softmax($(int64_t dim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor logsumexp(IntArrayRef dim, bool keepdim)
--
logsumexp_mab :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
logsumexp_mab self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->logsumexp(ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor matmul(const Tensor & other)
--
matmul_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
matmul_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->matmul(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor matrix_power(int64_t n)
--
matrix_power_m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
matrix_power_m6 self n =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->matrix_power($(int64_t n)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> max(int64_t dim, bool keepdim)
--
max_m6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_m6b self dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->max($(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- Tensor max_values(IntArrayRef dim, bool keepdim)
--
max_values_mab :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
max_values_mab self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->max_values(ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mean(c10::optional<ScalarType> dtype)
--
mean_ms :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
mean_ms self dtype =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->mean(static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mean(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype)
--
mean_mabs :: ForeignPtr CTensor -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
mean_mabs self dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->mean(ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> median(int64_t dim, bool keepdim)
--
median_m6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
median_m6b self dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->median($(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> min(int64_t dim, bool keepdim)
--
min_m6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
min_m6b self dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->min($(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- Tensor min_values(IntArrayRef dim, bool keepdim)
--
min_values_mab :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
min_values_mab self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->min_values(ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mm(const Tensor & mat2)
--
mm_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mm_mt self mat2 =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->mm(*$fptr-ptr:(Tensor* mat2)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> mode(int64_t dim, bool keepdim)
--
mode_m6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
mode_m6b self dim keepdim =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->mode($(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


-- Tensor mul(const Tensor & other)
--
mul_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mul_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->mul(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & mul_(const Tensor & other)
--
mul__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mul__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->mul_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor mul(Scalar other)
--
mul_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
mul_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->mul(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & mul_(Scalar other)
--
mul__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
mul__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->mul_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor mv(const Tensor & vec)
--
mv_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mv_mt self vec =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->mv(*$fptr-ptr:(Tensor* vec)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor mvlgamma(int64_t p)
--
mvlgamma_m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
mvlgamma_m6 self p =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->mvlgamma($(int64_t p)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & mvlgamma_(int64_t p)
--
mvlgamma__m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
mvlgamma__m6 self p =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->mvlgamma_($(int64_t p));
   }|] >> pure self


-- Tensor narrow_copy(int64_t dim, int64_t start, int64_t length)
--
narrow_copy_m666 :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
narrow_copy_m666 self dim start length =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->narrow_copy($(int64_t dim), $(int64_t start), $(int64_t length)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor narrow(int64_t dim, int64_t start, int64_t length)
--
narrow_m666 :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
narrow_m666 self dim start length =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->narrow($(int64_t dim), $(int64_t start), $(int64_t length)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor narrow(int64_t dim, const Tensor & start, int64_t length)
--
narrow_m6t6 :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
narrow_m6t6 self dim start length =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->narrow($(int64_t dim), *$fptr-ptr:(Tensor* start), $(int64_t length)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor permute(IntArrayRef dims)
--
permute_ma :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
permute_ma self dims =  V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->permute(ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor numpy_T()
--
numpy_t_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
numpy_t_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->numpy_T());
   }|] >>= newForeignPtr deleteTensor


-- bool is_pinned()
--
is_pinned_m :: ForeignPtr CTensor -> IO (CBool)
is_pinned_m self =  
  [C.block|bool {
    return $fptr-ptr:(Tensor *self)->is_pinned();
   }|]


-- Tensor pin_memory()
--
pin_memory_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pin_memory_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->pin_memory());
   }|] >>= newForeignPtr deleteTensor


-- Tensor pinverse(double rcond)
--
pinverse_md :: ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor)
pinverse_md self rcond =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->pinverse($(double rcond)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor reciprocal()
--
reciprocal_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
reciprocal_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->reciprocal());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & reciprocal_()
--
reciprocal__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
reciprocal__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->reciprocal_();
   }|] >> pure self


-- Tensor neg()
--
neg_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
neg_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->neg());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & neg_()
--
neg__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
neg__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->neg_();
   }|] >> pure self


-- Tensor repeat(IntArrayRef repeats)
--
repeat_ma :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
repeat_ma self repeats =  V.unsafeWith repeats $ \repeats__array -> let repeats__size = fromIntegral (V.length repeats) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->repeat(ArrayRef<int64_t>($(int64_t* repeats__array), $(size_t repeats__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor repeat_interleave(const Tensor & repeats, c10::optional<int64_t> dim)
--
repeat_interleave_mt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Maybe Int64 -> IO (ForeignPtr CTensor)
repeat_interleave_mt6 self repeats dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->repeat_interleave(*$fptr-ptr:(Tensor* repeats), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor repeat_interleave(int64_t repeats, c10::optional<int64_t> dim)
--
repeat_interleave_m66 :: ForeignPtr CTensor -> Int64 -> Maybe Int64 -> IO (ForeignPtr CTensor)
repeat_interleave_m66 self repeats dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->repeat_interleave($(int64_t repeats), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor reshape(IntArrayRef shape)
--
reshape_ma :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
reshape_ma self shape =  V.unsafeWith shape $ \shape__array -> let shape__size = fromIntegral (V.length shape) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->reshape(ArrayRef<int64_t>($(int64_t* shape__array), $(size_t shape__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor reshape_as(const Tensor & other)
--
reshape_as_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
reshape_as_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->reshape_as(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor round()
--
round_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
round_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->round());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & round_()
--
round__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
round__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->round_();
   }|] >> pure self


-- Tensor relu()
--
relu_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
relu_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->relu());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & relu_()
--
relu__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
relu__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->relu_();
   }|] >> pure self


-- Tensor prelu(const Tensor & weight)
--
prelu_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
prelu_mt self weight =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->prelu(*$fptr-ptr:(Tensor* weight)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor hardshrink(Scalar lambd)
--
hardshrink_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
hardshrink_ms self lambd =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->hardshrink(*$fptr-ptr:(Scalar* lambd)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rsqrt()
--
rsqrt_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rsqrt_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->rsqrt());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & rsqrt_()
--
rsqrt__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rsqrt__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->rsqrt_();
   }|] >> pure self


-- Tensor select(int64_t dim, int64_t index)
--
select_m66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
select_m66 self dim index =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->select($(int64_t dim), $(int64_t index)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor sigmoid()
--
sigmoid_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sigmoid_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->sigmoid());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sigmoid_()
--
sigmoid__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sigmoid__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->sigmoid_();
   }|] >> pure self


-- Tensor sin()
--
sin_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sin_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->sin());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sin_()
--
sin__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sin__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->sin_();
   }|] >> pure self


-- Tensor sinh()
--
sinh_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sinh_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->sinh());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sinh_()
--
sinh__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sinh__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->sinh_();
   }|] >> pure self


-- Tensor detach()
--
detach_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
detach_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->detach());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & detach_()
--
detach__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
detach__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->detach_();
   }|] >> pure self


-- int64_t size(int64_t dim)
--
size_m6 :: ForeignPtr CTensor -> Int64 -> IO (Int64)
size_m6 self dim =  
  [C.block|int64_t {
    return $fptr-ptr:(Tensor *self)->size($(int64_t dim));
   }|]


-- Tensor slice(int64_t dim, int64_t start, int64_t end, int64_t step)
--
slice_m6666 :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
slice_m6666 self dim start end step =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->slice($(int64_t dim), $(int64_t start), $(int64_t end), $(int64_t step)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> slogdet()
--
slogdet_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
slogdet_m self =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->slogdet());
   }|] >>= unTupleTensorTensor


-- Tensor smm(const Tensor & mat2)
--
smm_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
smm_mt self mat2 =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->smm(*$fptr-ptr:(Tensor* mat2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor softmax(int64_t dim, c10::optional<ScalarType> dtype)
--
softmax_m6s :: ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
softmax_m6s self dim dtype =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->softmax($(int64_t dim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- std::vector<Tensor> split(int64_t split_size, int64_t dim)
--
split_m66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (Vector (Ptr CTensor))
split_m66 self split_size dim =  
  [C.block|void* {
    return (void*)new std::vector<Tensor>($fptr-ptr:(Tensor *self)->split($(int64_t split_size), $(int64_t dim)));
   }|] >>= unVectorTensor


-- std::vector<Tensor> split_with_sizes(IntArrayRef split_sizes, int64_t dim)
--
split_with_sizes_ma6 :: ForeignPtr CTensor -> Vector Int64 -> Int64 -> IO (Vector (Ptr CTensor))
split_with_sizes_ma6 self split_sizes dim =  V.unsafeWith split_sizes $ \split_sizes__array -> let split_sizes__size = fromIntegral (V.length split_sizes) in 
  [C.block|void* {
    return (void*)new std::vector<Tensor>($fptr-ptr:(Tensor *self)->split_with_sizes(ArrayRef<int64_t>($(int64_t* split_sizes__array), $(size_t split_sizes__size)), $(int64_t dim)));
   }|] >>= unVectorTensor


-- Tensor squeeze()
--
squeeze_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
squeeze_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->squeeze());
   }|] >>= newForeignPtr deleteTensor


-- Tensor squeeze(int64_t dim)
--
squeeze_m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
squeeze_m6 self dim =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->squeeze($(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & squeeze_()
--
squeeze__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
squeeze__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->squeeze_();
   }|] >> pure self


-- Tensor & squeeze_(int64_t dim)
--
squeeze__m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
squeeze__m6 self dim =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->squeeze_($(int64_t dim));
   }|] >> pure self


-- Tensor sspaddmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha)
--
sspaddmm_mttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sspaddmm_mttss self mat1 mat2 beta alpha =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->sspaddmm(*$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor stft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided)
--
stft_m666tbb :: ForeignPtr CTensor -> Int64 -> Maybe Int64 -> Maybe Int64 -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor)
stft_m666tbb self n_fft hop_length win_length window normalized onesided =  let (hop_length__is_present, hop_length__value) = splitMaybe hop_length 0 in let (win_length__is_present, win_length__value) = splitMaybe win_length 0 in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->stft($(int64_t n_fft), ($(bool hop_length__is_present) ? make_optional($(int64_t hop_length__value)) : c10::nullopt), ($(bool win_length__is_present) ? make_optional($(int64_t win_length__value)) : c10::nullopt), *$fptr-ptr:(Tensor* window), $(bool normalized), $(bool onesided)));
   }|] >>= newForeignPtr deleteTensor


-- int64_t stride(int64_t dim)
--
stride_m6 :: ForeignPtr CTensor -> Int64 -> IO (Int64)
stride_m6 self dim =  
  [C.block|int64_t {
    return $fptr-ptr:(Tensor *self)->stride($(int64_t dim));
   }|]


-- Tensor sum(c10::optional<ScalarType> dtype)
--
sum_ms :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
sum_ms self dtype =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->sum(static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor sum(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype)
--
sum_mabs :: ForeignPtr CTensor -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
sum_mabs self dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->sum(ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor sum_to_size(IntArrayRef size)
--
sum_to_size_ma :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
sum_to_size_ma self size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->sum_to_size(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor sqrt()
--
sqrt_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sqrt_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->sqrt());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sqrt_()
--
sqrt__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sqrt__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->sqrt_();
   }|] >> pure self


-- Tensor square()
--
square_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
square_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->square());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & square_()
--
square__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
square__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->square_();
   }|] >> pure self


-- Tensor std(bool unbiased)
--
std_mb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
std_mb self unbiased =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->std($(bool unbiased)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor std(IntArrayRef dim, bool unbiased, bool keepdim)
--
std_mabb :: ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
std_mabb self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->std(ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor prod(c10::optional<ScalarType> dtype)
--
prod_ms :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
prod_ms self dtype =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->prod(static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor prod(int64_t dim, bool keepdim, c10::optional<ScalarType> dtype)
--
prod_m6bs :: ForeignPtr CTensor -> Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
prod_m6bs self dim keepdim dtype =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->prod($(int64_t dim), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor t()
--
t_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
t_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->t());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & t_()
--
t__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
t__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->t_();
   }|] >> pure self


-- Tensor tan()
--
tan_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tan_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->tan());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & tan_()
--
tan__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tan__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->tan_();
   }|] >> pure self


-- Tensor tanh()
--
tanh_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tanh_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->tanh());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & tanh_()
--
tanh__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tanh__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->tanh_();
   }|] >> pure self


-- Tensor transpose(int64_t dim0, int64_t dim1)
--
transpose_m66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
transpose_m66 self dim0 dim1 =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->transpose($(int64_t dim0), $(int64_t dim1)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & transpose_(int64_t dim0, int64_t dim1)
--
transpose__m66 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
transpose__m66 self dim0 dim1 =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->transpose_($(int64_t dim0), $(int64_t dim1));
   }|] >> pure self


-- Tensor flip(IntArrayRef dims)
--
flip_ma :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
flip_ma self dims =  V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->flip(ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor roll(IntArrayRef shifts, IntArrayRef dims)
--
roll_maa :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
roll_maa self shifts dims =  V.unsafeWith shifts $ \shifts__array -> let shifts__size = fromIntegral (V.length shifts) in V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->roll(ArrayRef<int64_t>($(int64_t* shifts__array), $(size_t shifts__size)), ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor rot90(int64_t k, IntArrayRef dims)
--
rot90_m6a :: ForeignPtr CTensor -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
rot90_m6a self k dims =  V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->rot90($(int64_t k), ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor true_divide(const Tensor & other)
--
true_divide_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
true_divide_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->true_divide(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & true_divide_(const Tensor & other)
--
true_divide__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
true_divide__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->true_divide_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor true_divide(Scalar other)
--
true_divide_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
true_divide_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->true_divide(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & true_divide_(Scalar other)
--
true_divide__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
true_divide__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->true_divide_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor trunc()
--
trunc_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
trunc_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->trunc());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & trunc_()
--
trunc__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
trunc__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->trunc_();
   }|] >> pure self


-- Tensor type_as(const Tensor & other)
--
type_as_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
type_as_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->type_as(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor unsqueeze(int64_t dim)
--
unsqueeze_m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
unsqueeze_m6 self dim =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->unsqueeze($(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & unsqueeze_(int64_t dim)
--
unsqueeze__m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
unsqueeze__m6 self dim =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->unsqueeze_($(int64_t dim));
   }|] >> pure self


-- Tensor var(bool unbiased)
--
var_mb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
var_mb self unbiased =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->var($(bool unbiased)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor var(IntArrayRef dim, bool unbiased, bool keepdim)
--
var_mabb :: ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
var_mabb self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->var(ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor view_as(const Tensor & other)
--
view_as_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
view_as_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->view_as(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor where(const Tensor & condition, const Tensor & other)
--
where_mtt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
where_mtt self condition other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->where(*$fptr-ptr:(Tensor* condition), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor norm(c10::optional<Scalar> p, ScalarType dtype)
--
norm_mss :: ForeignPtr CTensor -> ForeignPtr CScalar -> Int8 -> IO (ForeignPtr CTensor)
norm_mss self p dtype =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->norm(*$fptr-ptr:(Scalar* p), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor norm(Scalar p)
--
norm_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
norm_ms self p =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->norm(*$fptr-ptr:(Scalar* p)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype)
--
norm_msabs :: ForeignPtr CTensor -> ForeignPtr CScalar -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
norm_msabs self p dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->norm(*$fptr-ptr:(Scalar* p), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim)
--
norm_msab :: ForeignPtr CTensor -> ForeignPtr CScalar -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
norm_msab self p dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->norm(*$fptr-ptr:(Scalar* p), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor clone(c10::optional<MemoryFormat> memory_format)
--
clone_mm :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
clone_mm self memory_format =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->clone(static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & resize_as_(const Tensor & the_template, c10::optional<MemoryFormat> memory_format)
--
resize_as__mtm :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
resize_as__mtm self the_template memory_format =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->resize_as_(*$fptr-ptr:(Tensor* the_template), static_cast<MemoryFormat>($(int8_t memory_format)));
   }|] >> pure self


-- Tensor pow(Scalar exponent)
--
pow_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
pow_ms self exponent =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->pow(*$fptr-ptr:(Scalar* exponent)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & zero_()
--
zero__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
zero__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->zero_();
   }|] >> pure self


-- Tensor sub(const Tensor & other, Scalar alpha)
--
sub_mts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sub_mts self other alpha =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->sub(*$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sub_(const Tensor & other, Scalar alpha)
--
sub__mts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sub__mts self other alpha =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->sub_(*$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor sub(Scalar other, Scalar alpha)
--
sub_mss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sub_mss self other alpha =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->sub(*$fptr-ptr:(Scalar* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sub_(Scalar other, Scalar alpha)
--
sub__mss :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sub__mss self other alpha =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->sub_(*$fptr-ptr:(Scalar* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha)
--
addmm_mttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmm_mttss self mat1 mat2 beta alpha =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->addmm(*$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & addmm_(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha)
--
addmm__mttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmm__mttss self mat1 mat2 beta alpha =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->addmm_(*$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor & sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim)
--
sparse_resize__ma66 :: ForeignPtr CTensor -> Vector Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
sparse_resize__ma66 self size sparse_dim dense_dim =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    $fptr-ptr:(Tensor *self)->sparse_resize_(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(int64_t sparse_dim), $(int64_t dense_dim));
   }|] >> pure self


-- Tensor & sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim)
--
sparse_resize_and_clear__ma66 :: ForeignPtr CTensor -> Vector Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
sparse_resize_and_clear__ma66 self size sparse_dim dense_dim =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|void {
    $fptr-ptr:(Tensor *self)->sparse_resize_and_clear_(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(int64_t sparse_dim), $(int64_t dense_dim));
   }|] >> pure self


-- Tensor sparse_mask(const Tensor & mask)
--
sparse_mask_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sparse_mask_mt self mask =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->sparse_mask(*$fptr-ptr:(Tensor* mask)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor to_dense()
--
to_dense_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
to_dense_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->to_dense());
   }|] >>= newForeignPtr deleteTensor


-- int64_t sparse_dim()
--
sparse_dim_m :: ForeignPtr CTensor -> IO (Int64)
sparse_dim_m self =  
  [C.block|int64_t {
    return $fptr-ptr:(Tensor *self)->sparse_dim();
   }|]


-- int64_t _dimI()
--
_dimi_m :: ForeignPtr CTensor -> IO (Int64)
_dimi_m self =  
  [C.block|int64_t {
    return $fptr-ptr:(Tensor *self)->_dimI();
   }|]


-- int64_t dense_dim()
--
dense_dim_m :: ForeignPtr CTensor -> IO (Int64)
dense_dim_m self =  
  [C.block|int64_t {
    return $fptr-ptr:(Tensor *self)->dense_dim();
   }|]


-- int64_t _dimV()
--
_dimv_m :: ForeignPtr CTensor -> IO (Int64)
_dimv_m self =  
  [C.block|int64_t {
    return $fptr-ptr:(Tensor *self)->_dimV();
   }|]


-- int64_t _nnz()
--
_nnz_m :: ForeignPtr CTensor -> IO (Int64)
_nnz_m self =  
  [C.block|int64_t {
    return $fptr-ptr:(Tensor *self)->_nnz();
   }|]


-- Tensor coalesce()
--
coalesce_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
coalesce_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->coalesce());
   }|] >>= newForeignPtr deleteTensor


-- bool is_coalesced()
--
is_coalesced_m :: ForeignPtr CTensor -> IO (CBool)
is_coalesced_m self =  
  [C.block|bool {
    return $fptr-ptr:(Tensor *self)->is_coalesced();
   }|]


-- Tensor _indices()
--
_indices_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_indices_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->_indices());
   }|] >>= newForeignPtr deleteTensor


-- Tensor _values()
--
_values_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_values_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->_values());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & _coalesced_(bool coalesced)
--
_coalesced__mb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_coalesced__mb self coalesced =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->_coalesced_($(bool coalesced));
   }|] >> pure self


-- Tensor indices()
--
indices_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
indices_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->indices());
   }|] >>= newForeignPtr deleteTensor


-- Tensor values()
--
values_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
values_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->values());
   }|] >>= newForeignPtr deleteTensor


-- std::vector<Tensor> unbind(int64_t dim)
--
unbind_m6 :: ForeignPtr CTensor -> Int64 -> IO (Vector (Ptr CTensor))
unbind_m6 self dim =  
  [C.block|void* {
    return (void*)new std::vector<Tensor>($fptr-ptr:(Tensor *self)->unbind($(int64_t dim)));
   }|] >>= unVectorTensor


-- Tensor to_sparse(int64_t sparse_dim)
--
to_sparse_m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
to_sparse_m6 self sparse_dim =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->to_sparse($(int64_t sparse_dim)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor to_sparse()
--
to_sparse_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
to_sparse_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->to_sparse());
   }|] >>= newForeignPtr deleteTensor


-- Tensor to_mkldnn()
--
to_mkldnn_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
to_mkldnn_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->to_mkldnn());
   }|] >>= newForeignPtr deleteTensor


-- Tensor dequantize()
--
dequantize_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
dequantize_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->dequantize());
   }|] >>= newForeignPtr deleteTensor


-- double q_scale()
--
q_scale_m :: ForeignPtr CTensor -> IO (CDouble)
q_scale_m self =  
  [C.block|double {
    return $fptr-ptr:(Tensor *self)->q_scale();
   }|]


-- int64_t q_zero_point()
--
q_zero_point_m :: ForeignPtr CTensor -> IO (Int64)
q_zero_point_m self =  
  [C.block|int64_t {
    return $fptr-ptr:(Tensor *self)->q_zero_point();
   }|]


-- Tensor q_per_channel_scales()
--
q_per_channel_scales_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
q_per_channel_scales_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->q_per_channel_scales());
   }|] >>= newForeignPtr deleteTensor


-- Tensor q_per_channel_zero_points()
--
q_per_channel_zero_points_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
q_per_channel_zero_points_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->q_per_channel_zero_points());
   }|] >>= newForeignPtr deleteTensor


-- int64_t q_per_channel_axis()
--
q_per_channel_axis_m :: ForeignPtr CTensor -> IO (Int64)
q_per_channel_axis_m self =  
  [C.block|int64_t {
    return $fptr-ptr:(Tensor *self)->q_per_channel_axis();
   }|]


-- Tensor int_repr()
--
int_repr_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
int_repr_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->int_repr());
   }|] >>= newForeignPtr deleteTensor


-- QScheme qscheme()
--
qscheme_m :: ForeignPtr CTensor -> IO (Word8)
qscheme_m self =  
  [C.block|uint8_t {
    return static_cast<uint8_t>($fptr-ptr:(Tensor *self)->qscheme());
   }|]


-- Tensor to(const TensorOptions & options, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format)
--
to_mobbm :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> CBool -> CBool -> Int8 -> IO (ForeignPtr CTensor)
to_mobbm self options non_blocking copy memory_format =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->to(*$fptr-ptr:(TensorOptions* options), $(bool non_blocking), $(bool copy), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor to(Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format)
--
to_mdsbbm :: ForeignPtr CTensor -> Ptr CDevice -> Int8 -> CBool -> CBool -> Int8 -> IO (ForeignPtr CTensor)
to_mdsbbm self device dtype non_blocking copy memory_format =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->to(*$(Device* device), static_cast<ScalarType>($(int8_t dtype)), $(bool non_blocking), $(bool copy), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor to(ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format)
--
to_msbbm :: ForeignPtr CTensor -> Int8 -> CBool -> CBool -> Int8 -> IO (ForeignPtr CTensor)
to_msbbm self dtype non_blocking copy memory_format =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->to(static_cast<ScalarType>($(int8_t dtype)), $(bool non_blocking), $(bool copy), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor to(const Tensor & other, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format)
--
to_mtbbm :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> Int8 -> IO (ForeignPtr CTensor)
to_mtbbm self other non_blocking copy memory_format =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->to(*$fptr-ptr:(Tensor* other), $(bool non_blocking), $(bool copy), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


-- Scalar item()
--
item_m :: ForeignPtr CTensor -> IO (ForeignPtr CScalar)
item_m self =  
  [C.block|Scalar* {
    return new Scalar($fptr-ptr:(Tensor *self)->item());
   }|] >>= newForeignPtr deleteScalar'


-- Tensor & set_(Storage source)
--
set__ms :: ForeignPtr CTensor -> Ptr CStorage -> IO (ForeignPtr CTensor)
set__ms self source =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->set_(*$(Storage* source));
   }|] >> pure self


-- Tensor & set_(Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride)
--
set__ms6aa :: ForeignPtr CTensor -> Ptr CStorage -> Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
set__ms6aa self source storage_offset size stride =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in 
  [C.block|void {
    $fptr-ptr:(Tensor *self)->set_(*$(Storage* source), $(int64_t storage_offset), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)));
   }|] >> pure self


-- Tensor & set_(const Tensor & source)
--
set__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
set__mt self source =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->set_(*$fptr-ptr:(Tensor* source));
   }|] >> pure self


-- Tensor & set_()
--
set__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
set__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->set_();
   }|] >> pure self


-- bool is_set_to(const Tensor & tensor)
--
is_set_to_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (CBool)
is_set_to_mt self tensor =  
  [C.block|bool {
    return $fptr-ptr:(Tensor *self)->is_set_to(*$fptr-ptr:(Tensor* tensor));
   }|]


-- Tensor & masked_fill_(const Tensor & mask, Scalar value)
--
masked_fill__mts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
masked_fill__mts self mask value =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->masked_fill_(*$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


-- Tensor masked_fill(const Tensor & mask, Scalar value)
--
masked_fill_mts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
masked_fill_mts self mask value =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->masked_fill(*$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & masked_fill_(const Tensor & mask, const Tensor & value)
--
masked_fill__mtt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_fill__mtt self mask value =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->masked_fill_(*$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Tensor* value));
   }|] >> pure self


-- Tensor masked_fill(const Tensor & mask, const Tensor & value)
--
masked_fill_mtt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_fill_mtt self mask value =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->masked_fill(*$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Tensor* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & masked_scatter_(const Tensor & mask, const Tensor & source)
--
masked_scatter__mtt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_scatter__mtt self mask source =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->masked_scatter_(*$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Tensor* source));
   }|] >> pure self


-- Tensor masked_scatter(const Tensor & mask, const Tensor & source)
--
masked_scatter_mtt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_scatter_mtt self mask source =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->masked_scatter(*$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Tensor* source)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor view(IntArrayRef size)
--
view_ma :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
view_ma self size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->view(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size))));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & put_(const Tensor & index, const Tensor & source, bool accumulate)
--
put__mttb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
put__mttb self index source accumulate =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->put_(*$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source), $(bool accumulate));
   }|] >> pure self


-- Tensor & index_add_(int64_t dim, const Tensor & index, const Tensor & source)
--
index_add__m6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_add__m6tt self dim index source =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->index_add_($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source));
   }|] >> pure self


-- Tensor index_add(int64_t dim, const Tensor & index, const Tensor & source)
--
index_add_m6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_add_m6tt self dim index source =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->index_add($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & index_fill_(int64_t dim, const Tensor & index, Scalar value)
--
index_fill__m6ts :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
index_fill__m6ts self dim index value =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->index_fill_($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


-- Tensor index_fill(int64_t dim, const Tensor & index, Scalar value)
--
index_fill_m6ts :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
index_fill_m6ts self dim index value =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->index_fill($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & index_fill_(int64_t dim, const Tensor & index, const Tensor & value)
--
index_fill__m6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_fill__m6tt self dim index value =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->index_fill_($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* value));
   }|] >> pure self


-- Tensor index_fill(int64_t dim, const Tensor & index, const Tensor & value)
--
index_fill_m6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_fill_m6tt self dim index value =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->index_fill($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & scatter_(int64_t dim, const Tensor & index, const Tensor & src)
--
scatter__m6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
scatter__m6tt self dim index src =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->scatter_($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* src));
   }|] >> pure self


-- Tensor scatter(int64_t dim, const Tensor & index, const Tensor & src)
--
scatter_m6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
scatter_m6tt self dim index src =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->scatter($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* src)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & scatter_(int64_t dim, const Tensor & index, Scalar value)
--
scatter__m6ts :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
scatter__m6ts self dim index value =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->scatter_($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


-- Tensor scatter(int64_t dim, const Tensor & index, Scalar value)
--
scatter_m6ts :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
scatter_m6ts self dim index value =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->scatter($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & scatter_add_(int64_t dim, const Tensor & index, const Tensor & src)
--
scatter_add__m6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
scatter_add__m6tt self dim index src =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->scatter_add_($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* src));
   }|] >> pure self


-- Tensor scatter_add(int64_t dim, const Tensor & index, const Tensor & src)
--
scatter_add_m6tt :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
scatter_add_m6tt self dim index src =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->scatter_add($(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* src)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & lt_(Scalar other)
--
lt__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lt__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->lt_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & lt_(const Tensor & other)
--
lt__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lt__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->lt_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & gt_(Scalar other)
--
gt__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
gt__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->gt_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & gt_(const Tensor & other)
--
gt__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
gt__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->gt_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & le_(Scalar other)
--
le__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
le__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->le_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & le_(const Tensor & other)
--
le__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
le__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->le_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & ge_(Scalar other)
--
ge__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ge__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->ge_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & ge_(const Tensor & other)
--
ge__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ge__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->ge_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & eq_(Scalar other)
--
eq__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
eq__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->eq_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & eq_(const Tensor & other)
--
eq__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
eq__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->eq_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & ne_(Scalar other)
--
ne__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ne__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->ne_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & ne_(const Tensor & other)
--
ne__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ne__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->ne_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor bitwise_and(Scalar other)
--
bitwise_and_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
bitwise_and_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->bitwise_and(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bitwise_and(const Tensor & other)
--
bitwise_and_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_and_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->bitwise_and(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & bitwise_and_(Scalar other)
--
bitwise_and__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
bitwise_and__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->bitwise_and_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & bitwise_and_(const Tensor & other)
--
bitwise_and__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_and__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->bitwise_and_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor __and__(Scalar other)
--
__and___ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__and___ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__and__(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor __and__(const Tensor & other)
--
__and___mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__and___mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__and__(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & __iand__(Scalar other)
--
__iand___ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__iand___ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__iand__(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & __iand__(const Tensor & other)
--
__iand___mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__iand___mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__iand__(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor bitwise_or(Scalar other)
--
bitwise_or_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
bitwise_or_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->bitwise_or(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bitwise_or(const Tensor & other)
--
bitwise_or_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_or_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->bitwise_or(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & bitwise_or_(Scalar other)
--
bitwise_or__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
bitwise_or__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->bitwise_or_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & bitwise_or_(const Tensor & other)
--
bitwise_or__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_or__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->bitwise_or_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor __or__(Scalar other)
--
__or___ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__or___ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__or__(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor __or__(const Tensor & other)
--
__or___mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__or___mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__or__(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & __ior__(Scalar other)
--
__ior___ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__ior___ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__ior__(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & __ior__(const Tensor & other)
--
__ior___mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__ior___mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__ior__(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor bitwise_xor(Scalar other)
--
bitwise_xor_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
bitwise_xor_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->bitwise_xor(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor bitwise_xor(const Tensor & other)
--
bitwise_xor_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_xor_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->bitwise_xor(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & bitwise_xor_(Scalar other)
--
bitwise_xor__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
bitwise_xor__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->bitwise_xor_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & bitwise_xor_(const Tensor & other)
--
bitwise_xor__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_xor__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->bitwise_xor_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor __xor__(Scalar other)
--
__xor___ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__xor___ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__xor__(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor __xor__(const Tensor & other)
--
__xor___mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__xor___mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__xor__(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & __ixor__(Scalar other)
--
__ixor___ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__ixor___ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__ixor__(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & __ixor__(const Tensor & other)
--
__ixor___mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__ixor___mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__ixor__(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor __lshift__(Scalar other)
--
__lshift___ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__lshift___ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__lshift__(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor __lshift__(const Tensor & other)
--
__lshift___mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__lshift___mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__lshift__(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & __ilshift__(Scalar other)
--
__ilshift___ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__ilshift___ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__ilshift__(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & __ilshift__(const Tensor & other)
--
__ilshift___mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__ilshift___mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__ilshift__(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor __rshift__(Scalar other)
--
__rshift___ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__rshift___ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__rshift__(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor __rshift__(const Tensor & other)
--
__rshift___mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__rshift___mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__rshift__(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & __irshift__(Scalar other)
--
__irshift___ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__irshift___ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__irshift__(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & __irshift__(const Tensor & other)
--
__irshift___mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__irshift___mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->__irshift__(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & lgamma_()
--
lgamma__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lgamma__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->lgamma_();
   }|] >> pure self


-- Tensor & atan2_(const Tensor & other)
--
atan2__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan2__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->atan2_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & tril_(int64_t diagonal)
--
tril__m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
tril__m6 self diagonal =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->tril_($(int64_t diagonal));
   }|] >> pure self


-- Tensor & triu_(int64_t diagonal)
--
triu__m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
triu__m6 self diagonal =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->triu_($(int64_t diagonal));
   }|] >> pure self


-- Tensor & digamma_()
--
digamma__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
digamma__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->digamma_();
   }|] >> pure self


-- Tensor & polygamma_(int64_t n)
--
polygamma__m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
polygamma__m6 self n =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->polygamma_($(int64_t n));
   }|] >> pure self


-- Tensor & renorm_(Scalar p, int64_t dim, Scalar maxnorm)
--
renorm__ms6s :: ForeignPtr CTensor -> ForeignPtr CScalar -> Int64 -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
renorm__ms6s self p dim maxnorm =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->renorm_(*$fptr-ptr:(Scalar* p), $(int64_t dim), *$fptr-ptr:(Scalar* maxnorm));
   }|] >> pure self


-- Tensor & pow_(Scalar exponent)
--
pow__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
pow__ms self exponent =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->pow_(*$fptr-ptr:(Scalar* exponent));
   }|] >> pure self


-- Tensor & pow_(const Tensor & exponent)
--
pow__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pow__mt self exponent =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->pow_(*$fptr-ptr:(Tensor* exponent));
   }|] >> pure self


-- Tensor & lerp_(const Tensor & end, Scalar weight)
--
lerp__mts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lerp__mts self end weight =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->lerp_(*$fptr-ptr:(Tensor* end), *$fptr-ptr:(Scalar* weight));
   }|] >> pure self


-- Tensor & lerp_(const Tensor & end, const Tensor & weight)
--
lerp__mtt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lerp__mtt self end weight =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->lerp_(*$fptr-ptr:(Tensor* end), *$fptr-ptr:(Tensor* weight));
   }|] >> pure self


-- Tensor & fmod_(Scalar other)
--
fmod__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
fmod__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->fmod_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & fmod_(const Tensor & other)
--
fmod__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fmod__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->fmod_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & remainder_(Scalar other)
--
remainder__ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
remainder__ms self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->remainder_(*$fptr-ptr:(Scalar* other));
   }|] >> pure self


-- Tensor & remainder_(const Tensor & other)
--
remainder__mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
remainder__mt self other =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->remainder_(*$fptr-ptr:(Tensor* other));
   }|] >> pure self


-- Tensor & addbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha)
--
addbmm__mttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addbmm__mttss self batch1 batch2 beta alpha =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->addbmm_(*$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


-- Tensor addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha)
--
addbmm_mttss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addbmm_mttss self batch1 batch2 beta alpha =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->addbmm(*$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & addcdiv_(const Tensor & tensor1, const Tensor & tensor2, Scalar value)
--
addcdiv__mtts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcdiv__mtts self tensor1 tensor2 value =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->addcdiv_(*$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


-- Tensor & random_(int64_t from, c10::optional<int64_t> to, Generator * generator)
--
random__m66g :: ForeignPtr CTensor -> Int64 -> Maybe Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
random__m66g self from to generator =  let (to__is_present, to__value) = splitMaybe to 0 in 
  [C.block|void {
    $fptr-ptr:(Tensor *self)->random_($(int64_t from), ($(bool to__is_present) ? make_optional($(int64_t to__value)) : c10::nullopt), $(Generator* generator));
   }|] >> pure self


-- Tensor & random_(int64_t to, Generator * generator)
--
random__m6g :: ForeignPtr CTensor -> Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
random__m6g self to generator =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->random_($(int64_t to), $(Generator* generator));
   }|] >> pure self


-- Tensor & random_(Generator * generator)
--
random__mg :: ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
random__mg self generator =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->random_($(Generator* generator));
   }|] >> pure self


-- Tensor & uniform_(double from, double to, Generator * generator)
--
uniform__mddg :: ForeignPtr CTensor -> CDouble -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
uniform__mddg self from to generator =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->uniform_($(double from), $(double to), $(Generator* generator));
   }|] >> pure self


-- Tensor & cauchy_(double median, double sigma, Generator * generator)
--
cauchy__mddg :: ForeignPtr CTensor -> CDouble -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
cauchy__mddg self median sigma generator =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->cauchy_($(double median), $(double sigma), $(Generator* generator));
   }|] >> pure self


-- Tensor & log_normal_(double mean, double std, Generator * generator)
--
log_normal__mddg :: ForeignPtr CTensor -> CDouble -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
log_normal__mddg self mean std generator =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->log_normal_($(double mean), $(double std), $(Generator* generator));
   }|] >> pure self


-- Tensor & exponential_(double lambd, Generator * generator)
--
exponential__mdg :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
exponential__mdg self lambd generator =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->exponential_($(double lambd), $(Generator* generator));
   }|] >> pure self


-- Tensor & geometric_(double p, Generator * generator)
--
geometric__mdg :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
geometric__mdg self p generator =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->geometric_($(double p), $(Generator* generator));
   }|] >> pure self


-- Tensor diag(int64_t diagonal)
--
diag_m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
diag_m6 self diagonal =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->diag($(int64_t diagonal)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor cross(const Tensor & other, c10::optional<int64_t> dim)
--
cross_mt6 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Maybe Int64 -> IO (ForeignPtr CTensor)
cross_mt6 self other dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in 
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->cross(*$fptr-ptr:(Tensor* other), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor triu(int64_t diagonal)
--
triu_m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
triu_m6 self diagonal =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->triu($(int64_t diagonal)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor tril(int64_t diagonal)
--
tril_m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
tril_m6 self diagonal =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->tril($(int64_t diagonal)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor trace()
--
trace_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
trace_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->trace());
   }|] >>= newForeignPtr deleteTensor


-- Tensor ne(Scalar other)
--
ne_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ne_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->ne(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor ne(const Tensor & other)
--
ne_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ne_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->ne(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor eq(Scalar other)
--
eq_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
eq_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->eq(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor eq(const Tensor & other)
--
eq_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
eq_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->eq(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor ge(Scalar other)
--
ge_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ge_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->ge(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor ge(const Tensor & other)
--
ge_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ge_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->ge(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor le(Scalar other)
--
le_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
le_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->le(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor le(const Tensor & other)
--
le_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
le_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->le(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor gt(Scalar other)
--
gt_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
gt_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->gt(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor gt(const Tensor & other)
--
gt_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
gt_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->gt(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor lt(Scalar other)
--
lt_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lt_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->lt(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor lt(const Tensor & other)
--
lt_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lt_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->lt(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor take(const Tensor & index)
--
take_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
take_mt self index =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->take(*$fptr-ptr:(Tensor* index)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor index_select(int64_t dim, const Tensor & index)
--
index_select_m6t :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_select_m6t self dim index =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->index_select($(int64_t dim), *$fptr-ptr:(Tensor* index)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor masked_select(const Tensor & mask)
--
masked_select_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_select_mt self mask =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->masked_select(*$fptr-ptr:(Tensor* mask)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor nonzero()
--
nonzero_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
nonzero_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->nonzero());
   }|] >>= newForeignPtr deleteTensor


-- std::vector<Tensor> nonzero_numpy()
--
nonzero_numpy_m :: ForeignPtr CTensor -> IO (Vector (Ptr CTensor))
nonzero_numpy_m self =  
  [C.block|void* {
    return (void*)new std::vector<Tensor>($fptr-ptr:(Tensor *self)->nonzero_numpy());
   }|] >>= unVectorTensor


-- Tensor gather(int64_t dim, const Tensor & index, bool sparse_grad)
--
gather_m6tb :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
gather_m6tb self dim index sparse_grad =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->gather($(int64_t dim), *$fptr-ptr:(Tensor* index), $(bool sparse_grad)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value)
--
addcmul_mtts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcmul_mtts self tensor1 tensor2 value =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->addcmul(*$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & addcmul_(const Tensor & tensor1, const Tensor & tensor2, Scalar value)
--
addcmul__mtts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcmul__mtts self tensor1 tensor2 value =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->addcmul_(*$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


-- Tensor addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value)
--
addcdiv_mtts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcdiv_mtts self tensor1 tensor2 value =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->addcdiv(*$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> lstsq(const Tensor & A)
--
lstsq_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
lstsq_mt self a =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->lstsq(*$fptr-ptr:(Tensor* a)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> triangular_solve(const Tensor & A, bool upper, bool transpose, bool unitriangular)
--
triangular_solve_mtbbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
triangular_solve_mtbbb self a upper transpose unitriangular =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->triangular_solve(*$fptr-ptr:(Tensor* a), $(bool upper), $(bool transpose), $(bool unitriangular)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> symeig(bool eigenvectors, bool upper)
--
symeig_mbb :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
symeig_mbb self eigenvectors upper =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->symeig($(bool eigenvectors), $(bool upper)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> eig(bool eigenvectors)
--
eig_mb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
eig_mb self eigenvectors =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->eig($(bool eigenvectors)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor,Tensor> svd(bool some, bool compute_uv)
--
svd_mbb :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
svd_mbb self some compute_uv =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>($fptr-ptr:(Tensor *self)->svd($(bool some), $(bool compute_uv)));
   }|] >>= unTupleTensorTensorTensor


-- Tensor cholesky(bool upper)
--
cholesky_mb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_mb self upper =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->cholesky($(bool upper)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor cholesky_solve(const Tensor & input2, bool upper)
--
cholesky_solve_mtb :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_solve_mtb self input2 upper =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->cholesky_solve(*$fptr-ptr:(Tensor* input2), $(bool upper)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> solve(const Tensor & A)
--
solve_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
solve_mt self a =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->solve(*$fptr-ptr:(Tensor* a)));
   }|] >>= unTupleTensorTensor


-- Tensor cholesky_inverse(bool upper)
--
cholesky_inverse_mb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_inverse_mb self upper =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->cholesky_inverse($(bool upper)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> qr(bool some)
--
qr_mb :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
qr_mb self some =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->qr($(bool some)));
   }|] >>= unTupleTensorTensor


-- std::tuple<Tensor,Tensor> geqrf()
--
geqrf_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
geqrf_m self =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->geqrf());
   }|] >>= unTupleTensorTensor


-- Tensor orgqr(const Tensor & input2)
--
orgqr_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
orgqr_mt self input2 =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->orgqr(*$fptr-ptr:(Tensor* input2)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose)
--
ormqr_mttbb :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor)
ormqr_mttbb self input2 input3 left transpose =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->ormqr(*$fptr-ptr:(Tensor* input2), *$fptr-ptr:(Tensor* input3), $(bool left), $(bool transpose)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor lu_solve(const Tensor & LU_data, const Tensor & LU_pivots)
--
lu_solve_mtt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lu_solve_mtt self lu_data lu_pivots =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->lu_solve(*$fptr-ptr:(Tensor* lu_data), *$fptr-ptr:(Tensor* lu_pivots)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor multinomial(int64_t num_samples, bool replacement, Generator * generator)
--
multinomial_m6bg :: ForeignPtr CTensor -> Int64 -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
multinomial_m6bg self num_samples replacement generator =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->multinomial($(int64_t num_samples), $(bool replacement), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor lgamma()
--
lgamma_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lgamma_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->lgamma());
   }|] >>= newForeignPtr deleteTensor


-- Tensor digamma()
--
digamma_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
digamma_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->digamma());
   }|] >>= newForeignPtr deleteTensor


-- Tensor polygamma(int64_t n)
--
polygamma_m6 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
polygamma_m6 self n =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->polygamma($(int64_t n)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor erfinv()
--
erfinv_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfinv_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->erfinv());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & erfinv_()
--
erfinv__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfinv__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->erfinv_();
   }|] >> pure self


-- Tensor sign()
--
sign_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sign_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->sign());
   }|] >>= newForeignPtr deleteTensor


-- Tensor & sign_()
--
sign__m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sign__m self =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->sign_();
   }|] >> pure self


-- Tensor dist(const Tensor & other, Scalar p)
--
dist_mts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
dist_mts self other p =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->dist(*$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* p)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor atan2(const Tensor & other)
--
atan2_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan2_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->atan2(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor lerp(const Tensor & end, Scalar weight)
--
lerp_mts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lerp_mts self end weight =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->lerp(*$fptr-ptr:(Tensor* end), *$fptr-ptr:(Scalar* weight)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor lerp(const Tensor & end, const Tensor & weight)
--
lerp_mtt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lerp_mtt self end weight =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->lerp(*$fptr-ptr:(Tensor* end), *$fptr-ptr:(Tensor* weight)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor histc(int64_t bins, Scalar min, Scalar max)
--
histc_m6ss :: ForeignPtr CTensor -> Int64 -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
histc_m6ss self bins min max =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->histc($(int64_t bins), *$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor fmod(Scalar other)
--
fmod_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
fmod_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->fmod(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor fmod(const Tensor & other)
--
fmod_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fmod_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->fmod(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor remainder(Scalar other)
--
remainder_ms :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
remainder_ms self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->remainder(*$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor remainder(const Tensor & other)
--
remainder_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
remainder_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->remainder(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor min(const Tensor & other)
--
min_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
min_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->min(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor min()
--
min_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
min_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->min());
   }|] >>= newForeignPtr deleteTensor


-- Tensor max(const Tensor & other)
--
max_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
max_mt self other =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->max(*$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor max()
--
max_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
max_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->max());
   }|] >>= newForeignPtr deleteTensor


-- Tensor median()
--
median_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
median_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->median());
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> sort(int64_t dim, bool descending)
--
sort_m6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
sort_m6b self dim descending =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->sort($(int64_t dim), $(bool descending)));
   }|] >>= unTupleTensorTensor


-- Tensor argsort(int64_t dim, bool descending)
--
argsort_m6b :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
argsort_m6b self dim descending =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->argsort($(int64_t dim), $(bool descending)));
   }|] >>= newForeignPtr deleteTensor


-- std::tuple<Tensor,Tensor> topk(int64_t k, int64_t dim, bool largest, bool sorted)
--
topk_m66bb :: ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
topk_m66bb self k dim largest sorted =  
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>($fptr-ptr:(Tensor *self)->topk($(int64_t k), $(int64_t dim), $(bool largest), $(bool sorted)));
   }|] >>= unTupleTensorTensor


-- Tensor all()
--
all_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
all_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->all());
   }|] >>= newForeignPtr deleteTensor


-- Tensor any()
--
any_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
any_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->any());
   }|] >>= newForeignPtr deleteTensor


-- Tensor renorm(Scalar p, int64_t dim, Scalar maxnorm)
--
renorm_ms6s :: ForeignPtr CTensor -> ForeignPtr CScalar -> Int64 -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
renorm_ms6s self p dim maxnorm =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->renorm(*$fptr-ptr:(Scalar* p), $(int64_t dim), *$fptr-ptr:(Scalar* maxnorm)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor unfold(int64_t dimension, int64_t size, int64_t step)
--
unfold_m666 :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
unfold_m666 self dimension size step =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->unfold($(int64_t dimension), $(int64_t size), $(int64_t step)));
   }|] >>= newForeignPtr deleteTensor


-- bool equal(const Tensor & other)
--
equal_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (CBool)
equal_mt self other =  
  [C.block|bool {
    return $fptr-ptr:(Tensor *self)->equal(*$fptr-ptr:(Tensor* other));
   }|]


-- Tensor pow(const Tensor & exponent)
--
pow_mt :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pow_mt self exponent =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->pow(*$fptr-ptr:(Tensor* exponent)));
   }|] >>= newForeignPtr deleteTensor


-- Tensor & normal_(double mean, double std, Generator * generator)
--
normal__mddg :: ForeignPtr CTensor -> CDouble -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal__mddg self mean std generator =  
  [C.block|void {
    $fptr-ptr:(Tensor *self)->normal_($(double mean), $(double std), $(Generator* generator));
   }|] >> pure self


-- Tensor alias()
--
alias_m :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
alias_m self =  
  [C.block|Tensor* {
    return new Tensor($fptr-ptr:(Tensor *self)->alias());
   }|] >>= newForeignPtr deleteTensor

