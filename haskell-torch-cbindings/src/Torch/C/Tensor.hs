{-# LANGUAGE FlexibleInstances, OverloadedStrings, QuasiQuotes, ScopedTypeVariables, TemplateHaskell #-}

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

C.include "<ATen/ArrayRef.h>"
C.include "<torch/csrc/autograd/generated/VariableType.h>"

C.using "namespace at"
C.using "namespace torch::autograd"

C.verbatim "using edge_list = std::vector<torch::autograd::Edge>;"

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
getType t = toEnum . fromIntegral <$> [C.exp|int{(int) $fptr-ptr:(Tensor *t)->type().scalarType() }|]

getTypeString t = [C.exp|char*{(char*) $fptr-ptr:(Tensor *t)->type().toString().c_str() }|]

is_contiguous t = [C.exp|bool{$fptr-ptr:(Tensor *t)->is_contiguous() }|]

is_sparse t = [C.exp|bool{$fptr-ptr:(Tensor *t)->is_sparse() }|]

backend t = [C.exp|int{(int)$fptr-ptr:(Tensor *t)->options().backend() }|]

device t = [C.exp|int{(int)$fptr-ptr:(Tensor *t)->options().device().type() }|]

debugPrintCType t = [C.exp|void{std::cout << typeid((void*)$fptr-ptr:(Tensor *t)).name() << '\n';}|]

C.verbatim "std::vector<Tensor> pack_tensor_list(Tensor** arr, size_t len) { std::vector<Tensor> v; for(size_t i = 0; i < len; i++) { v.push_back(*(arr[i])); }; return v; }"

C.verbatim "std::array<bool,2> make_array_bool_2(bool *arr) { return std::array<bool,2>{arr[0], arr[1]}; }"
C.verbatim "std::array<bool,3> make_array_bool_3(bool *arr) { return std::array<bool,3>{arr[0], arr[1], arr[2]}; }"
C.verbatim "std::array<bool,4> make_array_bool_4(bool *arr) { return std::array<bool,4>{arr[0], arr[1], arr[2], arr[3]}; }"

C.verbatim "extern \"C\" void delete_tensor(Tensor* o) { delete o; }"
C.verbatim "extern \"C\" void delete_tensor_storage(Tensor* o) { free(o->data_ptr()); }"
C.verbatim "extern \"C\" void delete_tensor_options(TensorOptions* o) { delete(o); }"

foreign import ccall "&delete_tensor" deleteTensor :: FunPtr (Ptr CTensor -> IO ())
foreign import ccall "&delete_tensor_storage" deleteTensorStorage :: FunPtr (Ptr CTensor -> IO ())
foreign import ccall "&delete_tensor_options" deleteTensorOptions :: FunPtr (Ptr CTensorOptions -> IO ())

-- TODO We should just not export this, but we're not at the stage where we
-- handle export lists yet.
foreign import ccall "&delete_scalar" deleteScalar' :: FunPtr (Ptr CScalar -> IO ())

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
                           .is_variable(true)
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
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-- Everything below is AUTOGENERATED from generate-ctensor
--  __and__ __and__
--
__and__ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__and__ self other =
  [C.block|void {
    VariableType::__and__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  __and__ __and____1
--
__and____1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__and____1 self other =
  [C.block|void {
    VariableType::__and__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  __iand__ __iand__
--
__iand__ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__iand__ self other =
  [C.block|void {
    VariableType::__iand__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  __iand__ __iand____1
--
__iand____1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__iand____1 self other =
  [C.block|void {
    VariableType::__iand__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  __ilshift__ __ilshift__
--
__ilshift__ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__ilshift__ self other =
  [C.block|void {
    VariableType::__ilshift__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  __ilshift__ __ilshift____1
--
__ilshift____1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__ilshift____1 self other =
  [C.block|void {
    VariableType::__ilshift__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  __ior__ __ior__
--
__ior__ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__ior__ self other =
  [C.block|void {
    VariableType::__ior__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  __ior__ __ior____1
--
__ior____1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__ior____1 self other =
  [C.block|void {
    VariableType::__ior__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  __irshift__ __irshift__
--
__irshift__ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__irshift__ self other =
  [C.block|void {
    VariableType::__irshift__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  __irshift__ __irshift____1
--
__irshift____1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__irshift____1 self other =
  [C.block|void {
    VariableType::__irshift__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  __ixor__ __ixor__
--
__ixor__ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__ixor__ self other =
  [C.block|void {
    VariableType::__ixor__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  __ixor__ __ixor____1
--
__ixor____1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__ixor____1 self other =
  [C.block|void {
    VariableType::__ixor__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  __lshift__ __lshift__
--
__lshift__ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__lshift__ self other =
  [C.block|void {
    VariableType::__lshift__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  __lshift__ __lshift____1
--
__lshift____1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__lshift____1 self other =
  [C.block|void {
    VariableType::__lshift__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  __or__ __or__
--
__or__ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__or__ self other =
  [C.block|void {
    VariableType::__or__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  __or__ __or____1
--
__or____1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__or____1 self other =
  [C.block|void {
    VariableType::__or__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  __rshift__ __rshift__
--
__rshift__ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__rshift__ self other =
  [C.block|void {
    VariableType::__rshift__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  __rshift__ __rshift____1
--
__rshift____1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__rshift____1 self other =
  [C.block|void {
    VariableType::__rshift__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  __xor__ __xor__
--
__xor__ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
__xor__ self other =
  [C.block|void {
    VariableType::__xor__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  __xor__ __xor____1
--
__xor____1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
__xor____1 self other =
  [C.block|void {
    VariableType::__xor__(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  _adaptive_avg_pool2d _adaptive_avg_pool2d
--
_adaptive_avg_pool2d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
_adaptive_avg_pool2d self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_adaptive_avg_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


--  _addmm _addmm
--
_addmm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_addmm self mat1 mat2 beta alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::_addmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  _addmm_ _addmm_
--
_addmm_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_addmm_ self mat1 mat2 beta alpha =
  [C.block|void {
    VariableType::_addmm_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  _addmm_out _addmm_out
--
_addmm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_addmm_out out self mat1 mat2 beta alpha =
  [C.block|void {
    VariableType::_addmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  _addr _addr
--
_addr :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_addr self vec1 vec2 beta alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::_addr(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  _addr_ _addr_
--
_addr_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_addr_ self vec1 vec2 beta alpha =
  [C.block|void {
    VariableType::_addr_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  _addr_out _addr_out
--
_addr_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_addr_out out self vec1 vec2 beta alpha =
  [C.block|void {
    VariableType::_addr_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  _baddbmm_mkl_ _baddbmm_mkl_
--
_baddbmm_mkl_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_baddbmm_mkl_ self batch1 batch2 beta alpha =
  [C.block|void {
    VariableType::_baddbmm_mkl_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  _batch_norm_impl_index _batch_norm_impl_index
--
_batch_norm_impl_index :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, Int64)
_batch_norm_impl_index input weight bias running_mean running_var training momentum eps cudnn_enabled =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,int64_t>(VariableType::_batch_norm_impl_index(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool training), $(double momentum), $(double eps), $(bool cudnn_enabled)));
   }|] >>= unTupleTensorTensorTensorInt64


--  _cast_Byte _cast_byte
--
_cast_byte :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_byte self non_blocking =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cast_Byte(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


--  _cast_Char _cast_char
--
_cast_char :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_char self non_blocking =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cast_Char(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


--  _cast_Double _cast_double
--
_cast_double :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_double self non_blocking =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cast_Double(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


--  _cast_Float _cast_float
--
_cast_float :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_float self non_blocking =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cast_Float(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


--  _cast_Half _cast_half
--
_cast_half :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_half self non_blocking =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cast_Half(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


--  _cast_Int _cast_int
--
_cast_int :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_int self non_blocking =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cast_Int(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


--  _cast_Long _cast_long
--
_cast_long :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_long self non_blocking =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cast_Long(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


--  _cast_Short _cast_short
--
_cast_short :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cast_short self non_blocking =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cast_Short(*$fptr-ptr:(Tensor* self), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


--  _cat _cat
--
_cat :: Vector (Ptr CTensor) -> Int64 -> IO (ForeignPtr CTensor)
_cat tensors dim =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_cat(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size)), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  _cat_out _cat_out
--
_cat_out :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Int64 -> IO (ForeignPtr CTensor)
_cat_out out tensors dim =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in
  [C.block|void {
    VariableType::_cat_out(*$fptr-ptr:(Tensor* out), pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size)), $(int64_t dim));
   }|] >> pure (out)


--  _cholesky_helper _cholesky_helper
--
_cholesky_helper :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cholesky_helper self upper =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cholesky_helper(*$fptr-ptr:(Tensor* self), $(bool upper)));
   }|] >>= newForeignPtr deleteTensor


--  _cholesky_solve_helper _cholesky_solve_helper
--
_cholesky_solve_helper :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_cholesky_solve_helper self a upper =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cholesky_solve_helper(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a), $(bool upper)));
   }|] >>= newForeignPtr deleteTensor


--  _coalesced_ _coalesced_
--
_coalesced_ :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_coalesced_ self coalesced =
  [C.block|void {
    VariableType::_coalesced_(*$fptr-ptr:(Tensor* self), $(bool coalesced));
   }|] >> pure self


--  _convolution _convolution
--
_convolution :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> Vector Int64 -> Int64 -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor)
_convolution input weight bias stride padding dilation transposed output_padding groups benchmark deterministic cudnn_enabled =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_convolution(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool transposed), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic), $(bool cudnn_enabled)));
   }|] >>= newForeignPtr deleteTensor


--  _convolution_nogroup _convolution_nogroup
--
_convolution_nogroup :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> Vector Int64 -> IO (ForeignPtr CTensor)
_convolution_nogroup input weight bias stride padding dilation transposed output_padding =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_convolution_nogroup(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool transposed), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size))));
   }|] >>= newForeignPtr deleteTensor


--  _copy_from _copy_from
--
_copy_from :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_copy_from self dst non_blocking =
  [C.block|Tensor* {
    return new Tensor(VariableType::_copy_from(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* dst), $(bool non_blocking)));
   }|] >>= newForeignPtr deleteTensor


--  _ctc_loss _ctc_loss
--
_ctc_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_ctc_loss log_probs targets input_lengths target_lengths blank zero_infinity =  V.unsafeWith input_lengths $ \input_lengths__array -> let input_lengths__size = fromIntegral (V.length input_lengths) in V.unsafeWith target_lengths $ \target_lengths__array -> let target_lengths__size = fromIntegral (V.length target_lengths) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_ctc_loss(*$fptr-ptr:(Tensor* log_probs), *$fptr-ptr:(Tensor* targets), ArrayRef<int64_t>($(int64_t* input_lengths__array), $(size_t input_lengths__size)), ArrayRef<int64_t>($(int64_t* target_lengths__array), $(size_t target_lengths__size)), $(int64_t blank), $(bool zero_infinity)));
   }|] >>= unTupleTensorTensor


--  _cudnn_ctc_loss _cudnn_ctc_loss
--
_cudnn_ctc_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_cudnn_ctc_loss log_probs targets input_lengths target_lengths blank deterministic zero_infinity =  V.unsafeWith input_lengths $ \input_lengths__array -> let input_lengths__size = fromIntegral (V.length input_lengths) in V.unsafeWith target_lengths $ \target_lengths__array -> let target_lengths__size = fromIntegral (V.length target_lengths) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_cudnn_ctc_loss(*$fptr-ptr:(Tensor* log_probs), *$fptr-ptr:(Tensor* targets), ArrayRef<int64_t>($(int64_t* input_lengths__array), $(size_t input_lengths__size)), ArrayRef<int64_t>($(int64_t* target_lengths__array), $(size_t target_lengths__size)), $(int64_t blank), $(bool deterministic), $(bool zero_infinity)));
   }|] >>= unTupleTensorTensor


--  _cudnn_init_dropout_state _cudnn_init_dropout_state
--
_cudnn_init_dropout_state :: CDouble -> CBool -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
_cudnn_init_dropout_state dropout train dropout_seed options =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cudnn_init_dropout_state($(double dropout), $(bool train), $(int64_t dropout_seed), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  _cudnn_rnn _cudnn_rnn
--
_cudnn_rnn :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> CBool -> CDouble -> CBool -> CBool -> Vector Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_cudnn_rnn input weight weight_stride0 weight_buf hx cx mode hidden_size num_layers batch_first dropout train bidirectional batch_sizes dropout_state =  V.unsafeWith weight $ \weight__array -> let weight__size = fromIntegral (V.length weight) in V.unsafeWith batch_sizes $ \batch_sizes__array -> let batch_sizes__size = fromIntegral (V.length batch_sizes) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>(VariableType::_cudnn_rnn(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** weight__array), $(size_t weight__size)), $(int64_t weight_stride0), *$fptr-ptr:(Tensor* weight_buf), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* cx), $(int64_t mode), $(int64_t hidden_size), $(int64_t num_layers), $(bool batch_first), $(double dropout), $(bool train), $(bool bidirectional), ArrayRef<int64_t>($(int64_t* batch_sizes__array), $(size_t batch_sizes__size)), *$fptr-ptr:(Tensor* dropout_state)));
   }|] >>= unTupleTensorTensorTensorTensorTensor


--  _cudnn_rnn_flatten_weight _cudnn_rnn_flatten_weight
--
_cudnn_rnn_flatten_weight :: Vector (Ptr CTensor) -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
_cudnn_rnn_flatten_weight weight_arr weight_stride0 input_size mode hidden_size num_layers batch_first bidirectional =  V.unsafeWith weight_arr $ \weight_arr__array -> let weight_arr__size = fromIntegral (V.length weight_arr) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_cudnn_rnn_flatten_weight(pack_tensor_list($(Tensor** weight_arr__array), $(size_t weight_arr__size)), $(int64_t weight_stride0), $(int64_t input_size), $(int64_t mode), $(int64_t hidden_size), $(int64_t num_layers), $(bool batch_first), $(bool bidirectional)));
   }|] >>= newForeignPtr deleteTensor


--  _cufft_clear_plan_cache _cufft_clear_plan_cache
--
_cufft_clear_plan_cache :: Int64 -> IO ()
_cufft_clear_plan_cache device_index =
  [C.block|void {
    return VariableType::_cufft_clear_plan_cache($(int64_t device_index));
   }|]


--  _cufft_get_plan_cache_max_size _cufft_get_plan_cache_max_size
--
_cufft_get_plan_cache_max_size :: Int64 -> IO (Int64)
_cufft_get_plan_cache_max_size device_index =
  [C.block|int64_t {
    return VariableType::_cufft_get_plan_cache_max_size($(int64_t device_index));
   }|]


--  _cufft_get_plan_cache_size _cufft_get_plan_cache_size
--
_cufft_get_plan_cache_size :: Int64 -> IO (Int64)
_cufft_get_plan_cache_size device_index =
  [C.block|int64_t {
    return VariableType::_cufft_get_plan_cache_size($(int64_t device_index));
   }|]


--  _cufft_set_plan_cache_max_size _cufft_set_plan_cache_max_size
--
_cufft_set_plan_cache_max_size :: Int64 -> Int64 -> IO ()
_cufft_set_plan_cache_max_size device_index max_size =
  [C.block|void {
    return VariableType::_cufft_set_plan_cache_max_size($(int64_t device_index), $(int64_t max_size));
   }|]


--  _cumprod _cumprod
--
_cumprod :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_cumprod self dim =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cumprod(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  _cumprod_out _cumprod_out
--
_cumprod_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_cumprod_out out self dim =
  [C.block|void {
    VariableType::_cumprod_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|] >> pure (out)


--  _cumsum _cumsum
--
_cumsum :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_cumsum self dim =
  [C.block|Tensor* {
    return new Tensor(VariableType::_cumsum(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  _cumsum_out _cumsum_out
--
_cumsum_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_cumsum_out out self dim =
  [C.block|void {
    VariableType::_cumsum_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|] >> pure (out)


--  _debug_has_internal_overlap _debug_has_internal_overlap
--
_debug_has_internal_overlap :: ForeignPtr CTensor -> IO (Int64)
_debug_has_internal_overlap self =
  [C.block|int64_t {
    return VariableType::_debug_has_internal_overlap(*$fptr-ptr:(Tensor* self));
   }|]


--  _dequantize_linear _dequantize_linear
--
_dequantize_linear :: ForeignPtr CTensor -> CDouble -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
_dequantize_linear self scale zero_point dtype =
  [C.block|Tensor* {
    return new Tensor(VariableType::_dequantize_linear(*$fptr-ptr:(Tensor* self), $(double scale), $(int64_t zero_point), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  _dimI _dimi
--
_dimi :: ForeignPtr CTensor -> IO (Int64)
_dimi self =
  [C.block|int64_t {
    return VariableType::_dimI(*$fptr-ptr:(Tensor* self));
   }|]


--  _dimV _dimv
--
_dimv :: ForeignPtr CTensor -> IO (Int64)
_dimv self =
  [C.block|int64_t {
    return VariableType::_dimV(*$fptr-ptr:(Tensor* self));
   }|]


--  _dim_arange _dim_arange
--
_dim_arange :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_dim_arange like dim =
  [C.block|Tensor* {
    return new Tensor(VariableType::_dim_arange(*$fptr-ptr:(Tensor* like), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  _dirichlet_grad _dirichlet_grad
--
_dirichlet_grad :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_dirichlet_grad x alpha total =
  [C.block|Tensor* {
    return new Tensor(VariableType::_dirichlet_grad(*$fptr-ptr:(Tensor* x), *$fptr-ptr:(Tensor* alpha), *$fptr-ptr:(Tensor* total)));
   }|] >>= newForeignPtr deleteTensor


--  _embedding_bag _embedding_bag
--
_embedding_bag :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> Int64 -> CBool -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_embedding_bag weight indices offsets scale_grad_by_freq mode sparse per_sample_weights =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,Tensor>(VariableType::_embedding_bag(*$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* offsets), $(bool scale_grad_by_freq), $(int64_t mode), $(bool sparse), *$fptr-ptr:(Tensor* per_sample_weights)));
   }|] >>= unTupleTensorTensorTensorTensor


--  _empty_affine_quantized _empty_affine_quantized
--
_empty_affine_quantized :: Vector Int64 -> ForeignPtr CTensorOptions -> CDouble -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
_empty_affine_quantized size options scale zero_point memory_format =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_empty_affine_quantized(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options), $(double scale), $(int64_t zero_point), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


--  _fft_with_size _fft_with_size
--
_fft_with_size :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> CBool -> Vector Int64 -> CBool -> CBool -> Vector Int64 -> IO (ForeignPtr CTensor)
_fft_with_size self signal_ndim complex_input complex_output inverse checked_signal_sizes normalized onesided output_sizes =  V.unsafeWith checked_signal_sizes $ \checked_signal_sizes__array -> let checked_signal_sizes__size = fromIntegral (V.length checked_signal_sizes) in V.unsafeWith output_sizes $ \output_sizes__array -> let output_sizes__size = fromIntegral (V.length output_sizes) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_fft_with_size(*$fptr-ptr:(Tensor* self), $(int64_t signal_ndim), $(bool complex_input), $(bool complex_output), $(bool inverse), ArrayRef<int64_t>($(int64_t* checked_signal_sizes__array), $(size_t checked_signal_sizes__size)), $(bool normalized), $(bool onesided), ArrayRef<int64_t>($(int64_t* output_sizes__array), $(size_t output_sizes__size))));
   }|] >>= newForeignPtr deleteTensor


--  _fused_dropout _fused_dropout
--
_fused_dropout :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_fused_dropout self p generator =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_fused_dropout(*$fptr-ptr:(Tensor* self), $(double p), $(Generator* generator)));
   }|] >>= unTupleTensorTensor


--  _has_compatible_shallow_copy_type _has_compatible_shallow_copy_type
--
_has_compatible_shallow_copy_type :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (CBool)
_has_compatible_shallow_copy_type self from =
  [C.block|bool {
    return VariableType::_has_compatible_shallow_copy_type(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* from));
   }|]


--  _index_copy_ _index_copy_
--
_index_copy_ :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_index_copy_ self dim index source =
  [C.block|void {
    VariableType::_index_copy_(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source));
   }|] >> pure self


--  _index_put_impl_ _index_put_impl_
--
_index_put_impl_ :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor)
_index_put_impl_ self indices values accumulate unsafe =  V.unsafeWith indices $ \indices__array -> let indices__size = fromIntegral (V.length indices) in
  [C.block|void {
    VariableType::_index_put_impl_(*$fptr-ptr:(Tensor* self), pack_tensor_list($(Tensor** indices__array), $(size_t indices__size)), *$fptr-ptr:(Tensor* values), $(bool accumulate), $(bool unsafe));
   }|] >> pure self


--  _indices _indices
--
_indices :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_indices self =
  [C.block|Tensor* {
    return new Tensor(VariableType::_indices(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  _inverse_helper _inverse_helper
--
_inverse_helper :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_inverse_helper self =
  [C.block|Tensor* {
    return new Tensor(VariableType::_inverse_helper(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  _local_scalar_dense _local_scalar_dense
--
_local_scalar_dense :: ForeignPtr CTensor -> IO (ForeignPtr CScalar)
_local_scalar_dense self =
  [C.block|Scalar* {
    return new Scalar(VariableType::_local_scalar_dense(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteScalar'


--  _log_softmax _log_softmax
--
_log_softmax :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
_log_softmax self dim half_to_float =
  [C.block|Tensor* {
    return new Tensor(VariableType::_log_softmax(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool half_to_float)));
   }|] >>= newForeignPtr deleteTensor


--  _lu_solve_helper _lu_solve_helper
--
_lu_solve_helper :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_lu_solve_helper self lu_data lu_pivots =
  [C.block|Tensor* {
    return new Tensor(VariableType::_lu_solve_helper(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* lu_data), *$fptr-ptr:(Tensor* lu_pivots)));
   }|] >>= newForeignPtr deleteTensor


--  _lu_with_info _lu_with_info
--
_lu_with_info :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_lu_with_info self pivot check_errors =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::_lu_with_info(*$fptr-ptr:(Tensor* self), $(bool pivot), $(bool check_errors)));
   }|] >>= unTupleTensorTensorTensor


--  _masked_scale _masked_scale
--
_masked_scale :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor)
_masked_scale self mask scale =
  [C.block|Tensor* {
    return new Tensor(VariableType::_masked_scale(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask), $(double scale)));
   }|] >>= newForeignPtr deleteTensor


--  _max _max
--
_max :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_max self dim keepdim =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_max(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


--  _max_out _max_out
--
_max_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_max_out max max_indices self dim keepdim =
  [C.block|void {
    VariableType::_max_out(*$fptr-ptr:(Tensor* max), *$fptr-ptr:(Tensor* max_indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (max,max_indices)


--  _min _min
--
_min :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_min self dim keepdim =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_min(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


--  _min_out _min_out
--
_min_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_min_out min min_indices self dim keepdim =
  [C.block|void {
    VariableType::_min_out(*$fptr-ptr:(Tensor* min), *$fptr-ptr:(Tensor* min_indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (min,min_indices)


--  _mkldnn_reshape _mkldnn_reshape
--
_mkldnn_reshape :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
_mkldnn_reshape self shape =  V.unsafeWith shape $ \shape__array -> let shape__size = fromIntegral (V.length shape) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_mkldnn_reshape(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* shape__array), $(size_t shape__size))));
   }|] >>= newForeignPtr deleteTensor


--  _mkldnn_transpose _mkldnn_transpose
--
_mkldnn_transpose :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
_mkldnn_transpose self dim0 dim1 =
  [C.block|Tensor* {
    return new Tensor(VariableType::_mkldnn_transpose(*$fptr-ptr:(Tensor* self), $(int64_t dim0), $(int64_t dim1)));
   }|] >>= newForeignPtr deleteTensor


--  _mkldnn_transpose_ _mkldnn_transpose_
--
_mkldnn_transpose_ :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
_mkldnn_transpose_ self dim0 dim1 =
  [C.block|void {
    VariableType::_mkldnn_transpose_(*$fptr-ptr:(Tensor* self), $(int64_t dim0), $(int64_t dim1));
   }|] >> pure self


--  _mode _mode
--
_mode :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_mode self dim keepdim =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_mode(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


--  _mode_out _mode_out
--
_mode_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_mode_out values indices self dim keepdim =
  [C.block|void {
    VariableType::_mode_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (values,indices)


--  _multinomial_alias_draw _multinomial_alias_draw
--
_multinomial_alias_draw :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
_multinomial_alias_draw j q num_samples generator =
  [C.block|Tensor* {
    return new Tensor(VariableType::_multinomial_alias_draw(*$fptr-ptr:(Tensor* j), *$fptr-ptr:(Tensor* q), $(int64_t num_samples), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


--  _multinomial_alias_setup _multinomial_alias_setup
--
_multinomial_alias_setup :: ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_multinomial_alias_setup probs =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_multinomial_alias_setup(*$fptr-ptr:(Tensor* probs)));
   }|] >>= unTupleTensorTensor


--  _nnpack_available _nnpack_available
--
_nnpack_available :: IO (CBool)
_nnpack_available  =
  [C.block|bool {
    return VariableType::_nnpack_available();
   }|]


--  _nnpack_spatial_convolution _nnpack_spatial_convolution
--
_nnpack_spatial_convolution :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
_nnpack_spatial_convolution input weight bias padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_nnpack_spatial_convolution(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


--  _nnz _nnz
--
_nnz :: ForeignPtr CTensor -> IO (Int64)
_nnz self =
  [C.block|int64_t {
    return VariableType::_nnz(*$fptr-ptr:(Tensor* self));
   }|]


--  _pack_padded_sequence _pack_padded_sequence
--
_pack_padded_sequence :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_pack_padded_sequence input lengths batch_first =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_pack_padded_sequence(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* lengths), $(bool batch_first)));
   }|] >>= unTupleTensorTensor


--  _pad_packed_sequence _pad_packed_sequence
--
_pad_packed_sequence :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> ForeignPtr CScalar -> Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_pad_packed_sequence dataX batch_sizes batch_first padding_value total_length =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_pad_packed_sequence(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), $(bool batch_first), *$fptr-ptr:(Scalar* padding_value), $(int64_t total_length)));
   }|] >>= unTupleTensorTensor


--  _per_tensor_affine_qtensor _per_tensor_affine_qtensor
--
_per_tensor_affine_qtensor :: ForeignPtr CTensor -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
_per_tensor_affine_qtensor self scale zero_point =
  [C.block|Tensor* {
    return new Tensor(VariableType::_per_tensor_affine_qtensor(*$fptr-ptr:(Tensor* self), $(double scale), $(int64_t zero_point)));
   }|] >>= newForeignPtr deleteTensor


--  _qr_helper _qr_helper
--
_qr_helper :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_qr_helper self some =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_qr_helper(*$fptr-ptr:(Tensor* self), $(bool some)));
   }|] >>= unTupleTensorTensor


--  _reshape_from_tensor _reshape_from_tensor
--
_reshape_from_tensor :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_reshape_from_tensor self shape =
  [C.block|Tensor* {
    return new Tensor(VariableType::_reshape_from_tensor(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* shape)));
   }|] >>= newForeignPtr deleteTensor


--  _s_where _s_where
--
_s_where :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_s_where condition self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::_s_where(*$fptr-ptr:(Tensor* condition), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  _sample_dirichlet _sample_dirichlet
--
_sample_dirichlet :: ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
_sample_dirichlet self generator =
  [C.block|Tensor* {
    return new Tensor(VariableType::_sample_dirichlet(*$fptr-ptr:(Tensor* self), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


--  _shape_as_tensor _shape_as_tensor
--
_shape_as_tensor :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_shape_as_tensor self =
  [C.block|Tensor* {
    return new Tensor(VariableType::_shape_as_tensor(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  _sobol_engine_draw _sobol_engine_draw
--
_sobol_engine_draw :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> Int64 -> Int64 -> Int8 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_sobol_engine_draw quasi n sobolstate dimension num_generated dtype =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_sobol_engine_draw(*$fptr-ptr:(Tensor* quasi), $(int64_t n), *$fptr-ptr:(Tensor* sobolstate), $(int64_t dimension), $(int64_t num_generated), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= unTupleTensorTensor


--  _sobol_engine_ff_ _sobol_engine_ff_
--
_sobol_engine_ff_ :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
_sobol_engine_ff_ self n sobolstate dimension num_generated =
  [C.block|void {
    VariableType::_sobol_engine_ff_(*$fptr-ptr:(Tensor* self), $(int64_t n), *$fptr-ptr:(Tensor* sobolstate), $(int64_t dimension), $(int64_t num_generated));
   }|] >> pure self


--  _sobol_engine_initialize_state_ _sobol_engine_initialize_state_
--
_sobol_engine_initialize_state_ :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_sobol_engine_initialize_state_ self dimension =
  [C.block|void {
    VariableType::_sobol_engine_initialize_state_(*$fptr-ptr:(Tensor* self), $(int64_t dimension));
   }|] >> pure self


--  _sobol_engine_scramble_ _sobol_engine_scramble_
--
_sobol_engine_scramble_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_sobol_engine_scramble_ self ltm dimension =
  [C.block|void {
    VariableType::_sobol_engine_scramble_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* ltm), $(int64_t dimension));
   }|] >> pure self


--  _softmax _softmax
--
_softmax :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
_softmax self dim half_to_float =
  [C.block|Tensor* {
    return new Tensor(VariableType::_softmax(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool half_to_float)));
   }|] >>= newForeignPtr deleteTensor


--  _solve_helper _solve_helper
--
_solve_helper :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_solve_helper self a =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_solve_helper(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a)));
   }|] >>= unTupleTensorTensor


--  _sparse_add_out _sparse_add_out
--
_sparse_add_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_sparse_add_out out self other alpha =
  [C.block|void {
    VariableType::_sparse_add_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  _sparse_addmm _sparse_addmm
--
_sparse_addmm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_sparse_addmm self sparse dense beta alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::_sparse_addmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* sparse), *$fptr-ptr:(Tensor* dense), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  _sparse_coo_tensor_unsafe _sparse_coo_tensor_unsafe
--
_sparse_coo_tensor_unsafe :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
_sparse_coo_tensor_unsafe indices values size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_sparse_coo_tensor_unsafe(*$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* values), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  _sparse_coo_tensor_with_dims _sparse_coo_tensor_with_dims
--
_sparse_coo_tensor_with_dims :: Int64 -> Int64 -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
_sparse_coo_tensor_with_dims sparse_dim dense_dim size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_sparse_coo_tensor_with_dims($(int64_t sparse_dim), $(int64_t dense_dim), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  _sparse_coo_tensor_with_dims_and_tensors _sparse_coo_tensor_with_dims_and_tensors
--
_sparse_coo_tensor_with_dims_and_tensors :: Int64 -> Int64 -> Vector Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
_sparse_coo_tensor_with_dims_and_tensors sparse_dim dense_dim size indices values options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_sparse_coo_tensor_with_dims_and_tensors($(int64_t sparse_dim), $(int64_t dense_dim), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* values), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  _sparse_dense_add_out _sparse_dense_add_out
--
_sparse_dense_add_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_sparse_dense_add_out out self other alpha =
  [C.block|void {
    VariableType::_sparse_dense_add_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  _sparse_div_scalar_out _sparse_div_scalar_out
--
_sparse_div_scalar_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_sparse_div_scalar_out out self other =
  [C.block|void {
    VariableType::_sparse_div_scalar_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


--  _sparse_div_zerodim_out _sparse_div_zerodim_out
--
_sparse_div_zerodim_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_sparse_div_zerodim_out out self other =
  [C.block|void {
    VariableType::_sparse_div_zerodim_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  _sparse_mm _sparse_mm
--
_sparse_mm :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_sparse_mm sparse dense =
  [C.block|Tensor* {
    return new Tensor(VariableType::_sparse_mm(*$fptr-ptr:(Tensor* sparse), *$fptr-ptr:(Tensor* dense)));
   }|] >>= newForeignPtr deleteTensor


--  _sparse_mul_out _sparse_mul_out
--
_sparse_mul_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_sparse_mul_out out self other =
  [C.block|void {
    VariableType::_sparse_mul_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  _sparse_mul_scalar_out _sparse_mul_scalar_out
--
_sparse_mul_scalar_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
_sparse_mul_scalar_out out self other =
  [C.block|void {
    VariableType::_sparse_mul_scalar_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


--  _sparse_mul_zerodim_out _sparse_mul_zerodim_out
--
_sparse_mul_zerodim_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_sparse_mul_zerodim_out out self other =
  [C.block|void {
    VariableType::_sparse_mul_zerodim_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  _sparse_sum _sparse_sum
--
_sparse_sum :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_sparse_sum self =
  [C.block|Tensor* {
    return new Tensor(VariableType::_sparse_sum(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  _sparse_sum _sparse_sum__1
--
_sparse_sum__1 :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
_sparse_sum__1 self dtype =
  [C.block|Tensor* {
    return new Tensor(VariableType::_sparse_sum(*$fptr-ptr:(Tensor* self), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  _sparse_sum _sparse_sum__2
--
_sparse_sum__2 :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
_sparse_sum__2 self dim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_sparse_sum(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size))));
   }|] >>= newForeignPtr deleteTensor


--  _sparse_sum _sparse_sum__3
--
_sparse_sum__3 :: ForeignPtr CTensor -> Vector Int64 -> Int8 -> IO (ForeignPtr CTensor)
_sparse_sum__3 self dim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_sparse_sum(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  _standard_gamma _standard_gamma
--
_standard_gamma :: ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
_standard_gamma self generator =
  [C.block|Tensor* {
    return new Tensor(VariableType::_standard_gamma(*$fptr-ptr:(Tensor* self), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


--  _standard_gamma_grad _standard_gamma_grad
--
_standard_gamma_grad :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_standard_gamma_grad self output =
  [C.block|Tensor* {
    return new Tensor(VariableType::_standard_gamma_grad(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* output)));
   }|] >>= newForeignPtr deleteTensor


--  _std _std
--
_std :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_std self unbiased =
  [C.block|Tensor* {
    return new Tensor(VariableType::_std(*$fptr-ptr:(Tensor* self), $(bool unbiased)));
   }|] >>= newForeignPtr deleteTensor


--  _svd_helper _svd_helper
--
_svd_helper :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_svd_helper self some compute_uv =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::_svd_helper(*$fptr-ptr:(Tensor* self), $(bool some), $(bool compute_uv)));
   }|] >>= unTupleTensorTensorTensor


--  _symeig_helper _symeig_helper
--
_symeig_helper :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_symeig_helper self eigenvectors upper =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_symeig_helper(*$fptr-ptr:(Tensor* self), $(bool eigenvectors), $(bool upper)));
   }|] >>= unTupleTensorTensor


--  _thnn_fused_gru_cell _thnn_fused_gru_cell
--
_thnn_fused_gru_cell :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_thnn_fused_gru_cell input_gates hidden_gates hx input_bias hidden_bias =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_thnn_fused_gru_cell(*$fptr-ptr:(Tensor* input_gates), *$fptr-ptr:(Tensor* hidden_gates), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* input_bias), *$fptr-ptr:(Tensor* hidden_bias)));
   }|] >>= unTupleTensorTensor


--  _thnn_fused_lstm_cell _thnn_fused_lstm_cell
--
_thnn_fused_lstm_cell :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_thnn_fused_lstm_cell input_gates hidden_gates cx input_bias hidden_bias =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::_thnn_fused_lstm_cell(*$fptr-ptr:(Tensor* input_gates), *$fptr-ptr:(Tensor* hidden_gates), *$fptr-ptr:(Tensor* cx), *$fptr-ptr:(Tensor* input_bias), *$fptr-ptr:(Tensor* hidden_bias)));
   }|] >>= unTupleTensorTensorTensor


--  _triangular_solve_helper _triangular_solve_helper
--
_triangular_solve_helper :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_triangular_solve_helper self a upper transpose unitriangular =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_triangular_solve_helper(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a), $(bool upper), $(bool transpose), $(bool unitriangular)));
   }|] >>= unTupleTensorTensor


--  _trilinear _trilinear
--
_trilinear :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
_trilinear i1 i2 i3 expand1 expand2 expand3 sumdim unroll_dim =  V.unsafeWith expand1 $ \expand1__array -> let expand1__size = fromIntegral (V.length expand1) in V.unsafeWith expand2 $ \expand2__array -> let expand2__size = fromIntegral (V.length expand2) in V.unsafeWith expand3 $ \expand3__array -> let expand3__size = fromIntegral (V.length expand3) in V.unsafeWith sumdim $ \sumdim__array -> let sumdim__size = fromIntegral (V.length sumdim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_trilinear(*$fptr-ptr:(Tensor* i1), *$fptr-ptr:(Tensor* i2), *$fptr-ptr:(Tensor* i3), ArrayRef<int64_t>($(int64_t* expand1__array), $(size_t expand1__size)), ArrayRef<int64_t>($(int64_t* expand2__array), $(size_t expand2__size)), ArrayRef<int64_t>($(int64_t* expand3__array), $(size_t expand3__size)), ArrayRef<int64_t>($(int64_t* sumdim__array), $(size_t sumdim__size)), $(int64_t unroll_dim)));
   }|] >>= newForeignPtr deleteTensor


--  _unique _unique
--
_unique :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_unique self sorted return_inverse =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_unique(*$fptr-ptr:(Tensor* self), $(bool sorted), $(bool return_inverse)));
   }|] >>= unTupleTensorTensor


--  _unique2 _unique2
--
_unique2 :: ForeignPtr CTensor -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
_unique2 self sorted return_inverse return_counts =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::_unique2(*$fptr-ptr:(Tensor* self), $(bool sorted), $(bool return_inverse), $(bool return_counts)));
   }|] >>= unTupleTensorTensorTensor


--  _unsafe_view _unsafe_view
--
_unsafe_view :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
_unsafe_view self size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::_unsafe_view(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size))));
   }|] >>= newForeignPtr deleteTensor


--  _values _values
--
_values :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
_values self =
  [C.block|Tensor* {
    return new Tensor(VariableType::_values(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  _var _var
--
_var :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
_var self unbiased =
  [C.block|Tensor* {
    return new Tensor(VariableType::_var(*$fptr-ptr:(Tensor* self), $(bool unbiased)));
   }|] >>= newForeignPtr deleteTensor


--  _weight_norm _weight_norm
--
_weight_norm :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
_weight_norm v g dim =
  [C.block|Tensor* {
    return new Tensor(VariableType::_weight_norm(*$fptr-ptr:(Tensor* v), *$fptr-ptr:(Tensor* g), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  _weight_norm_cuda_interface _weight_norm_cuda_interface
--
_weight_norm_cuda_interface :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
_weight_norm_cuda_interface v g dim =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::_weight_norm_cuda_interface(*$fptr-ptr:(Tensor* v), *$fptr-ptr:(Tensor* g), $(int64_t dim)));
   }|] >>= unTupleTensorTensor


--  abs abs
--
abs :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
abs self =
  [C.block|Tensor* {
    return new Tensor(VariableType::abs(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  abs_ abs_
--
abs_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
abs_ self =
  [C.block|void {
    VariableType::abs_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  abs_out abs_out
--
abs_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
abs_out out self =
  [C.block|void {
    VariableType::abs_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  acos acos
--
acos :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
acos self =
  [C.block|Tensor* {
    return new Tensor(VariableType::acos(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  acos_ acos_
--
acos_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
acos_ self =
  [C.block|void {
    VariableType::acos_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  acos_out acos_out
--
acos_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
acos_out out self =
  [C.block|void {
    VariableType::acos_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  adaptive_avg_pool1d adaptive_avg_pool1d
--
adaptive_avg_pool1d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
adaptive_avg_pool1d self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::adaptive_avg_pool1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


--  adaptive_avg_pool2d adaptive_avg_pool2d
--
adaptive_avg_pool2d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
adaptive_avg_pool2d self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::adaptive_avg_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


--  adaptive_avg_pool2d_out adaptive_avg_pool2d_out
--
adaptive_avg_pool2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
adaptive_avg_pool2d_out out self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::adaptive_avg_pool2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out)


--  adaptive_avg_pool3d adaptive_avg_pool3d
--
adaptive_avg_pool3d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
adaptive_avg_pool3d self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::adaptive_avg_pool3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


--  adaptive_avg_pool3d_out adaptive_avg_pool3d_out
--
adaptive_avg_pool3d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
adaptive_avg_pool3d_out out self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::adaptive_avg_pool3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out)


--  adaptive_max_pool1d adaptive_max_pool1d
--
adaptive_max_pool1d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
adaptive_max_pool1d self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::adaptive_max_pool1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= unTupleTensorTensor


--  adaptive_max_pool2d adaptive_max_pool2d
--
adaptive_max_pool2d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
adaptive_max_pool2d self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::adaptive_max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= unTupleTensorTensor


--  adaptive_max_pool2d_out adaptive_max_pool2d_out
--
adaptive_max_pool2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
adaptive_max_pool2d_out out indices self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::adaptive_max_pool2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out,indices)


--  adaptive_max_pool3d adaptive_max_pool3d
--
adaptive_max_pool3d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
adaptive_max_pool3d self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::adaptive_max_pool3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= unTupleTensorTensor


--  adaptive_max_pool3d_out adaptive_max_pool3d_out
--
adaptive_max_pool3d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
adaptive_max_pool3d_out out indices self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::adaptive_max_pool3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out,indices)


--  add add
--
add :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
add self other alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::add(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  add add__1
--
add__1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
add__1 self other alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::add(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  add_ add_
--
add_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
add_ self other alpha =
  [C.block|void {
    VariableType::add_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  add_ add___1
--
add___1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
add___1 self other alpha =
  [C.block|void {
    VariableType::add_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  add_out add_out
--
add_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
add_out out self other alpha =
  [C.block|void {
    VariableType::add_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  addbmm addbmm
--
addbmm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addbmm self batch1 batch2 beta alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::addbmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  addbmm_ addbmm_
--
addbmm_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addbmm_ self batch1 batch2 beta alpha =
  [C.block|void {
    VariableType::addbmm_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  addbmm_out addbmm_out
--
addbmm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addbmm_out out self batch1 batch2 beta alpha =
  [C.block|void {
    VariableType::addbmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  addcdiv addcdiv
--
addcdiv :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcdiv self tensor1 tensor2 value =
  [C.block|Tensor* {
    return new Tensor(VariableType::addcdiv(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


--  addcdiv_ addcdiv_
--
addcdiv_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcdiv_ self tensor1 tensor2 value =
  [C.block|void {
    VariableType::addcdiv_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


--  addcdiv_out addcdiv_out
--
addcdiv_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcdiv_out out self tensor1 tensor2 value =
  [C.block|void {
    VariableType::addcdiv_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value));
   }|] >> pure (out)


--  addcmul addcmul
--
addcmul :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcmul self tensor1 tensor2 value =
  [C.block|Tensor* {
    return new Tensor(VariableType::addcmul(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


--  addcmul_ addcmul_
--
addcmul_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcmul_ self tensor1 tensor2 value =
  [C.block|void {
    VariableType::addcmul_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


--  addcmul_out addcmul_out
--
addcmul_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addcmul_out out self tensor1 tensor2 value =
  [C.block|void {
    VariableType::addcmul_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor1), *$fptr-ptr:(Tensor* tensor2), *$fptr-ptr:(Scalar* value));
   }|] >> pure (out)


--  addmm addmm
--
addmm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmm self mat1 mat2 beta alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::addmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  addmm_ addmm_
--
addmm_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmm_ self mat1 mat2 beta alpha =
  [C.block|void {
    VariableType::addmm_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  addmm_out addmm_out
--
addmm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmm_out out self mat1 mat2 beta alpha =
  [C.block|void {
    VariableType::addmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  addmv addmv
--
addmv :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmv self mat vec beta alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::addmv(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat), *$fptr-ptr:(Tensor* vec), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  addmv_ addmv_
--
addmv_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmv_ self mat vec beta alpha =
  [C.block|void {
    VariableType::addmv_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat), *$fptr-ptr:(Tensor* vec), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  addmv_out addmv_out
--
addmv_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addmv_out out self mat vec beta alpha =
  [C.block|void {
    VariableType::addmv_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat), *$fptr-ptr:(Tensor* vec), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  addr addr
--
addr :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addr self vec1 vec2 beta alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::addr(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  addr_ addr_
--
addr_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addr_ self vec1 vec2 beta alpha =
  [C.block|void {
    VariableType::addr_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  addr_out addr_out
--
addr_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
addr_out out self vec1 vec2 beta alpha =
  [C.block|void {
    VariableType::addr_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec1), *$fptr-ptr:(Tensor* vec2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  affine_grid_generator affine_grid_generator
--
affine_grid_generator :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
affine_grid_generator theta size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::affine_grid_generator(*$fptr-ptr:(Tensor* theta), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size))));
   }|] >>= newForeignPtr deleteTensor


--  alias alias
--
alias :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
alias self =
  [C.block|Tensor* {
    return new Tensor(VariableType::alias(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  all all
--
all :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
all self dim keepdim =
  [C.block|Tensor* {
    return new Tensor(VariableType::all(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  all all__1
--
all__1 :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
all__1 self =
  [C.block|Tensor* {
    return new Tensor(VariableType::all(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  all_out all_out
--
all_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
all_out out self dim keepdim =
  [C.block|void {
    VariableType::all_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (out)


--  allclose allclose
--
allclose :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> CBool -> IO (CBool)
allclose self other rtol atol equal_nan =
  [C.block|bool {
    return VariableType::allclose(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), $(double rtol), $(double atol), $(bool equal_nan));
   }|]


--  alpha_dropout alpha_dropout
--
alpha_dropout :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
alpha_dropout input p train =
  [C.block|Tensor* {
    return new Tensor(VariableType::alpha_dropout(*$fptr-ptr:(Tensor* input), $(double p), $(bool train)));
   }|] >>= newForeignPtr deleteTensor


--  alpha_dropout_ alpha_dropout_
--
alpha_dropout_ :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
alpha_dropout_ self p train =
  [C.block|void {
    VariableType::alpha_dropout_(*$fptr-ptr:(Tensor* self), $(double p), $(bool train));
   }|] >> pure self


--  any any
--
any :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
any self dim keepdim =
  [C.block|Tensor* {
    return new Tensor(VariableType::any(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  any any__1
--
any__1 :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
any__1 self =
  [C.block|Tensor* {
    return new Tensor(VariableType::any(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  any_out any_out
--
any_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
any_out out self dim keepdim =
  [C.block|void {
    VariableType::any_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (out)


--  arange arange
--
arange :: ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
arange end options =
  [C.block|Tensor* {
    return new Tensor(VariableType::arange(*$fptr-ptr:(Scalar* end), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  arange arange__1
--
arange__1 :: ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
arange__1 start end options =
  [C.block|Tensor* {
    return new Tensor(VariableType::arange(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  arange arange__2
--
arange__2 :: ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
arange__2 start end step options =
  [C.block|Tensor* {
    return new Tensor(VariableType::arange(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(Scalar* step), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  arange_out arange_out
--
arange_out :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
arange_out out end =
  [C.block|void {
    VariableType::arange_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Scalar* end));
   }|] >> pure (out)


--  arange_out arange_out__1
--
arange_out__1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
arange_out__1 out start end step =
  [C.block|void {
    VariableType::arange_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(Scalar* step));
   }|] >> pure (out)


--  argmax argmax
--
argmax :: ForeignPtr CTensor -> Maybe Int64 -> CBool -> IO (ForeignPtr CTensor)
argmax self dim keepdim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in
  [C.block|Tensor* {
    return new Tensor(VariableType::argmax(*$fptr-ptr:(Tensor* self), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  argmin argmin
--
argmin :: ForeignPtr CTensor -> Maybe Int64 -> CBool -> IO (ForeignPtr CTensor)
argmin self dim keepdim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in
  [C.block|Tensor* {
    return new Tensor(VariableType::argmin(*$fptr-ptr:(Tensor* self), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  argsort argsort
--
argsort :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
argsort self dim descending =
  [C.block|Tensor* {
    return new Tensor(VariableType::argsort(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool descending)));
   }|] >>= newForeignPtr deleteTensor


--  as_strided as_strided
--
as_strided :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Maybe Int64 -> IO (ForeignPtr CTensor)
as_strided self size stride storage_offset =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in let (storage_offset__is_present, storage_offset__value) = splitMaybe storage_offset 0 in
  [C.block|Tensor* {
    return new Tensor(VariableType::as_strided(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ($(bool storage_offset__is_present) ? make_optional($(int64_t storage_offset__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


--  as_strided_ as_strided_
--
as_strided_ :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Maybe Int64 -> IO (ForeignPtr CTensor)
as_strided_ self size stride storage_offset =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in let (storage_offset__is_present, storage_offset__value) = splitMaybe storage_offset 0 in
  [C.block|void {
    VariableType::as_strided_(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ($(bool storage_offset__is_present) ? make_optional($(int64_t storage_offset__value)) : c10::nullopt));
   }|] >> pure self


--  asin asin
--
asin :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
asin self =
  [C.block|Tensor* {
    return new Tensor(VariableType::asin(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  asin_ asin_
--
asin_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
asin_ self =
  [C.block|void {
    VariableType::asin_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  asin_out asin_out
--
asin_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
asin_out out self =
  [C.block|void {
    VariableType::asin_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  atan atan
--
atan :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan self =
  [C.block|Tensor* {
    return new Tensor(VariableType::atan(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  atan2 atan2
--
atan2 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan2 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::atan2(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  atan2_ atan2_
--
atan2_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan2_ self other =
  [C.block|void {
    VariableType::atan2_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  atan2_out atan2_out
--
atan2_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan2_out out self other =
  [C.block|void {
    VariableType::atan2_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  atan_ atan_
--
atan_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan_ self =
  [C.block|void {
    VariableType::atan_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  atan_out atan_out
--
atan_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
atan_out out self =
  [C.block|void {
    VariableType::atan_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  avg_pool1d avg_pool1d
--
avg_pool1d :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
avg_pool1d self kernel_size stride padding ceil_mode count_include_pad =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::avg_pool1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad)));
   }|] >>= newForeignPtr deleteTensor


--  avg_pool2d avg_pool2d
--
avg_pool2d :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor)
avg_pool2d self kernel_size stride padding ceil_mode count_include_pad divisor_override =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in let (divisor_override__is_present, divisor_override__value) = splitMaybe divisor_override 0 in
  [C.block|Tensor* {
    return new Tensor(VariableType::avg_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad), ($(bool divisor_override__is_present) ? make_optional($(int64_t divisor_override__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


--  avg_pool2d_out avg_pool2d_out
--
avg_pool2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor)
avg_pool2d_out out self kernel_size stride padding ceil_mode count_include_pad divisor_override =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in let (divisor_override__is_present, divisor_override__value) = splitMaybe divisor_override 0 in
  [C.block|void {
    VariableType::avg_pool2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad), ($(bool divisor_override__is_present) ? make_optional($(int64_t divisor_override__value)) : c10::nullopt));
   }|] >> pure (out)


--  avg_pool3d avg_pool3d
--
avg_pool3d :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor)
avg_pool3d self kernel_size stride padding ceil_mode count_include_pad divisor_override =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in let (divisor_override__is_present, divisor_override__value) = splitMaybe divisor_override 0 in
  [C.block|Tensor* {
    return new Tensor(VariableType::avg_pool3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad), ($(bool divisor_override__is_present) ? make_optional($(int64_t divisor_override__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


--  avg_pool3d_out avg_pool3d_out
--
avg_pool3d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor)
avg_pool3d_out out self kernel_size stride padding ceil_mode count_include_pad divisor_override =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in let (divisor_override__is_present, divisor_override__value) = splitMaybe divisor_override 0 in
  [C.block|void {
    VariableType::avg_pool3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), $(bool ceil_mode), $(bool count_include_pad), ($(bool divisor_override__is_present) ? make_optional($(int64_t divisor_override__value)) : c10::nullopt));
   }|] >> pure (out)


--  backward backward
--
backward :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> IO ()
backward self gradient keep_graph create_graph =
  [C.block|void {
    return VariableType::backward(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* gradient), $(bool keep_graph), $(bool create_graph));
   }|]


--  baddbmm baddbmm
--
baddbmm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
baddbmm self batch1 batch2 beta alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::baddbmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  baddbmm_ baddbmm_
--
baddbmm_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
baddbmm_ self batch1 batch2 beta alpha =
  [C.block|void {
    VariableType::baddbmm_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  baddbmm_out baddbmm_out
--
baddbmm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
baddbmm_out out self batch1 batch2 beta alpha =
  [C.block|void {
    VariableType::baddbmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* batch1), *$fptr-ptr:(Tensor* batch2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  bartlett_window bartlett_window
--
bartlett_window :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
bartlett_window window_length options =
  [C.block|Tensor* {
    return new Tensor(VariableType::bartlett_window($(int64_t window_length), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  bartlett_window bartlett_window__1
--
bartlett_window__1 :: Int64 -> CBool -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
bartlett_window__1 window_length periodic options =
  [C.block|Tensor* {
    return new Tensor(VariableType::bartlett_window($(int64_t window_length), $(bool periodic), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  batch_norm batch_norm
--
batch_norm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> CBool -> IO (ForeignPtr CTensor)
batch_norm input weight bias running_mean running_var training momentum eps cudnn_enabled =
  [C.block|Tensor* {
    return new Tensor(VariableType::batch_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool training), $(double momentum), $(double eps), $(bool cudnn_enabled)));
   }|] >>= newForeignPtr deleteTensor


--  batch_norm_elemt batch_norm_elemt
--
batch_norm_elemt :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor)
batch_norm_elemt input weight bias mean invstd eps =
  [C.block|Tensor* {
    return new Tensor(VariableType::batch_norm_elemt(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* mean), *$fptr-ptr:(Tensor* invstd), $(double eps)));
   }|] >>= newForeignPtr deleteTensor


--  batch_norm_gather_stats batch_norm_gather_stats
--
batch_norm_gather_stats :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
batch_norm_gather_stats input mean invstd running_mean running_var momentum eps count =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::batch_norm_gather_stats(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* mean), *$fptr-ptr:(Tensor* invstd), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(double momentum), $(double eps), $(int64_t count)));
   }|] >>= unTupleTensorTensor


--  batch_norm_gather_stats_with_counts batch_norm_gather_stats_with_counts
--
batch_norm_gather_stats_with_counts :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> Vector Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
batch_norm_gather_stats_with_counts input mean invstd running_mean running_var momentum eps counts =  V.unsafeWith counts $ \counts__array -> let counts__size = fromIntegral (V.length counts) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::batch_norm_gather_stats_with_counts(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* mean), *$fptr-ptr:(Tensor* invstd), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(double momentum), $(double eps), ArrayRef<int64_t>($(int64_t* counts__array), $(size_t counts__size))));
   }|] >>= unTupleTensorTensor


--  batch_norm_stats batch_norm_stats
--
batch_norm_stats :: ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
batch_norm_stats input eps =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::batch_norm_stats(*$fptr-ptr:(Tensor* input), $(double eps)));
   }|] >>= unTupleTensorTensor


--  batch_norm_update_stats batch_norm_update_stats
--
batch_norm_update_stats :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
batch_norm_update_stats input running_mean running_var momentum =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::batch_norm_update_stats(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(double momentum)));
   }|] >>= unTupleTensorTensor


--  bernoulli bernoulli
--
bernoulli :: ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
bernoulli self generator =
  [C.block|Tensor* {
    return new Tensor(VariableType::bernoulli(*$fptr-ptr:(Tensor* self), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


--  bernoulli bernoulli__1
--
bernoulli__1 :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
bernoulli__1 self p generator =
  [C.block|Tensor* {
    return new Tensor(VariableType::bernoulli(*$fptr-ptr:(Tensor* self), $(double p), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


--  bernoulli_ bernoulli_
--
bernoulli_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
bernoulli_ self p generator =
  [C.block|void {
    VariableType::bernoulli_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* p), $(Generator* generator));
   }|] >> pure self


--  bernoulli_ bernoulli___1
--
bernoulli___1 :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
bernoulli___1 self p generator =
  [C.block|void {
    VariableType::bernoulli_(*$fptr-ptr:(Tensor* self), $(double p), $(Generator* generator));
   }|] >> pure self


--  bernoulli_out bernoulli_out
--
bernoulli_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
bernoulli_out out self generator =
  [C.block|void {
    VariableType::bernoulli_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(Generator* generator));
   }|] >> pure (out)


--  bilinear bilinear
--
bilinear :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bilinear input1 input2 weight bias =
  [C.block|Tensor* {
    return new Tensor(VariableType::bilinear(*$fptr-ptr:(Tensor* input1), *$fptr-ptr:(Tensor* input2), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


--  binary_cross_entropy binary_cross_entropy
--
binary_cross_entropy :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
binary_cross_entropy self target weight reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::binary_cross_entropy(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  binary_cross_entropy_out binary_cross_entropy_out
--
binary_cross_entropy_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
binary_cross_entropy_out out self target weight reduction =
  [C.block|void {
    VariableType::binary_cross_entropy_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction));
   }|] >> pure (out)


--  binary_cross_entropy_with_logits binary_cross_entropy_with_logits
--
binary_cross_entropy_with_logits :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
binary_cross_entropy_with_logits self target weight pos_weight reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::binary_cross_entropy_with_logits(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* pos_weight), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  bincount bincount
--
bincount :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
bincount self weights minlength =
  [C.block|Tensor* {
    return new Tensor(VariableType::bincount(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weights), $(int64_t minlength)));
   }|] >>= newForeignPtr deleteTensor


--  bitwise_not bitwise_not
--
bitwise_not :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_not self =
  [C.block|Tensor* {
    return new Tensor(VariableType::bitwise_not(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  bitwise_not_ bitwise_not_
--
bitwise_not_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_not_ self =
  [C.block|void {
    VariableType::bitwise_not_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  bitwise_not_out bitwise_not_out
--
bitwise_not_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bitwise_not_out out self =
  [C.block|void {
    VariableType::bitwise_not_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  blackman_window blackman_window
--
blackman_window :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
blackman_window window_length options =
  [C.block|Tensor* {
    return new Tensor(VariableType::blackman_window($(int64_t window_length), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  blackman_window blackman_window__1
--
blackman_window__1 :: Int64 -> CBool -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
blackman_window__1 window_length periodic options =
  [C.block|Tensor* {
    return new Tensor(VariableType::blackman_window($(int64_t window_length), $(bool periodic), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  bmm bmm
--
bmm :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bmm self mat2 =
  [C.block|Tensor* {
    return new Tensor(VariableType::bmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat2)));
   }|] >>= newForeignPtr deleteTensor


--  bmm_out bmm_out
--
bmm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
bmm_out out self mat2 =
  [C.block|void {
    VariableType::bmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat2));
   }|] >> pure (out)


--  broadcast_tensors broadcast_tensors
--
broadcast_tensors :: Vector (Ptr CTensor) -> IO (Vector (Ptr CTensor))
broadcast_tensors tensors =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in
  [C.block|void* {
    return (void*)new std::vector<Tensor>(VariableType::broadcast_tensors(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size))));
   }|] >>= unVectorTensor


--  cartesian_prod cartesian_prod
--
cartesian_prod :: Vector (Ptr CTensor) -> IO (ForeignPtr CTensor)
cartesian_prod tensors =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in
  [C.block|Tensor* {
    return new Tensor(VariableType::cartesian_prod(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size))));
   }|] >>= newForeignPtr deleteTensor


--  cat cat
--
cat :: Vector (Ptr CTensor) -> Int64 -> IO (ForeignPtr CTensor)
cat tensors dim =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in
  [C.block|Tensor* {
    return new Tensor(VariableType::cat(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size)), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  cat_out cat_out
--
cat_out :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Int64 -> IO (ForeignPtr CTensor)
cat_out out tensors dim =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in
  [C.block|void {
    VariableType::cat_out(*$fptr-ptr:(Tensor* out), pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size)), $(int64_t dim));
   }|] >> pure (out)


--  cauchy_ cauchy_
--
cauchy_ :: ForeignPtr CTensor -> CDouble -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
cauchy_ self median sigma generator =
  [C.block|void {
    VariableType::cauchy_(*$fptr-ptr:(Tensor* self), $(double median), $(double sigma), $(Generator* generator));
   }|] >> pure self


--  cdist cdist
--
cdist :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor)
cdist x1 x2 p =
  [C.block|Tensor* {
    return new Tensor(VariableType::cdist(*$fptr-ptr:(Tensor* x1), *$fptr-ptr:(Tensor* x2), $(double p)));
   }|] >>= newForeignPtr deleteTensor


--  ceil ceil
--
ceil :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ceil self =
  [C.block|Tensor* {
    return new Tensor(VariableType::ceil(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  ceil_ ceil_
--
ceil_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ceil_ self =
  [C.block|void {
    VariableType::ceil_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  ceil_out ceil_out
--
ceil_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ceil_out out self =
  [C.block|void {
    VariableType::ceil_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  celu celu
--
celu :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
celu self alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::celu(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  celu_ celu_
--
celu_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
celu_ self alpha =
  [C.block|void {
    VariableType::celu_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  chain_matmul chain_matmul
--
chain_matmul :: Vector (Ptr CTensor) -> IO (ForeignPtr CTensor)
chain_matmul matrices =  V.unsafeWith matrices $ \matrices__array -> let matrices__size = fromIntegral (V.length matrices) in
  [C.block|Tensor* {
    return new Tensor(VariableType::chain_matmul(pack_tensor_list($(Tensor** matrices__array), $(size_t matrices__size))));
   }|] >>= newForeignPtr deleteTensor


--  cholesky cholesky
--
cholesky :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky self upper =
  [C.block|Tensor* {
    return new Tensor(VariableType::cholesky(*$fptr-ptr:(Tensor* self), $(bool upper)));
   }|] >>= newForeignPtr deleteTensor


--  cholesky_inverse cholesky_inverse
--
cholesky_inverse :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_inverse self upper =
  [C.block|Tensor* {
    return new Tensor(VariableType::cholesky_inverse(*$fptr-ptr:(Tensor* self), $(bool upper)));
   }|] >>= newForeignPtr deleteTensor


--  cholesky_inverse_out cholesky_inverse_out
--
cholesky_inverse_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_inverse_out out self upper =
  [C.block|void {
    VariableType::cholesky_inverse_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(bool upper));
   }|] >> pure (out)


--  cholesky_out cholesky_out
--
cholesky_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_out out self upper =
  [C.block|void {
    VariableType::cholesky_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(bool upper));
   }|] >> pure (out)


--  cholesky_solve cholesky_solve
--
cholesky_solve :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_solve self input2 upper =
  [C.block|Tensor* {
    return new Tensor(VariableType::cholesky_solve(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* input2), $(bool upper)));
   }|] >>= newForeignPtr deleteTensor


--  cholesky_solve_out cholesky_solve_out
--
cholesky_solve_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
cholesky_solve_out out self input2 upper =
  [C.block|void {
    VariableType::cholesky_solve_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* input2), $(bool upper));
   }|] >> pure (out)


--  chunk chunk
--
chunk :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (Vector (Ptr CTensor))
chunk self chunks dim =
  [C.block|void* {
    return (void*)new std::vector<Tensor>(VariableType::chunk(*$fptr-ptr:(Tensor* self), $(int64_t chunks), $(int64_t dim)));
   }|] >>= unVectorTensor


--  clamp clamp
--
clamp :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp self min max =
  [C.block|Tensor* {
    return new Tensor(VariableType::clamp(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max)));
   }|] >>= newForeignPtr deleteTensor


--  clamp_ clamp_
--
clamp_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_ self min max =
  [C.block|void {
    VariableType::clamp_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max));
   }|] >> pure self


--  clamp_max clamp_max
--
clamp_max :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_max self max =
  [C.block|Tensor* {
    return new Tensor(VariableType::clamp_max(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* max)));
   }|] >>= newForeignPtr deleteTensor


--  clamp_max_ clamp_max_
--
clamp_max_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_max_ self max =
  [C.block|void {
    VariableType::clamp_max_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* max));
   }|] >> pure self


--  clamp_max_out clamp_max_out
--
clamp_max_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_max_out out self max =
  [C.block|void {
    VariableType::clamp_max_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* max));
   }|] >> pure (out)


--  clamp_min clamp_min
--
clamp_min :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_min self min =
  [C.block|Tensor* {
    return new Tensor(VariableType::clamp_min(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min)));
   }|] >>= newForeignPtr deleteTensor


--  clamp_min_ clamp_min_
--
clamp_min_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_min_ self min =
  [C.block|void {
    VariableType::clamp_min_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min));
   }|] >> pure self


--  clamp_min_out clamp_min_out
--
clamp_min_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_min_out out self min =
  [C.block|void {
    VariableType::clamp_min_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min));
   }|] >> pure (out)


--  clamp_out clamp_out
--
clamp_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
clamp_out out self min max =
  [C.block|void {
    VariableType::clamp_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max));
   }|] >> pure (out)


--  clone clone
--
clone :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
clone self =
  [C.block|Tensor* {
    return new Tensor(VariableType::clone(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  coalesce coalesce
--
coalesce :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
coalesce self =
  [C.block|Tensor* {
    return new Tensor(VariableType::coalesce(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  col2im col2im
--
col2im :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
col2im self output_size kernel_size dilation padding stride =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in
  [C.block|Tensor* {
    return new Tensor(VariableType::col2im(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size))));
   }|] >>= newForeignPtr deleteTensor


--  col2im_out col2im_out
--
col2im_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
col2im_out out self output_size kernel_size dilation padding stride =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in
  [C.block|void {
    VariableType::col2im_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)));
   }|] >> pure (out)


--  combinations combinations
--
combinations :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
combinations self r with_replacement =
  [C.block|Tensor* {
    return new Tensor(VariableType::combinations(*$fptr-ptr:(Tensor* self), $(int64_t r), $(bool with_replacement)));
   }|] >>= newForeignPtr deleteTensor


--  constant_pad_nd constant_pad_nd
--
constant_pad_nd :: ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
constant_pad_nd self pad value =  V.unsafeWith pad $ \pad__array -> let pad__size = fromIntegral (V.length pad) in
  [C.block|Tensor* {
    return new Tensor(VariableType::constant_pad_nd(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* pad__array), $(size_t pad__size)), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


--  contiguous contiguous
--
contiguous :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
contiguous self memory_format =
  [C.block|Tensor* {
    return new Tensor(VariableType::contiguous(*$fptr-ptr:(Tensor* self), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


--  conv1d conv1d
--
conv1d :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
conv1d input weight bias stride padding dilation groups =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::conv1d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


--  conv2d conv2d
--
conv2d :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
conv2d input weight bias stride padding dilation groups =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::conv2d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


--  conv3d conv3d
--
conv3d :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
conv3d input weight bias stride padding dilation groups =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::conv3d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


--  conv_dilated2d conv_dilated2d
--
conv_dilated2d :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_dilated2d self weight kernel_size bias stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::conv_dilated2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


--  conv_dilated3d conv_dilated3d
--
conv_dilated3d :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_dilated3d self weight kernel_size bias stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::conv_dilated3d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


--  conv_tbc conv_tbc
--
conv_tbc :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
conv_tbc self weight bias pad =
  [C.block|Tensor* {
    return new Tensor(VariableType::conv_tbc(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), $(int64_t pad)));
   }|] >>= newForeignPtr deleteTensor


--  conv_transpose1d conv_transpose1d
--
conv_transpose1d :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose1d input weight bias stride padding output_padding groups dilation =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::conv_transpose1d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


--  conv_transpose2d conv_transpose2d
--
conv_transpose2d :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose2d input weight bias stride padding output_padding groups dilation =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::conv_transpose2d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


--  conv_transpose2d conv_transpose2d__1
--
conv_transpose2d__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose2d__1 self weight kernel_size bias stride padding output_padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::conv_transpose2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


--  conv_transpose2d_out conv_transpose2d_out
--
conv_transpose2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose2d_out out self weight kernel_size bias stride padding output_padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|void {
    VariableType::conv_transpose2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)));
   }|] >> pure (out)


--  conv_transpose3d conv_transpose3d
--
conv_transpose3d :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose3d input weight bias stride padding output_padding groups dilation =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::conv_transpose3d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


--  conv_transpose3d conv_transpose3d__1
--
conv_transpose3d__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose3d__1 self weight kernel_size bias stride padding output_padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::conv_transpose3d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


--  conv_transpose3d_out conv_transpose3d_out
--
conv_transpose3d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
conv_transpose3d_out out self weight kernel_size bias stride padding output_padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|void {
    VariableType::conv_transpose3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)));
   }|] >> pure (out)


--  convolution convolution
--
convolution :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
convolution input weight bias stride padding dilation transposed output_padding groups =  V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::convolution(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool transposed), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


--  copy_ copy_
--
copy_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
copy_ self src non_blocking =
  [C.block|void {
    VariableType::copy_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* src), $(bool non_blocking));
   }|] >> pure self


--  copy_sparse_to_sparse_ copy_sparse_to_sparse_
--
copy_sparse_to_sparse_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
copy_sparse_to_sparse_ self src non_blocking =
  [C.block|void {
    VariableType::copy_sparse_to_sparse_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* src), $(bool non_blocking));
   }|] >> pure self


--  cos cos
--
cos :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cos self =
  [C.block|Tensor* {
    return new Tensor(VariableType::cos(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  cos_ cos_
--
cos_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cos_ self =
  [C.block|void {
    VariableType::cos_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  cos_out cos_out
--
cos_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cos_out out self =
  [C.block|void {
    VariableType::cos_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  cosh cosh
--
cosh :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cosh self =
  [C.block|Tensor* {
    return new Tensor(VariableType::cosh(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  cosh_ cosh_
--
cosh_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cosh_ self =
  [C.block|void {
    VariableType::cosh_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  cosh_out cosh_out
--
cosh_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cosh_out out self =
  [C.block|void {
    VariableType::cosh_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  cosine_embedding_loss cosine_embedding_loss
--
cosine_embedding_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
cosine_embedding_loss input1 input2 target margin reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::cosine_embedding_loss(*$fptr-ptr:(Tensor* input1), *$fptr-ptr:(Tensor* input2), *$fptr-ptr:(Tensor* target), $(double margin), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  cosine_similarity cosine_similarity
--
cosine_similarity :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CDouble -> IO (ForeignPtr CTensor)
cosine_similarity x1 x2 dim eps =
  [C.block|Tensor* {
    return new Tensor(VariableType::cosine_similarity(*$fptr-ptr:(Tensor* x1), *$fptr-ptr:(Tensor* x2), $(int64_t dim), $(double eps)));
   }|] >>= newForeignPtr deleteTensor


--  cross cross
--
cross :: ForeignPtr CTensor -> ForeignPtr CTensor -> Maybe Int64 -> IO (ForeignPtr CTensor)
cross self other dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in
  [C.block|Tensor* {
    return new Tensor(VariableType::cross(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


--  cross_out cross_out
--
cross_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Maybe Int64 -> IO (ForeignPtr CTensor)
cross_out out self other dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in
  [C.block|void {
    VariableType::cross_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt));
   }|] >> pure (out)


--  ctc_loss ctc_loss
--
ctc_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor)
ctc_loss log_probs targets input_lengths target_lengths blank reduction zero_infinity =  V.unsafeWith input_lengths $ \input_lengths__array -> let input_lengths__size = fromIntegral (V.length input_lengths) in V.unsafeWith target_lengths $ \target_lengths__array -> let target_lengths__size = fromIntegral (V.length target_lengths) in
  [C.block|Tensor* {
    return new Tensor(VariableType::ctc_loss(*$fptr-ptr:(Tensor* log_probs), *$fptr-ptr:(Tensor* targets), ArrayRef<int64_t>($(int64_t* input_lengths__array), $(size_t input_lengths__size)), ArrayRef<int64_t>($(int64_t* target_lengths__array), $(size_t target_lengths__size)), $(int64_t blank), $(int64_t reduction), $(bool zero_infinity)));
   }|] >>= newForeignPtr deleteTensor


--  ctc_loss ctc_loss__1
--
ctc_loss__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor)
ctc_loss__1 log_probs targets input_lengths target_lengths blank reduction zero_infinity =
  [C.block|Tensor* {
    return new Tensor(VariableType::ctc_loss(*$fptr-ptr:(Tensor* log_probs), *$fptr-ptr:(Tensor* targets), *$fptr-ptr:(Tensor* input_lengths), *$fptr-ptr:(Tensor* target_lengths), $(int64_t blank), $(int64_t reduction), $(bool zero_infinity)));
   }|] >>= newForeignPtr deleteTensor


--  cudnn_affine_grid_generator cudnn_affine_grid_generator
--
cudnn_affine_grid_generator :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
cudnn_affine_grid_generator theta n c h w =
  [C.block|Tensor* {
    return new Tensor(VariableType::cudnn_affine_grid_generator(*$fptr-ptr:(Tensor* theta), $(int64_t n), $(int64_t c), $(int64_t h), $(int64_t w)));
   }|] >>= newForeignPtr deleteTensor


--  cudnn_batch_norm cudnn_batch_norm
--
cudnn_batch_norm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
cudnn_batch_norm input weight bias running_mean running_var training exponential_average_factor epsilon =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::cudnn_batch_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool training), $(double exponential_average_factor), $(double epsilon)));
   }|] >>= unTupleTensorTensorTensor


--  cudnn_convolution cudnn_convolution
--
cudnn_convolution :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
cudnn_convolution self weight bias padding stride dilation groups benchmark deterministic =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::cudnn_convolution(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic)));
   }|] >>= newForeignPtr deleteTensor


--  cudnn_convolution_transpose cudnn_convolution_transpose
--
cudnn_convolution_transpose :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
cudnn_convolution_transpose self weight bias padding output_padding stride dilation groups benchmark deterministic =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::cudnn_convolution_transpose(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic)));
   }|] >>= newForeignPtr deleteTensor


--  cudnn_grid_sampler cudnn_grid_sampler
--
cudnn_grid_sampler :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
cudnn_grid_sampler self grid =
  [C.block|Tensor* {
    return new Tensor(VariableType::cudnn_grid_sampler(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* grid)));
   }|] >>= newForeignPtr deleteTensor


--  cudnn_is_acceptable cudnn_is_acceptable
--
cudnn_is_acceptable :: ForeignPtr CTensor -> IO (CBool)
cudnn_is_acceptable self =
  [C.block|bool {
    return VariableType::cudnn_is_acceptable(*$fptr-ptr:(Tensor* self));
   }|]


--  cumprod cumprod
--
cumprod :: ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
cumprod self dim dtype =
  [C.block|Tensor* {
    return new Tensor(VariableType::cumprod(*$fptr-ptr:(Tensor* self), $(int64_t dim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  cumprod_out cumprod_out
--
cumprod_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
cumprod_out out self dim dtype =
  [C.block|void {
    VariableType::cumprod_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), static_cast<ScalarType>($(int8_t dtype)));
   }|] >> pure (out)


--  cumsum cumsum
--
cumsum :: ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
cumsum self dim dtype =
  [C.block|Tensor* {
    return new Tensor(VariableType::cumsum(*$fptr-ptr:(Tensor* self), $(int64_t dim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  cumsum_out cumsum_out
--
cumsum_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
cumsum_out out self dim dtype =
  [C.block|void {
    VariableType::cumsum_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), static_cast<ScalarType>($(int8_t dtype)));
   }|] >> pure (out)


--  dense_dim dense_dim
--
dense_dim :: ForeignPtr CTensor -> IO (Int64)
dense_dim self =
  [C.block|int64_t {
    return VariableType::dense_dim(*$fptr-ptr:(Tensor* self));
   }|]


--  dequantize dequantize
--
dequantize :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
dequantize self =
  [C.block|Tensor* {
    return new Tensor(VariableType::dequantize(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  det det
--
det :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
det self =
  [C.block|Tensor* {
    return new Tensor(VariableType::det(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  detach detach
--
detach :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
detach self =
  [C.block|Tensor* {
    return new Tensor(VariableType::detach(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  detach_ detach_
--
detach_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
detach_ self =
  [C.block|void {
    VariableType::detach_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  diag diag
--
diag :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
diag self diagonal =
  [C.block|Tensor* {
    return new Tensor(VariableType::diag(*$fptr-ptr:(Tensor* self), $(int64_t diagonal)));
   }|] >>= newForeignPtr deleteTensor


--  diag_embed diag_embed
--
diag_embed :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
diag_embed self offset dim1 dim2 =
  [C.block|Tensor* {
    return new Tensor(VariableType::diag_embed(*$fptr-ptr:(Tensor* self), $(int64_t offset), $(int64_t dim1), $(int64_t dim2)));
   }|] >>= newForeignPtr deleteTensor


--  diag_out diag_out
--
diag_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
diag_out out self diagonal =
  [C.block|void {
    VariableType::diag_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t diagonal));
   }|] >> pure (out)


--  diagflat diagflat
--
diagflat :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
diagflat self offset =
  [C.block|Tensor* {
    return new Tensor(VariableType::diagflat(*$fptr-ptr:(Tensor* self), $(int64_t offset)));
   }|] >>= newForeignPtr deleteTensor


--  diagonal diagonal
--
diagonal :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
diagonal self offset dim1 dim2 =
  [C.block|Tensor* {
    return new Tensor(VariableType::diagonal(*$fptr-ptr:(Tensor* self), $(int64_t offset), $(int64_t dim1), $(int64_t dim2)));
   }|] >>= newForeignPtr deleteTensor


--  digamma digamma
--
digamma :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
digamma self =
  [C.block|Tensor* {
    return new Tensor(VariableType::digamma(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  digamma_ digamma_
--
digamma_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
digamma_ self =
  [C.block|void {
    VariableType::digamma_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  digamma_out digamma_out
--
digamma_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
digamma_out out self =
  [C.block|void {
    VariableType::digamma_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  dist dist
--
dist :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
dist self other p =
  [C.block|Tensor* {
    return new Tensor(VariableType::dist(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* p)));
   }|] >>= newForeignPtr deleteTensor


--  div div
--
div :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
div self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::div(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  div div__1
--
div__1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
div__1 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::div(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


--  div_ div_
--
div_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
div_ self other =
  [C.block|void {
    VariableType::div_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  div_ div___1
--
div___1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
div___1 self other =
  [C.block|void {
    VariableType::div_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  div_out div_out
--
div_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
div_out out self other =
  [C.block|void {
    VariableType::div_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  dot dot
--
dot :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
dot self tensor =
  [C.block|Tensor* {
    return new Tensor(VariableType::dot(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor)));
   }|] >>= newForeignPtr deleteTensor


--  dot_out dot_out
--
dot_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
dot_out out self tensor =
  [C.block|void {
    VariableType::dot_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor));
   }|] >> pure (out)


--  dropout dropout
--
dropout :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
dropout input p train =
  [C.block|Tensor* {
    return new Tensor(VariableType::dropout(*$fptr-ptr:(Tensor* input), $(double p), $(bool train)));
   }|] >>= newForeignPtr deleteTensor


--  dropout_ dropout_
--
dropout_ :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
dropout_ self p train =
  [C.block|void {
    VariableType::dropout_(*$fptr-ptr:(Tensor* self), $(double p), $(bool train));
   }|] >> pure self


--  eig eig
--
eig :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
eig self eigenvectors =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::eig(*$fptr-ptr:(Tensor* self), $(bool eigenvectors)));
   }|] >>= unTupleTensorTensor


--  eig_out eig_out
--
eig_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
eig_out e v self eigenvectors =
  [C.block|void {
    VariableType::eig_out(*$fptr-ptr:(Tensor* e), *$fptr-ptr:(Tensor* v), *$fptr-ptr:(Tensor* self), $(bool eigenvectors));
   }|] >> pure (e,v)


--  einsum einsum
--
einsum :: Ptr CChar -> Vector (Ptr CTensor) -> IO (ForeignPtr CTensor)
einsum equation tensors =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in
  [C.block|Tensor* {
    return new Tensor(VariableType::einsum($(char* equation), pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size))));
   }|] >>= newForeignPtr deleteTensor


--  elu elu
--
elu :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
elu self alpha scale input_scale =
  [C.block|Tensor* {
    return new Tensor(VariableType::elu(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* alpha), *$fptr-ptr:(Scalar* scale), *$fptr-ptr:(Scalar* input_scale)));
   }|] >>= newForeignPtr deleteTensor


--  elu_ elu_
--
elu_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
elu_ self alpha scale input_scale =
  [C.block|void {
    VariableType::elu_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* alpha), *$fptr-ptr:(Scalar* scale), *$fptr-ptr:(Scalar* input_scale));
   }|] >> pure self


--  elu_out elu_out
--
elu_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
elu_out out self alpha scale input_scale =
  [C.block|void {
    VariableType::elu_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* alpha), *$fptr-ptr:(Scalar* scale), *$fptr-ptr:(Scalar* input_scale));
   }|] >> pure (out)


--  embedding embedding
--
embedding :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
embedding weight indices padding_idx scale_grad_by_freq sparse =
  [C.block|Tensor* {
    return new Tensor(VariableType::embedding(*$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* indices), $(int64_t padding_idx), $(bool scale_grad_by_freq), $(bool sparse)));
   }|] >>= newForeignPtr deleteTensor


--  embedding_bag embedding_bag
--
embedding_bag :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> Int64 -> CBool -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
embedding_bag weight indices offsets scale_grad_by_freq mode sparse per_sample_weights =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,Tensor>(VariableType::embedding_bag(*$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* offsets), $(bool scale_grad_by_freq), $(int64_t mode), $(bool sparse), *$fptr-ptr:(Tensor* per_sample_weights)));
   }|] >>= unTupleTensorTensorTensorTensor


--  embedding_renorm_ embedding_renorm_
--
embedding_renorm_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> IO (ForeignPtr CTensor)
embedding_renorm_ self indices max_norm norm_type =
  [C.block|void {
    VariableType::embedding_renorm_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* indices), $(double max_norm), $(double norm_type));
   }|] >> pure self


--  empty empty
--
empty :: Vector Int64 -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
empty size options memory_format =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::empty(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


--  empty_like empty_like
--
empty_like :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
empty_like self =
  [C.block|Tensor* {
    return new Tensor(VariableType::empty_like(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  empty_like empty_like__1
--
empty_like__1 :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> Int8 -> IO (ForeignPtr CTensor)
empty_like__1 self options memory_format =
  [C.block|Tensor* {
    return new Tensor(VariableType::empty_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options), static_cast<MemoryFormat>($(int8_t memory_format))));
   }|] >>= newForeignPtr deleteTensor


--  empty_out empty_out
--
empty_out :: ForeignPtr CTensor -> Vector Int64 -> Int8 -> IO (ForeignPtr CTensor)
empty_out out size memory_format =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::empty_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), static_cast<MemoryFormat>($(int8_t memory_format)));
   }|] >> pure (out)


--  empty_strided empty_strided
--
empty_strided :: Vector Int64 -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
empty_strided size stride options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in
  [C.block|Tensor* {
    return new Tensor(VariableType::empty_strided(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  eq eq
--
eq :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
eq self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::eq(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


--  eq eq__1
--
eq__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
eq__1 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::eq(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  eq_ eq_
--
eq_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
eq_ self other =
  [C.block|void {
    VariableType::eq_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  eq_ eq___1
--
eq___1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
eq___1 self other =
  [C.block|void {
    VariableType::eq_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  eq_out eq_out
--
eq_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
eq_out out self other =
  [C.block|void {
    VariableType::eq_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


--  eq_out eq_out__1
--
eq_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
eq_out__1 out self other =
  [C.block|void {
    VariableType::eq_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  equal equal
--
equal :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (CBool)
equal self other =
  [C.block|bool {
    return VariableType::equal(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|]


--  erf erf
--
erf :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erf self =
  [C.block|Tensor* {
    return new Tensor(VariableType::erf(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  erf_ erf_
--
erf_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erf_ self =
  [C.block|void {
    VariableType::erf_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  erf_out erf_out
--
erf_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erf_out out self =
  [C.block|void {
    VariableType::erf_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  erfc erfc
--
erfc :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfc self =
  [C.block|Tensor* {
    return new Tensor(VariableType::erfc(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  erfc_ erfc_
--
erfc_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfc_ self =
  [C.block|void {
    VariableType::erfc_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  erfc_out erfc_out
--
erfc_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfc_out out self =
  [C.block|void {
    VariableType::erfc_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  erfinv erfinv
--
erfinv :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfinv self =
  [C.block|Tensor* {
    return new Tensor(VariableType::erfinv(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  erfinv_ erfinv_
--
erfinv_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfinv_ self =
  [C.block|void {
    VariableType::erfinv_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  erfinv_out erfinv_out
--
erfinv_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
erfinv_out out self =
  [C.block|void {
    VariableType::erfinv_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  exp exp
--
exp :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
exp self =
  [C.block|Tensor* {
    return new Tensor(VariableType::exp(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  exp_ exp_
--
exp_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
exp_ self =
  [C.block|void {
    VariableType::exp_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  exp_out exp_out
--
exp_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
exp_out out self =
  [C.block|void {
    VariableType::exp_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  expand expand
--
expand :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
expand self size implicit =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::expand(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(bool implicit)));
   }|] >>= newForeignPtr deleteTensor


--  expand_as expand_as
--
expand_as :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
expand_as self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::expand_as(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  expm1 expm1
--
expm1 :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
expm1 self =
  [C.block|Tensor* {
    return new Tensor(VariableType::expm1(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  expm1_ expm1_
--
expm1_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
expm1_ self =
  [C.block|void {
    VariableType::expm1_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  expm1_out expm1_out
--
expm1_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
expm1_out out self =
  [C.block|void {
    VariableType::expm1_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  exponential_ exponential_
--
exponential_ :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
exponential_ self lambd generator =
  [C.block|void {
    VariableType::exponential_(*$fptr-ptr:(Tensor* self), $(double lambd), $(Generator* generator));
   }|] >> pure self


--  eye eye
--
eye :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
eye n options =
  [C.block|Tensor* {
    return new Tensor(VariableType::eye($(int64_t n), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  eye eye__1
--
eye__1 :: Int64 -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
eye__1 n m options =
  [C.block|Tensor* {
    return new Tensor(VariableType::eye($(int64_t n), $(int64_t m), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  eye_out eye_out
--
eye_out :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
eye_out out n =
  [C.block|void {
    VariableType::eye_out(*$fptr-ptr:(Tensor* out), $(int64_t n));
   }|] >> pure (out)


--  eye_out eye_out__1
--
eye_out__1 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
eye_out__1 out n m =
  [C.block|void {
    VariableType::eye_out(*$fptr-ptr:(Tensor* out), $(int64_t n), $(int64_t m));
   }|] >> pure (out)


--  fake_quantize_per_tensor_affine fake_quantize_per_tensor_affine
--
fake_quantize_per_tensor_affine :: ForeignPtr CTensor -> CDouble -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
fake_quantize_per_tensor_affine self scale zero_point quant_min quant_max =
  [C.block|Tensor* {
    return new Tensor(VariableType::fake_quantize_per_tensor_affine(*$fptr-ptr:(Tensor* self), $(double scale), $(int64_t zero_point), $(int64_t quant_min), $(int64_t quant_max)));
   }|] >>= newForeignPtr deleteTensor


--  fbgemm_is_cpu_supported fbgemm_is_cpu_supported
--
fbgemm_is_cpu_supported :: IO (CBool)
fbgemm_is_cpu_supported  =
  [C.block|bool {
    return VariableType::fbgemm_is_cpu_supported();
   }|]


--  fbgemm_linear_fp16_weight fbgemm_linear_fp16_weight
--
fbgemm_linear_fp16_weight :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fbgemm_linear_fp16_weight input packed_weight bias =
  [C.block|Tensor* {
    return new Tensor(VariableType::fbgemm_linear_fp16_weight(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* packed_weight), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


--  fbgemm_linear_int8_weight fbgemm_linear_int8_weight
--
fbgemm_linear_int8_weight :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fbgemm_linear_int8_weight input weight packed col_offsets weight_scale weight_zero_point bias =
  [C.block|Tensor* {
    return new Tensor(VariableType::fbgemm_linear_int8_weight(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* packed), *$fptr-ptr:(Tensor* col_offsets), *$fptr-ptr:(Scalar* weight_scale), *$fptr-ptr:(Scalar* weight_zero_point), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


--  fbgemm_linear_quantize_weight fbgemm_linear_quantize_weight
--
fbgemm_linear_quantize_weight :: ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor, CDouble, Int64)
fbgemm_linear_quantize_weight input =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,double,int64_t>(VariableType::fbgemm_linear_quantize_weight(*$fptr-ptr:(Tensor* input)));
   }|] >>= unTupleTensorTensorDoubleInt64


--  fbgemm_pack_gemm_matrix_fp16 fbgemm_pack_gemm_matrix_fp16
--
fbgemm_pack_gemm_matrix_fp16 :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fbgemm_pack_gemm_matrix_fp16 input =
  [C.block|Tensor* {
    return new Tensor(VariableType::fbgemm_pack_gemm_matrix_fp16(*$fptr-ptr:(Tensor* input)));
   }|] >>= newForeignPtr deleteTensor


--  fbgemm_pack_quantized_matrix fbgemm_pack_quantized_matrix
--
fbgemm_pack_quantized_matrix :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
fbgemm_pack_quantized_matrix input k n =
  [C.block|Tensor* {
    return new Tensor(VariableType::fbgemm_pack_quantized_matrix(*$fptr-ptr:(Tensor* input), $(int64_t k), $(int64_t n)));
   }|] >>= newForeignPtr deleteTensor


--  feature_alpha_dropout feature_alpha_dropout
--
feature_alpha_dropout :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
feature_alpha_dropout input p train =
  [C.block|Tensor* {
    return new Tensor(VariableType::feature_alpha_dropout(*$fptr-ptr:(Tensor* input), $(double p), $(bool train)));
   }|] >>= newForeignPtr deleteTensor


--  feature_alpha_dropout_ feature_alpha_dropout_
--
feature_alpha_dropout_ :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
feature_alpha_dropout_ self p train =
  [C.block|void {
    VariableType::feature_alpha_dropout_(*$fptr-ptr:(Tensor* self), $(double p), $(bool train));
   }|] >> pure self


--  feature_dropout feature_dropout
--
feature_dropout :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
feature_dropout input p train =
  [C.block|Tensor* {
    return new Tensor(VariableType::feature_dropout(*$fptr-ptr:(Tensor* input), $(double p), $(bool train)));
   }|] >>= newForeignPtr deleteTensor


--  feature_dropout_ feature_dropout_
--
feature_dropout_ :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
feature_dropout_ self p train =
  [C.block|void {
    VariableType::feature_dropout_(*$fptr-ptr:(Tensor* self), $(double p), $(bool train));
   }|] >> pure self


--  fft fft
--
fft :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
fft self signal_ndim normalized =
  [C.block|Tensor* {
    return new Tensor(VariableType::fft(*$fptr-ptr:(Tensor* self), $(int64_t signal_ndim), $(bool normalized)));
   }|] >>= newForeignPtr deleteTensor


--  fill_ fill_
--
fill_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
fill_ self value =
  [C.block|void {
    VariableType::fill_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


--  fill_ fill___1
--
fill___1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fill___1 self value =
  [C.block|void {
    VariableType::fill_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* value));
   }|] >> pure self


--  fill_diagonal_ fill_diagonal_
--
fill_diagonal_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> CBool -> IO (ForeignPtr CTensor)
fill_diagonal_ self fill_value wrap =
  [C.block|void {
    VariableType::fill_diagonal_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* fill_value), $(bool wrap));
   }|] >> pure self


--  flatten flatten
--
flatten :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
flatten self start_dim end_dim =
  [C.block|Tensor* {
    return new Tensor(VariableType::flatten(*$fptr-ptr:(Tensor* self), $(int64_t start_dim), $(int64_t end_dim)));
   }|] >>= newForeignPtr deleteTensor


--  flip flip
--
flip :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
flip self dims =  V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in
  [C.block|Tensor* {
    return new Tensor(VariableType::flip(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


--  floor floor
--
floor :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
floor self =
  [C.block|Tensor* {
    return new Tensor(VariableType::floor(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  floor_ floor_
--
floor_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
floor_ self =
  [C.block|void {
    VariableType::floor_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  floor_out floor_out
--
floor_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
floor_out out self =
  [C.block|void {
    VariableType::floor_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  fmod fmod
--
fmod :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
fmod self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::fmod(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


--  fmod fmod__1
--
fmod__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fmod__1 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::fmod(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  fmod_ fmod_
--
fmod_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
fmod_ self other =
  [C.block|void {
    VariableType::fmod_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  fmod_ fmod___1
--
fmod___1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fmod___1 self other =
  [C.block|void {
    VariableType::fmod_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  fmod_out fmod_out
--
fmod_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
fmod_out out self other =
  [C.block|void {
    VariableType::fmod_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


--  fmod_out fmod_out__1
--
fmod_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
fmod_out__1 out self other =
  [C.block|void {
    VariableType::fmod_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  frac frac
--
frac :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
frac self =
  [C.block|Tensor* {
    return new Tensor(VariableType::frac(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  frac_ frac_
--
frac_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
frac_ self =
  [C.block|void {
    VariableType::frac_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  frac_out frac_out
--
frac_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
frac_out out self =
  [C.block|void {
    VariableType::frac_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  fractional_max_pool2d fractional_max_pool2d
--
fractional_max_pool2d :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
fractional_max_pool2d self kernel_size output_size random_samples =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::fractional_max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), *$fptr-ptr:(Tensor* random_samples)));
   }|] >>= unTupleTensorTensor


--  fractional_max_pool2d_out fractional_max_pool2d_out
--
fractional_max_pool2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
fractional_max_pool2d_out output indices self kernel_size output_size random_samples =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::fractional_max_pool2d_out(*$fptr-ptr:(Tensor* output), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), *$fptr-ptr:(Tensor* random_samples));
   }|] >> pure (output,indices)


--  fractional_max_pool3d fractional_max_pool3d
--
fractional_max_pool3d :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
fractional_max_pool3d self kernel_size output_size random_samples =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::fractional_max_pool3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), *$fptr-ptr:(Tensor* random_samples)));
   }|] >>= unTupleTensorTensor


--  fractional_max_pool3d_out fractional_max_pool3d_out
--
fractional_max_pool3d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
fractional_max_pool3d_out output indices self kernel_size output_size random_samples =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::fractional_max_pool3d_out(*$fptr-ptr:(Tensor* output), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), *$fptr-ptr:(Tensor* random_samples));
   }|] >> pure (output,indices)


--  frobenius_norm frobenius_norm
--
frobenius_norm :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
frobenius_norm self =
  [C.block|Tensor* {
    return new Tensor(VariableType::frobenius_norm(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  frobenius_norm frobenius_norm__1
--
frobenius_norm__1 :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
frobenius_norm__1 self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::frobenius_norm(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  frobenius_norm_out frobenius_norm_out
--
frobenius_norm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
frobenius_norm_out out self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|void {
    VariableType::frobenius_norm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim));
   }|] >> pure (out)


--  from_file from_file
--
from_file :: Ptr CChar -> CBool -> Maybe Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
from_file filename shared size options =  let (size__is_present, size__value) = splitMaybe size 0 in
  [C.block|Tensor* {
    return new Tensor(VariableType::from_file($(char* filename), $(bool shared), ($(bool size__is_present) ? make_optional($(int64_t size__value)) : c10::nullopt), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  full full
--
full :: Vector Int64 -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
full size fill_value options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::full(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(Scalar* fill_value), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  full_like full_like
--
full_like :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
full_like self fill_value =
  [C.block|Tensor* {
    return new Tensor(VariableType::full_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* fill_value)));
   }|] >>= newForeignPtr deleteTensor


--  full_like full_like__1
--
full_like__1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
full_like__1 self fill_value options =
  [C.block|Tensor* {
    return new Tensor(VariableType::full_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* fill_value), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  full_out full_out
--
full_out :: ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
full_out out size fill_value =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::full_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(Scalar* fill_value));
   }|] >> pure (out)


--  gather gather
--
gather :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
gather self dim index sparse_grad =
  [C.block|Tensor* {
    return new Tensor(VariableType::gather(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), $(bool sparse_grad)));
   }|] >>= newForeignPtr deleteTensor


--  gather_out gather_out
--
gather_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
gather_out out self dim index sparse_grad =
  [C.block|void {
    VariableType::gather_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), $(bool sparse_grad));
   }|] >> pure (out)


--  ge ge
--
ge :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ge self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::ge(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


--  ge ge__1
--
ge__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ge__1 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::ge(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  ge_ ge_
--
ge_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ge_ self other =
  [C.block|void {
    VariableType::ge_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  ge_ ge___1
--
ge___1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ge___1 self other =
  [C.block|void {
    VariableType::ge_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  ge_out ge_out
--
ge_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ge_out out self other =
  [C.block|void {
    VariableType::ge_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


--  ge_out ge_out__1
--
ge_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ge_out__1 out self other =
  [C.block|void {
    VariableType::ge_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  gelu gelu
--
gelu :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
gelu self =
  [C.block|Tensor* {
    return new Tensor(VariableType::gelu(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  geometric_ geometric_
--
geometric_ :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
geometric_ self p generator =
  [C.block|void {
    VariableType::geometric_(*$fptr-ptr:(Tensor* self), $(double p), $(Generator* generator));
   }|] >> pure self


--  geqrf geqrf
--
geqrf :: ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
geqrf self =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::geqrf(*$fptr-ptr:(Tensor* self)));
   }|] >>= unTupleTensorTensor


--  geqrf_out geqrf_out
--
geqrf_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
geqrf_out a tau self =
  [C.block|void {
    VariableType::geqrf_out(*$fptr-ptr:(Tensor* a), *$fptr-ptr:(Tensor* tau), *$fptr-ptr:(Tensor* self));
   }|] >> pure (a,tau)


--  ger ger
--
ger :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ger self vec2 =
  [C.block|Tensor* {
    return new Tensor(VariableType::ger(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec2)));
   }|] >>= newForeignPtr deleteTensor


--  ger_out ger_out
--
ger_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ger_out out self vec2 =
  [C.block|void {
    VariableType::ger_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec2));
   }|] >> pure (out)


--  glu glu
--
glu :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
glu self dim =
  [C.block|Tensor* {
    return new Tensor(VariableType::glu(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  glu_out glu_out
--
glu_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
glu_out out self dim =
  [C.block|void {
    VariableType::glu_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|] >> pure (out)


--  grid_sampler grid_sampler
--
grid_sampler :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
grid_sampler input grid interpolation_mode padding_mode =
  [C.block|Tensor* {
    return new Tensor(VariableType::grid_sampler(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* grid), $(int64_t interpolation_mode), $(int64_t padding_mode)));
   }|] >>= newForeignPtr deleteTensor


--  grid_sampler_2d grid_sampler_2d
--
grid_sampler_2d :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
grid_sampler_2d input grid interpolation_mode padding_mode =
  [C.block|Tensor* {
    return new Tensor(VariableType::grid_sampler_2d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* grid), $(int64_t interpolation_mode), $(int64_t padding_mode)));
   }|] >>= newForeignPtr deleteTensor


--  grid_sampler_3d grid_sampler_3d
--
grid_sampler_3d :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
grid_sampler_3d input grid interpolation_mode padding_mode =
  [C.block|Tensor* {
    return new Tensor(VariableType::grid_sampler_3d(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* grid), $(int64_t interpolation_mode), $(int64_t padding_mode)));
   }|] >>= newForeignPtr deleteTensor


--  group_norm group_norm
--
group_norm :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
group_norm input num_groups weight bias eps cudnn_enabled =
  [C.block|Tensor* {
    return new Tensor(VariableType::group_norm(*$fptr-ptr:(Tensor* input), $(int64_t num_groups), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), $(double eps), $(bool cudnn_enabled)));
   }|] >>= newForeignPtr deleteTensor


--  gru gru
--
gru :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
gru input hx params has_biases num_layers dropout train bidirectional batch_first =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::gru(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), $(bool batch_first)));
   }|] >>= unTupleTensorTensor


--  gru gru__1
--
gru__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
gru__1 dataX batch_sizes hx params has_biases num_layers dropout train bidirectional =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::gru(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional)));
   }|] >>= unTupleTensorTensor


--  gru_cell gru_cell
--
gru_cell :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
gru_cell input hx w_ih w_hh b_ih b_hh =
  [C.block|Tensor* {
    return new Tensor(VariableType::gru_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh)));
   }|] >>= newForeignPtr deleteTensor


--  gt gt
--
gt :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
gt self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::gt(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


--  gt gt__1
--
gt__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
gt__1 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::gt(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  gt_ gt_
--
gt_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
gt_ self other =
  [C.block|void {
    VariableType::gt_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  gt_ gt___1
--
gt___1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
gt___1 self other =
  [C.block|void {
    VariableType::gt_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  gt_out gt_out
--
gt_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
gt_out out self other =
  [C.block|void {
    VariableType::gt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


--  gt_out gt_out__1
--
gt_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
gt_out__1 out self other =
  [C.block|void {
    VariableType::gt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  hamming_window hamming_window
--
hamming_window :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hamming_window window_length options =
  [C.block|Tensor* {
    return new Tensor(VariableType::hamming_window($(int64_t window_length), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  hamming_window hamming_window__1
--
hamming_window__1 :: Int64 -> CBool -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hamming_window__1 window_length periodic options =
  [C.block|Tensor* {
    return new Tensor(VariableType::hamming_window($(int64_t window_length), $(bool periodic), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  hamming_window hamming_window__2
--
hamming_window__2 :: Int64 -> CBool -> CDouble -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hamming_window__2 window_length periodic alpha options =
  [C.block|Tensor* {
    return new Tensor(VariableType::hamming_window($(int64_t window_length), $(bool periodic), $(double alpha), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  hamming_window hamming_window__3
--
hamming_window__3 :: Int64 -> CBool -> CDouble -> CDouble -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hamming_window__3 window_length periodic alpha beta options =
  [C.block|Tensor* {
    return new Tensor(VariableType::hamming_window($(int64_t window_length), $(bool periodic), $(double alpha), $(double beta), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  hann_window hann_window
--
hann_window :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hann_window window_length options =
  [C.block|Tensor* {
    return new Tensor(VariableType::hann_window($(int64_t window_length), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  hann_window hann_window__1
--
hann_window__1 :: Int64 -> CBool -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
hann_window__1 window_length periodic options =
  [C.block|Tensor* {
    return new Tensor(VariableType::hann_window($(int64_t window_length), $(bool periodic), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  hardshrink hardshrink
--
hardshrink :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
hardshrink self lambd =
  [C.block|Tensor* {
    return new Tensor(VariableType::hardshrink(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* lambd)));
   }|] >>= newForeignPtr deleteTensor


--  hardtanh hardtanh
--
hardtanh :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
hardtanh self min_val max_val =
  [C.block|Tensor* {
    return new Tensor(VariableType::hardtanh(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min_val), *$fptr-ptr:(Scalar* max_val)));
   }|] >>= newForeignPtr deleteTensor


--  hardtanh_ hardtanh_
--
hardtanh_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
hardtanh_ self min_val max_val =
  [C.block|void {
    VariableType::hardtanh_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min_val), *$fptr-ptr:(Scalar* max_val));
   }|] >> pure self


--  hardtanh_out hardtanh_out
--
hardtanh_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
hardtanh_out out self min_val max_val =
  [C.block|void {
    VariableType::hardtanh_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* min_val), *$fptr-ptr:(Scalar* max_val));
   }|] >> pure (out)


--  hinge_embedding_loss hinge_embedding_loss
--
hinge_embedding_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
hinge_embedding_loss self target margin reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::hinge_embedding_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(double margin), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  histc histc
--
histc :: ForeignPtr CTensor -> Int64 -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
histc self bins min max =
  [C.block|Tensor* {
    return new Tensor(VariableType::histc(*$fptr-ptr:(Tensor* self), $(int64_t bins), *$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max)));
   }|] >>= newForeignPtr deleteTensor


--  histc_out histc_out
--
histc_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
histc_out out self bins min max =
  [C.block|void {
    VariableType::histc_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t bins), *$fptr-ptr:(Scalar* min), *$fptr-ptr:(Scalar* max));
   }|] >> pure (out)


--  hspmm hspmm
--
hspmm :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
hspmm mat1 mat2 =
  [C.block|Tensor* {
    return new Tensor(VariableType::hspmm(*$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2)));
   }|] >>= newForeignPtr deleteTensor


--  hspmm_out hspmm_out
--
hspmm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
hspmm_out out mat1 mat2 =
  [C.block|void {
    VariableType::hspmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2));
   }|] >> pure (out)


--  ifft ifft
--
ifft :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor)
ifft self signal_ndim normalized =
  [C.block|Tensor* {
    return new Tensor(VariableType::ifft(*$fptr-ptr:(Tensor* self), $(int64_t signal_ndim), $(bool normalized)));
   }|] >>= newForeignPtr deleteTensor


--  im2col im2col
--
im2col :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
im2col self kernel_size dilation padding stride =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in
  [C.block|Tensor* {
    return new Tensor(VariableType::im2col(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size))));
   }|] >>= newForeignPtr deleteTensor


--  im2col_out im2col_out
--
im2col_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
im2col_out out self kernel_size dilation padding stride =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in
  [C.block|void {
    VariableType::im2col_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)));
   }|] >> pure (out)


--  index index
--
index :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> IO (ForeignPtr CTensor)
index self indices =  V.unsafeWith indices $ \indices__array -> let indices__size = fromIntegral (V.length indices) in
  [C.block|Tensor* {
    return new Tensor(VariableType::index(*$fptr-ptr:(Tensor* self), pack_tensor_list($(Tensor** indices__array), $(size_t indices__size))));
   }|] >>= newForeignPtr deleteTensor


--  index_add index_add
--
index_add :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_add self dim index source =
  [C.block|Tensor* {
    return new Tensor(VariableType::index_add(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source)));
   }|] >>= newForeignPtr deleteTensor


--  index_add_ index_add_
--
index_add_ :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_add_ self dim index source =
  [C.block|void {
    VariableType::index_add_(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source));
   }|] >> pure self


--  index_copy index_copy
--
index_copy :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_copy self dim index source =
  [C.block|Tensor* {
    return new Tensor(VariableType::index_copy(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source)));
   }|] >>= newForeignPtr deleteTensor


--  index_copy_ index_copy_
--
index_copy_ :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_copy_ self dim index source =
  [C.block|void {
    VariableType::index_copy_(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source));
   }|] >> pure self


--  index_fill index_fill
--
index_fill :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
index_fill self dim index value =
  [C.block|Tensor* {
    return new Tensor(VariableType::index_fill(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


--  index_fill index_fill__1
--
index_fill__1 :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_fill__1 self dim index value =
  [C.block|Tensor* {
    return new Tensor(VariableType::index_fill(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* value)));
   }|] >>= newForeignPtr deleteTensor


--  index_fill_ index_fill_
--
index_fill_ :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
index_fill_ self dim index value =
  [C.block|void {
    VariableType::index_fill_(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


--  index_fill_ index_fill___1
--
index_fill___1 :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_fill___1 self dim index value =
  [C.block|void {
    VariableType::index_fill_(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* value));
   }|] >> pure self


--  index_put index_put
--
index_put :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
index_put self indices values accumulate =  V.unsafeWith indices $ \indices__array -> let indices__size = fromIntegral (V.length indices) in
  [C.block|Tensor* {
    return new Tensor(VariableType::index_put(*$fptr-ptr:(Tensor* self), pack_tensor_list($(Tensor** indices__array), $(size_t indices__size)), *$fptr-ptr:(Tensor* values), $(bool accumulate)));
   }|] >>= newForeignPtr deleteTensor


--  index_put_ index_put_
--
index_put_ :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
index_put_ self indices values accumulate =  V.unsafeWith indices $ \indices__array -> let indices__size = fromIntegral (V.length indices) in
  [C.block|void {
    VariableType::index_put_(*$fptr-ptr:(Tensor* self), pack_tensor_list($(Tensor** indices__array), $(size_t indices__size)), *$fptr-ptr:(Tensor* values), $(bool accumulate));
   }|] >> pure self


--  index_select index_select
--
index_select :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_select self dim index =
  [C.block|Tensor* {
    return new Tensor(VariableType::index_select(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index)));
   }|] >>= newForeignPtr deleteTensor


--  index_select_out index_select_out
--
index_select_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
index_select_out out self dim index =
  [C.block|void {
    VariableType::index_select_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index));
   }|] >> pure (out)


--  indices indices
--
indices :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
indices self =
  [C.block|Tensor* {
    return new Tensor(VariableType::indices(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  instance_norm instance_norm
--
instance_norm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> CBool -> IO (ForeignPtr CTensor)
instance_norm input weight bias running_mean running_var use_input_stats momentum eps cudnn_enabled =
  [C.block|Tensor* {
    return new Tensor(VariableType::instance_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool use_input_stats), $(double momentum), $(double eps), $(bool cudnn_enabled)));
   }|] >>= newForeignPtr deleteTensor


--  int_repr int_repr
--
int_repr :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
int_repr self =
  [C.block|Tensor* {
    return new Tensor(VariableType::int_repr(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  inverse inverse
--
inverse :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
inverse self =
  [C.block|Tensor* {
    return new Tensor(VariableType::inverse(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  inverse_out inverse_out
--
inverse_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
inverse_out out self =
  [C.block|void {
    VariableType::inverse_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  irfft irfft
--
irfft :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> Vector Int64 -> IO (ForeignPtr CTensor)
irfft self signal_ndim normalized onesided signal_sizes =  V.unsafeWith signal_sizes $ \signal_sizes__array -> let signal_sizes__size = fromIntegral (V.length signal_sizes) in
  [C.block|Tensor* {
    return new Tensor(VariableType::irfft(*$fptr-ptr:(Tensor* self), $(int64_t signal_ndim), $(bool normalized), $(bool onesided), ArrayRef<int64_t>($(int64_t* signal_sizes__array), $(size_t signal_sizes__size))));
   }|] >>= newForeignPtr deleteTensor


--  is_coalesced is_coalesced
--
is_coalesced :: ForeignPtr CTensor -> IO (CBool)
is_coalesced self =
  [C.block|bool {
    return VariableType::is_coalesced(*$fptr-ptr:(Tensor* self));
   }|]


--  is_complex is_complex
--
is_complex :: ForeignPtr CTensor -> IO (CBool)
is_complex self =
  [C.block|bool {
    return VariableType::is_complex(*$fptr-ptr:(Tensor* self));
   }|]


--  is_distributed is_distributed
--
is_distributed :: ForeignPtr CTensor -> IO (CBool)
is_distributed self =
  [C.block|bool {
    return VariableType::is_distributed(*$fptr-ptr:(Tensor* self));
   }|]


--  is_floating_point is_floating_point
--
is_floating_point :: ForeignPtr CTensor -> IO (CBool)
is_floating_point self =
  [C.block|bool {
    return VariableType::is_floating_point(*$fptr-ptr:(Tensor* self));
   }|]


--  is_nonzero is_nonzero
--
is_nonzero :: ForeignPtr CTensor -> IO (CBool)
is_nonzero self =
  [C.block|bool {
    return VariableType::is_nonzero(*$fptr-ptr:(Tensor* self));
   }|]


--  is_same_size is_same_size
--
is_same_size :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (CBool)
is_same_size self other =
  [C.block|bool {
    return VariableType::is_same_size(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|]


--  is_set_to is_set_to
--
is_set_to :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (CBool)
is_set_to self tensor =
  [C.block|bool {
    return VariableType::is_set_to(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* tensor));
   }|]


--  is_signed is_signed
--
is_signed :: ForeignPtr CTensor -> IO (CBool)
is_signed self =
  [C.block|bool {
    return VariableType::is_signed(*$fptr-ptr:(Tensor* self));
   }|]


--  isclose isclose
--
isclose :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> CBool -> IO (ForeignPtr CTensor)
isclose self other rtol atol equal_nan =
  [C.block|Tensor* {
    return new Tensor(VariableType::isclose(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), $(double rtol), $(double atol), $(bool equal_nan)));
   }|] >>= newForeignPtr deleteTensor


--  isnan isnan
--
isnan :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
isnan self =
  [C.block|Tensor* {
    return new Tensor(VariableType::isnan(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  item item
--
item :: ForeignPtr CTensor -> IO (ForeignPtr CScalar)
item self =
  [C.block|Scalar* {
    return new Scalar(VariableType::item(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteScalar'


--  kl_div kl_div
--
kl_div :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
kl_div self target reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::kl_div(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  kthvalue kthvalue
--
kthvalue :: ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
kthvalue self k dim keepdim =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::kthvalue(*$fptr-ptr:(Tensor* self), $(int64_t k), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


--  kthvalue_out kthvalue_out
--
kthvalue_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
kthvalue_out values indices self k dim keepdim =
  [C.block|void {
    VariableType::kthvalue_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t k), $(int64_t dim), $(bool keepdim));
   }|] >> pure (values,indices)


--  l1_loss l1_loss
--
l1_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
l1_loss self target reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::l1_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  l1_loss_out l1_loss_out
--
l1_loss_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
l1_loss_out out self target reduction =
  [C.block|void {
    VariableType::l1_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction));
   }|] >> pure (out)


--  layer_norm layer_norm
--
layer_norm :: ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
layer_norm input normalized_shape weight bias eps cudnn_enable =  V.unsafeWith normalized_shape $ \normalized_shape__array -> let normalized_shape__size = fromIntegral (V.length normalized_shape) in
  [C.block|Tensor* {
    return new Tensor(VariableType::layer_norm(*$fptr-ptr:(Tensor* input), ArrayRef<int64_t>($(int64_t* normalized_shape__array), $(size_t normalized_shape__size)), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), $(double eps), $(bool cudnn_enable)));
   }|] >>= newForeignPtr deleteTensor


--  le le
--
le :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
le self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::le(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


--  le le__1
--
le__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
le__1 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::le(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  le_ le_
--
le_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
le_ self other =
  [C.block|void {
    VariableType::le_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  le_ le___1
--
le___1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
le___1 self other =
  [C.block|void {
    VariableType::le_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  le_out le_out
--
le_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
le_out out self other =
  [C.block|void {
    VariableType::le_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


--  le_out le_out__1
--
le_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
le_out__1 out self other =
  [C.block|void {
    VariableType::le_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  leaky_relu leaky_relu
--
leaky_relu :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
leaky_relu self negative_slope =
  [C.block|Tensor* {
    return new Tensor(VariableType::leaky_relu(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* negative_slope)));
   }|] >>= newForeignPtr deleteTensor


--  leaky_relu_ leaky_relu_
--
leaky_relu_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
leaky_relu_ self negative_slope =
  [C.block|void {
    VariableType::leaky_relu_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* negative_slope));
   }|] >> pure self


--  leaky_relu_out leaky_relu_out
--
leaky_relu_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
leaky_relu_out out self negative_slope =
  [C.block|void {
    VariableType::leaky_relu_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* negative_slope));
   }|] >> pure (out)


--  lerp lerp
--
lerp :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lerp self end weight =
  [C.block|Tensor* {
    return new Tensor(VariableType::lerp(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* end), *$fptr-ptr:(Scalar* weight)));
   }|] >>= newForeignPtr deleteTensor


--  lerp lerp__1
--
lerp__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lerp__1 self end weight =
  [C.block|Tensor* {
    return new Tensor(VariableType::lerp(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* end), *$fptr-ptr:(Tensor* weight)));
   }|] >>= newForeignPtr deleteTensor


--  lerp_ lerp_
--
lerp_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lerp_ self end weight =
  [C.block|void {
    VariableType::lerp_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* end), *$fptr-ptr:(Scalar* weight));
   }|] >> pure self


--  lerp_ lerp___1
--
lerp___1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lerp___1 self end weight =
  [C.block|void {
    VariableType::lerp_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* end), *$fptr-ptr:(Tensor* weight));
   }|] >> pure self


--  lerp_out lerp_out
--
lerp_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lerp_out out self end weight =
  [C.block|void {
    VariableType::lerp_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* end), *$fptr-ptr:(Scalar* weight));
   }|] >> pure (out)


--  lerp_out lerp_out__1
--
lerp_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lerp_out__1 out self end weight =
  [C.block|void {
    VariableType::lerp_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* end), *$fptr-ptr:(Tensor* weight));
   }|] >> pure (out)


--  lgamma lgamma
--
lgamma :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lgamma self =
  [C.block|Tensor* {
    return new Tensor(VariableType::lgamma(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  lgamma_ lgamma_
--
lgamma_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lgamma_ self =
  [C.block|void {
    VariableType::lgamma_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  lgamma_out lgamma_out
--
lgamma_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lgamma_out out self =
  [C.block|void {
    VariableType::lgamma_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  linear linear
--
linear :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
linear input weight bias =
  [C.block|Tensor* {
    return new Tensor(VariableType::linear(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


--  linspace linspace
--
linspace :: ForeignPtr CScalar -> ForeignPtr CScalar -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
linspace start end steps options =
  [C.block|Tensor* {
    return new Tensor(VariableType::linspace(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), $(int64_t steps), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  linspace_out linspace_out
--
linspace_out :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> Int64 -> IO (ForeignPtr CTensor)
linspace_out out start end steps =
  [C.block|void {
    VariableType::linspace_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), $(int64_t steps));
   }|] >> pure (out)


--  log log
--
log :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log self =
  [C.block|Tensor* {
    return new Tensor(VariableType::log(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  log10 log10
--
log10 :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log10 self =
  [C.block|Tensor* {
    return new Tensor(VariableType::log10(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  log10_ log10_
--
log10_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log10_ self =
  [C.block|void {
    VariableType::log10_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  log10_out log10_out
--
log10_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log10_out out self =
  [C.block|void {
    VariableType::log10_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  log1p log1p
--
log1p :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log1p self =
  [C.block|Tensor* {
    return new Tensor(VariableType::log1p(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  log1p_ log1p_
--
log1p_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log1p_ self =
  [C.block|void {
    VariableType::log1p_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  log1p_out log1p_out
--
log1p_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log1p_out out self =
  [C.block|void {
    VariableType::log1p_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  log2 log2
--
log2 :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log2 self =
  [C.block|Tensor* {
    return new Tensor(VariableType::log2(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  log2_ log2_
--
log2_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log2_ self =
  [C.block|void {
    VariableType::log2_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  log2_out log2_out
--
log2_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log2_out out self =
  [C.block|void {
    VariableType::log2_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  log_ log_
--
log_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log_ self =
  [C.block|void {
    VariableType::log_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  log_normal_ log_normal_
--
log_normal_ :: ForeignPtr CTensor -> CDouble -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
log_normal_ self mean std generator =
  [C.block|void {
    VariableType::log_normal_(*$fptr-ptr:(Tensor* self), $(double mean), $(double std), $(Generator* generator));
   }|] >> pure self


--  log_out log_out
--
log_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log_out out self =
  [C.block|void {
    VariableType::log_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  log_sigmoid log_sigmoid
--
log_sigmoid :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log_sigmoid self =
  [C.block|Tensor* {
    return new Tensor(VariableType::log_sigmoid(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  log_sigmoid_out log_sigmoid_out
--
log_sigmoid_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
log_sigmoid_out out self =
  [C.block|void {
    VariableType::log_sigmoid_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  log_softmax log_softmax
--
log_softmax :: ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
log_softmax self dim dtype =
  [C.block|Tensor* {
    return new Tensor(VariableType::log_softmax(*$fptr-ptr:(Tensor* self), $(int64_t dim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  logdet logdet
--
logdet :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
logdet self =
  [C.block|Tensor* {
    return new Tensor(VariableType::logdet(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  logspace logspace
--
logspace :: ForeignPtr CScalar -> ForeignPtr CScalar -> Int64 -> CDouble -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
logspace start end steps base options =
  [C.block|Tensor* {
    return new Tensor(VariableType::logspace(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), $(int64_t steps), $(double base), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  logspace_out logspace_out
--
logspace_out :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> Int64 -> CDouble -> IO (ForeignPtr CTensor)
logspace_out out start end steps base =
  [C.block|void {
    VariableType::logspace_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), $(int64_t steps), $(double base));
   }|] >> pure (out)


--  logsumexp logsumexp
--
logsumexp :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
logsumexp self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::logsumexp(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  logsumexp_out logsumexp_out
--
logsumexp_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
logsumexp_out out self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|void {
    VariableType::logsumexp_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim));
   }|] >> pure (out)


--  lstm lstm
--
lstm :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
lstm input hx params has_biases num_layers dropout train bidirectional batch_first =  V.unsafeWith hx $ \hx__array -> let hx__size = fromIntegral (V.length hx) in V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::lstm(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** hx__array), $(size_t hx__size)), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), $(bool batch_first)));
   }|] >>= unTupleTensorTensorTensor


--  lstm lstm__1
--
lstm__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
lstm__1 dataX batch_sizes hx params has_biases num_layers dropout train bidirectional =  V.unsafeWith hx $ \hx__array -> let hx__size = fromIntegral (V.length hx) in V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::lstm(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), pack_tensor_list($(Tensor** hx__array), $(size_t hx__size)), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional)));
   }|] >>= unTupleTensorTensorTensor


--  lstm_cell lstm_cell
--
lstm_cell :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
lstm_cell input hx w_ih w_hh b_ih b_hh =  V.unsafeWith hx $ \hx__array -> let hx__size = fromIntegral (V.length hx) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::lstm_cell(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** hx__array), $(size_t hx__size)), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh)));
   }|] >>= unTupleTensorTensor


--  lstsq lstsq
--
lstsq :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
lstsq self a =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::lstsq(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a)));
   }|] >>= unTupleTensorTensor


--  lstsq_out lstsq_out
--
lstsq_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
lstsq_out x qr self a =
  [C.block|void {
    VariableType::lstsq_out(*$fptr-ptr:(Tensor* x), *$fptr-ptr:(Tensor* qr), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a));
   }|] >> pure (x,qr)


--  lt lt
--
lt :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lt self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::lt(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


--  lt lt__1
--
lt__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lt__1 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::lt(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  lt_ lt_
--
lt_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lt_ self other =
  [C.block|void {
    VariableType::lt_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  lt_ lt___1
--
lt___1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lt___1 self other =
  [C.block|void {
    VariableType::lt_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  lt_out lt_out
--
lt_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
lt_out out self other =
  [C.block|void {
    VariableType::lt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


--  lt_out lt_out__1
--
lt_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lt_out__1 out self other =
  [C.block|void {
    VariableType::lt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  lu_solve lu_solve
--
lu_solve :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lu_solve self lu_data lu_pivots =
  [C.block|Tensor* {
    return new Tensor(VariableType::lu_solve(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* lu_data), *$fptr-ptr:(Tensor* lu_pivots)));
   }|] >>= newForeignPtr deleteTensor


--  lu_solve_out lu_solve_out
--
lu_solve_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
lu_solve_out out self lu_data lu_pivots =
  [C.block|void {
    VariableType::lu_solve_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* lu_data), *$fptr-ptr:(Tensor* lu_pivots));
   }|] >> pure (out)


--  margin_ranking_loss margin_ranking_loss
--
margin_ranking_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
margin_ranking_loss input1 input2 target margin reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::margin_ranking_loss(*$fptr-ptr:(Tensor* input1), *$fptr-ptr:(Tensor* input2), *$fptr-ptr:(Tensor* target), $(double margin), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  masked_fill masked_fill
--
masked_fill :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
masked_fill self mask value =
  [C.block|Tensor* {
    return new Tensor(VariableType::masked_fill(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


--  masked_fill masked_fill__1
--
masked_fill__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_fill__1 self mask value =
  [C.block|Tensor* {
    return new Tensor(VariableType::masked_fill(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Tensor* value)));
   }|] >>= newForeignPtr deleteTensor


--  masked_fill_ masked_fill_
--
masked_fill_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
masked_fill_ self mask value =
  [C.block|void {
    VariableType::masked_fill_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


--  masked_fill_ masked_fill___1
--
masked_fill___1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_fill___1 self mask value =
  [C.block|void {
    VariableType::masked_fill_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Tensor* value));
   }|] >> pure self


--  masked_scatter masked_scatter
--
masked_scatter :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_scatter self mask source =
  [C.block|Tensor* {
    return new Tensor(VariableType::masked_scatter(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Tensor* source)));
   }|] >>= newForeignPtr deleteTensor


--  masked_scatter_ masked_scatter_
--
masked_scatter_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_scatter_ self mask source =
  [C.block|void {
    VariableType::masked_scatter_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask), *$fptr-ptr:(Tensor* source));
   }|] >> pure self


--  masked_select masked_select
--
masked_select :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_select self mask =
  [C.block|Tensor* {
    return new Tensor(VariableType::masked_select(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask)));
   }|] >>= newForeignPtr deleteTensor


--  masked_select_out masked_select_out
--
masked_select_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
masked_select_out out self mask =
  [C.block|void {
    VariableType::masked_select_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask));
   }|] >> pure (out)


--  matmul matmul
--
matmul :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
matmul self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::matmul(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  matmul_out matmul_out
--
matmul_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
matmul_out out self other =
  [C.block|void {
    VariableType::matmul_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  matrix_power matrix_power
--
matrix_power :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
matrix_power self n =
  [C.block|Tensor* {
    return new Tensor(VariableType::matrix_power(*$fptr-ptr:(Tensor* self), $(int64_t n)));
   }|] >>= newForeignPtr deleteTensor


--  matrix_rank matrix_rank
--
matrix_rank :: ForeignPtr CTensor -> CDouble -> CBool -> IO (ForeignPtr CTensor)
matrix_rank self tol symmetric =
  [C.block|Tensor* {
    return new Tensor(VariableType::matrix_rank(*$fptr-ptr:(Tensor* self), $(double tol), $(bool symmetric)));
   }|] >>= newForeignPtr deleteTensor


--  matrix_rank matrix_rank__1
--
matrix_rank__1 :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
matrix_rank__1 self symmetric =
  [C.block|Tensor* {
    return new Tensor(VariableType::matrix_rank(*$fptr-ptr:(Tensor* self), $(bool symmetric)));
   }|] >>= newForeignPtr deleteTensor


--  max max
--
max :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max self dim keepdim =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::max(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


--  max max__1
--
max__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
max__1 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::max(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  max max__2
--
max__2 :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
max__2 self =
  [C.block|Tensor* {
    return new Tensor(VariableType::max(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  max_out max_out
--
max_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_out max max_values self dim keepdim =
  [C.block|void {
    VariableType::max_out(*$fptr-ptr:(Tensor* max), *$fptr-ptr:(Tensor* max_values), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (max,max_values)


--  max_out max_out__1
--
max_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
max_out__1 out self other =
  [C.block|void {
    VariableType::max_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  max_pool1d max_pool1d
--
max_pool1d :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
max_pool1d self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::max_pool1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


--  max_pool1d_with_indices max_pool1d_with_indices
--
max_pool1d_with_indices :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool1d_with_indices self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::max_pool1d_with_indices(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= unTupleTensorTensor


--  max_pool2d max_pool2d
--
max_pool2d :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
max_pool2d self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


--  max_pool2d_with_indices max_pool2d_with_indices
--
max_pool2d_with_indices :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool2d_with_indices self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::max_pool2d_with_indices(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= unTupleTensorTensor


--  max_pool2d_with_indices_out max_pool2d_with_indices_out
--
max_pool2d_with_indices_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool2d_with_indices_out out indices self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|void {
    VariableType::max_pool2d_with_indices_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode));
   }|] >> pure (out,indices)


--  max_pool3d max_pool3d
--
max_pool3d :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
max_pool3d self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::max_pool3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


--  max_pool3d_with_indices max_pool3d_with_indices
--
max_pool3d_with_indices :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool3d_with_indices self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::max_pool3d_with_indices(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= unTupleTensorTensor


--  max_pool3d_with_indices_out max_pool3d_with_indices_out
--
max_pool3d_with_indices_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
max_pool3d_with_indices_out out indices self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|void {
    VariableType::max_pool3d_with_indices_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode));
   }|] >> pure (out,indices)


--  max_unpool2d max_unpool2d
--
max_unpool2d :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
max_unpool2d self indices output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::max_unpool2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* indices), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


--  max_unpool2d_out max_unpool2d_out
--
max_unpool2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
max_unpool2d_out out self indices output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::max_unpool2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* indices), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out)


--  max_unpool3d max_unpool3d
--
max_unpool3d :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
max_unpool3d self indices output_size stride padding =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::max_unpool3d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* indices), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


--  max_unpool3d_out max_unpool3d_out
--
max_unpool3d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
max_unpool3d_out out self indices output_size stride padding =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|void {
    VariableType::max_unpool3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* indices), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


--  max_values max_values
--
max_values :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
max_values self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::max_values(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  mean mean
--
mean :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
mean self dtype =
  [C.block|Tensor* {
    return new Tensor(VariableType::mean(*$fptr-ptr:(Tensor* self), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  mean mean__1
--
mean__1 :: ForeignPtr CTensor -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
mean__1 self dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::mean(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  mean_out mean_out
--
mean_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
mean_out out self dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|void {
    VariableType::mean_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype)));
   }|] >> pure (out)


--  median median
--
median :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
median self dim keepdim =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::median(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


--  median median__1
--
median__1 :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
median__1 self =
  [C.block|Tensor* {
    return new Tensor(VariableType::median(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  median_out median_out
--
median_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
median_out values indices self dim keepdim =
  [C.block|void {
    VariableType::median_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (values,indices)


--  meshgrid meshgrid
--
meshgrid :: Vector (Ptr CTensor) -> IO (Vector (Ptr CTensor))
meshgrid tensors =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in
  [C.block|void* {
    return (void*)new std::vector<Tensor>(VariableType::meshgrid(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size))));
   }|] >>= unVectorTensor


--  min min
--
min :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
min self dim keepdim =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::min(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


--  min min__1
--
min__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
min__1 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::min(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  min min__2
--
min__2 :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
min__2 self =
  [C.block|Tensor* {
    return new Tensor(VariableType::min(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  min_out min_out
--
min_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
min_out min min_indices self dim keepdim =
  [C.block|void {
    VariableType::min_out(*$fptr-ptr:(Tensor* min), *$fptr-ptr:(Tensor* min_indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (min,min_indices)


--  min_out min_out__1
--
min_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
min_out__1 out self other =
  [C.block|void {
    VariableType::min_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  min_values min_values
--
min_values :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
min_values self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::min_values(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  miopen_batch_norm miopen_batch_norm
--
miopen_batch_norm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
miopen_batch_norm input weight bias running_mean running_var training exponential_average_factor epsilon =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::miopen_batch_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool training), $(double exponential_average_factor), $(double epsilon)));
   }|] >>= unTupleTensorTensorTensor


--  miopen_convolution miopen_convolution
--
miopen_convolution :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
miopen_convolution self weight bias padding stride dilation groups benchmark deterministic =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::miopen_convolution(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic)));
   }|] >>= newForeignPtr deleteTensor


--  miopen_convolution_transpose miopen_convolution_transpose
--
miopen_convolution_transpose :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
miopen_convolution_transpose self weight bias padding output_padding stride dilation groups benchmark deterministic =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith output_padding $ \output_padding__array -> let output_padding__size = fromIntegral (V.length output_padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::miopen_convolution_transpose(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* output_padding__array), $(size_t output_padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic)));
   }|] >>= newForeignPtr deleteTensor


--  miopen_depthwise_convolution miopen_depthwise_convolution
--
miopen_depthwise_convolution :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
miopen_depthwise_convolution self weight bias padding stride dilation groups benchmark deterministic =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::miopen_depthwise_convolution(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups), $(bool benchmark), $(bool deterministic)));
   }|] >>= newForeignPtr deleteTensor


--  miopen_rnn miopen_rnn
--
miopen_rnn :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> CBool -> CDouble -> CBool -> CBool -> Vector Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
miopen_rnn input weight weight_stride0 hx cx mode hidden_size num_layers batch_first dropout train bidirectional batch_sizes dropout_state =  V.unsafeWith weight $ \weight__array -> let weight__size = fromIntegral (V.length weight) in V.unsafeWith batch_sizes $ \batch_sizes__array -> let batch_sizes__size = fromIntegral (V.length batch_sizes) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>(VariableType::miopen_rnn(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** weight__array), $(size_t weight__size)), $(int64_t weight_stride0), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* cx), $(int64_t mode), $(int64_t hidden_size), $(int64_t num_layers), $(bool batch_first), $(double dropout), $(bool train), $(bool bidirectional), ArrayRef<int64_t>($(int64_t* batch_sizes__array), $(size_t batch_sizes__size)), *$fptr-ptr:(Tensor* dropout_state)));
   }|] >>= unTupleTensorTensorTensorTensorTensor


--  mkldnn_adaptive_avg_pool2d mkldnn_adaptive_avg_pool2d
--
mkldnn_adaptive_avg_pool2d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
mkldnn_adaptive_avg_pool2d self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::mkldnn_adaptive_avg_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


--  mkldnn_convolution mkldnn_convolution
--
mkldnn_convolution :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
mkldnn_convolution self weight bias padding stride dilation groups =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::mkldnn_convolution(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


--  mkldnn_linear mkldnn_linear
--
mkldnn_linear :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mkldnn_linear input weight bias =
  [C.block|Tensor* {
    return new Tensor(VariableType::mkldnn_linear(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias)));
   }|] >>= newForeignPtr deleteTensor


--  mkldnn_max_pool2d mkldnn_max_pool2d
--
mkldnn_max_pool2d :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
mkldnn_max_pool2d self kernel_size stride padding dilation ceil_mode =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::mkldnn_max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(bool ceil_mode)));
   }|] >>= newForeignPtr deleteTensor


--  mkldnn_reorder_conv2d_weight mkldnn_reorder_conv2d_weight
--
mkldnn_reorder_conv2d_weight :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Int64 -> IO (ForeignPtr CTensor)
mkldnn_reorder_conv2d_weight self padding stride dilation groups =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::mkldnn_reorder_conv2d_weight(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)), $(int64_t groups)));
   }|] >>= newForeignPtr deleteTensor


--  mm mm
--
mm :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mm self mat2 =
  [C.block|Tensor* {
    return new Tensor(VariableType::mm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat2)));
   }|] >>= newForeignPtr deleteTensor


--  mm_out mm_out
--
mm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mm_out out self mat2 =
  [C.block|void {
    VariableType::mm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat2));
   }|] >> pure (out)


--  mode mode
--
mode :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
mode self dim keepdim =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::mode(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


--  mode_out mode_out
--
mode_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
mode_out values indices self dim keepdim =
  [C.block|void {
    VariableType::mode_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim));
   }|] >> pure (values,indices)


--  mse_loss mse_loss
--
mse_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
mse_loss self target reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::mse_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  mse_loss_out mse_loss_out
--
mse_loss_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
mse_loss_out out self target reduction =
  [C.block|void {
    VariableType::mse_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction));
   }|] >> pure (out)


--  mul mul
--
mul :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mul self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::mul(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  mul mul__1
--
mul__1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
mul__1 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::mul(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


--  mul_ mul_
--
mul_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mul_ self other =
  [C.block|void {
    VariableType::mul_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  mul_ mul___1
--
mul___1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
mul___1 self other =
  [C.block|void {
    VariableType::mul_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  mul_out mul_out
--
mul_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mul_out out self other =
  [C.block|void {
    VariableType::mul_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  multi_margin_loss multi_margin_loss
--
multi_margin_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
multi_margin_loss self target p margin weight reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::multi_margin_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Scalar* p), *$fptr-ptr:(Scalar* margin), *$fptr-ptr:(Tensor* weight), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  multi_margin_loss_out multi_margin_loss_out
--
multi_margin_loss_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
multi_margin_loss_out out self target p margin weight reduction =
  [C.block|void {
    VariableType::multi_margin_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Scalar* p), *$fptr-ptr:(Scalar* margin), *$fptr-ptr:(Tensor* weight), $(int64_t reduction));
   }|] >> pure (out)


--  multilabel_margin_loss multilabel_margin_loss
--
multilabel_margin_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
multilabel_margin_loss self target reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::multilabel_margin_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  multilabel_margin_loss_out multilabel_margin_loss_out
--
multilabel_margin_loss_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
multilabel_margin_loss_out out self target reduction =
  [C.block|void {
    VariableType::multilabel_margin_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction));
   }|] >> pure (out)


--  multinomial multinomial
--
multinomial :: ForeignPtr CTensor -> Int64 -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
multinomial self num_samples replacement generator =
  [C.block|Tensor* {
    return new Tensor(VariableType::multinomial(*$fptr-ptr:(Tensor* self), $(int64_t num_samples), $(bool replacement), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


--  multinomial_out multinomial_out
--
multinomial_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
multinomial_out out self num_samples replacement generator =
  [C.block|void {
    VariableType::multinomial_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t num_samples), $(bool replacement), $(Generator* generator));
   }|] >> pure (out)


--  mv mv
--
mv :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mv self vec =
  [C.block|Tensor* {
    return new Tensor(VariableType::mv(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec)));
   }|] >>= newForeignPtr deleteTensor


--  mv_out mv_out
--
mv_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
mv_out out self vec =
  [C.block|void {
    VariableType::mv_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* vec));
   }|] >> pure (out)


--  mvlgamma mvlgamma
--
mvlgamma :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
mvlgamma self p =
  [C.block|Tensor* {
    return new Tensor(VariableType::mvlgamma(*$fptr-ptr:(Tensor* self), $(int64_t p)));
   }|] >>= newForeignPtr deleteTensor


--  mvlgamma_ mvlgamma_
--
mvlgamma_ :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
mvlgamma_ self p =
  [C.block|void {
    VariableType::mvlgamma_(*$fptr-ptr:(Tensor* self), $(int64_t p));
   }|] >> pure self


--  narrow narrow
--
narrow :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
narrow self dim start length =
  [C.block|Tensor* {
    return new Tensor(VariableType::narrow(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(int64_t start), $(int64_t length)));
   }|] >>= newForeignPtr deleteTensor


--  narrow_copy narrow_copy
--
narrow_copy :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
narrow_copy self dim start length =
  [C.block|Tensor* {
    return new Tensor(VariableType::narrow_copy(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(int64_t start), $(int64_t length)));
   }|] >>= newForeignPtr deleteTensor


--  native_batch_norm native_batch_norm
--
native_batch_norm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CDouble -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
native_batch_norm input weight bias running_mean running_var training momentum eps =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::native_batch_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), *$fptr-ptr:(Tensor* running_mean), *$fptr-ptr:(Tensor* running_var), $(bool training), $(double momentum), $(double eps)));
   }|] >>= unTupleTensorTensorTensor


--  native_layer_norm native_layer_norm
--
native_layer_norm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> CDouble -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
native_layer_norm input weight bias m n eps =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::native_layer_norm(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* weight), *$fptr-ptr:(Tensor* bias), $(int64_t m), $(int64_t n), $(double eps)));
   }|] >>= unTupleTensorTensorTensor


--  native_norm native_norm
--
native_norm :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
native_norm self p =
  [C.block|Tensor* {
    return new Tensor(VariableType::native_norm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p)));
   }|] >>= newForeignPtr deleteTensor


--  ne ne
--
ne :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ne self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::ne(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


--  ne ne__1
--
ne__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ne__1 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::ne(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  ne_ ne_
--
ne_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ne_ self other =
  [C.block|void {
    VariableType::ne_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  ne_ ne___1
--
ne___1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ne___1 self other =
  [C.block|void {
    VariableType::ne_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  ne_out ne_out
--
ne_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
ne_out out self other =
  [C.block|void {
    VariableType::ne_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


--  ne_out ne_out__1
--
ne_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ne_out__1 out self other =
  [C.block|void {
    VariableType::ne_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  neg neg
--
neg :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
neg self =
  [C.block|Tensor* {
    return new Tensor(VariableType::neg(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  neg_ neg_
--
neg_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
neg_ self =
  [C.block|void {
    VariableType::neg_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  neg_out neg_out
--
neg_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
neg_out out self =
  [C.block|void {
    VariableType::neg_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  nll_loss nll_loss
--
nll_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
nll_loss self target weight reduction ignore_index =
  [C.block|Tensor* {
    return new Tensor(VariableType::nll_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction), $(int64_t ignore_index)));
   }|] >>= newForeignPtr deleteTensor


--  nll_loss2d nll_loss2d
--
nll_loss2d :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
nll_loss2d self target weight reduction ignore_index =
  [C.block|Tensor* {
    return new Tensor(VariableType::nll_loss2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction), $(int64_t ignore_index)));
   }|] >>= newForeignPtr deleteTensor


--  nll_loss2d_out nll_loss2d_out
--
nll_loss2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
nll_loss2d_out out self target weight reduction ignore_index =
  [C.block|void {
    VariableType::nll_loss2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction), $(int64_t ignore_index));
   }|] >> pure (out)


--  nll_loss_out nll_loss_out
--
nll_loss_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
nll_loss_out out self target weight reduction ignore_index =
  [C.block|void {
    VariableType::nll_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), *$fptr-ptr:(Tensor* weight), $(int64_t reduction), $(int64_t ignore_index));
   }|] >> pure (out)


--  nonzero nonzero
--
nonzero :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
nonzero self =
  [C.block|Tensor* {
    return new Tensor(VariableType::nonzero(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  nonzero_numpy nonzero_numpy
--
nonzero_numpy :: ForeignPtr CTensor -> IO (Vector (Ptr CTensor))
nonzero_numpy self =
  [C.block|void* {
    return (void*)new std::vector<Tensor>(VariableType::nonzero_numpy(*$fptr-ptr:(Tensor* self)));
   }|] >>= unVectorTensor


--  nonzero_out nonzero_out
--
nonzero_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
nonzero_out out self =
  [C.block|void {
    VariableType::nonzero_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  norm norm
--
norm :: ForeignPtr CTensor -> ForeignPtr CScalar -> Int8 -> IO (ForeignPtr CTensor)
norm self p dtype =
  [C.block|Tensor* {
    return new Tensor(VariableType::norm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  norm norm__1
--
norm__1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
norm__1 self p =
  [C.block|Tensor* {
    return new Tensor(VariableType::norm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p)));
   }|] >>= newForeignPtr deleteTensor


--  norm norm__2
--
norm__2 :: ForeignPtr CTensor -> ForeignPtr CScalar -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
norm__2 self p dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::norm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  norm norm__3
--
norm__3 :: ForeignPtr CTensor -> ForeignPtr CScalar -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
norm__3 self p dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::norm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  norm_except_dim norm_except_dim
--
norm_except_dim :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
norm_except_dim v pow dim =
  [C.block|Tensor* {
    return new Tensor(VariableType::norm_except_dim(*$fptr-ptr:(Tensor* v), $(int64_t pow), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  norm_out norm_out
--
norm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
norm_out out self p dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|void {
    VariableType::norm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype)));
   }|] >> pure (out)


--  norm_out norm_out__1
--
norm_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
norm_out__1 out self p dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|void {
    VariableType::norm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim));
   }|] >> pure (out)


--  normal normal
--
normal :: ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal mean std generator =
  [C.block|Tensor* {
    return new Tensor(VariableType::normal(*$fptr-ptr:(Tensor* mean), $(double std), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


--  normal normal__1
--
normal__1 :: CDouble -> ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal__1 mean std generator =
  [C.block|Tensor* {
    return new Tensor(VariableType::normal($(double mean), *$fptr-ptr:(Tensor* std), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


--  normal normal__2
--
normal__2 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal__2 mean std generator =
  [C.block|Tensor* {
    return new Tensor(VariableType::normal(*$fptr-ptr:(Tensor* mean), *$fptr-ptr:(Tensor* std), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


--  normal normal__3
--
normal__3 :: CDouble -> CDouble -> Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
normal__3 mean std size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::normal($(double mean), $(double std), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  normal_ normal_
--
normal_ :: ForeignPtr CTensor -> CDouble -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal_ self mean std generator =
  [C.block|void {
    VariableType::normal_(*$fptr-ptr:(Tensor* self), $(double mean), $(double std), $(Generator* generator));
   }|] >> pure self


--  normal_out normal_out
--
normal_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal_out out mean std generator =
  [C.block|void {
    VariableType::normal_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* mean), $(double std), $(Generator* generator));
   }|] >> pure (out)


--  normal_out normal_out__1
--
normal_out__1 :: ForeignPtr CTensor -> CDouble -> ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal_out__1 out mean std generator =
  [C.block|void {
    VariableType::normal_out(*$fptr-ptr:(Tensor* out), $(double mean), *$fptr-ptr:(Tensor* std), $(Generator* generator));
   }|] >> pure (out)


--  normal_out normal_out__2
--
normal_out__2 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal_out__2 out mean std generator =
  [C.block|void {
    VariableType::normal_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* mean), *$fptr-ptr:(Tensor* std), $(Generator* generator));
   }|] >> pure (out)


--  normal_out normal_out__3
--
normal_out__3 :: ForeignPtr CTensor -> CDouble -> CDouble -> Vector Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
normal_out__3 out mean std size generator =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::normal_out(*$fptr-ptr:(Tensor* out), $(double mean), $(double std), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator));
   }|] >> pure (out)


--  nuclear_norm nuclear_norm
--
nuclear_norm :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
nuclear_norm self keepdim =
  [C.block|Tensor* {
    return new Tensor(VariableType::nuclear_norm(*$fptr-ptr:(Tensor* self), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  nuclear_norm nuclear_norm__1
--
nuclear_norm__1 :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
nuclear_norm__1 self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::nuclear_norm(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  nuclear_norm_out nuclear_norm_out
--
nuclear_norm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
nuclear_norm_out out self keepdim =
  [C.block|void {
    VariableType::nuclear_norm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(bool keepdim));
   }|] >> pure (out)


--  nuclear_norm_out nuclear_norm_out__1
--
nuclear_norm_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
nuclear_norm_out__1 out self dim keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|void {
    VariableType::nuclear_norm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim));
   }|] >> pure (out)


--  numel numel
--
numel :: ForeignPtr CTensor -> IO (Int64)
numel self =
  [C.block|int64_t {
    return VariableType::numel(*$fptr-ptr:(Tensor* self));
   }|]


--  numpy_T numpy_t
--
numpy_t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
numpy_t self =
  [C.block|Tensor* {
    return new Tensor(VariableType::numpy_T(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  one_hot one_hot
--
one_hot :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
one_hot self num_classes =
  [C.block|Tensor* {
    return new Tensor(VariableType::one_hot(*$fptr-ptr:(Tensor* self), $(int64_t num_classes)));
   }|] >>= newForeignPtr deleteTensor


--  ones ones
--
ones :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
ones size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::ones(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  ones_like ones_like
--
ones_like :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
ones_like self =
  [C.block|Tensor* {
    return new Tensor(VariableType::ones_like(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  ones_like ones_like__1
--
ones_like__1 :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
ones_like__1 self options =
  [C.block|Tensor* {
    return new Tensor(VariableType::ones_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  ones_out ones_out
--
ones_out :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
ones_out out size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::ones_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure (out)


--  orgqr orgqr
--
orgqr :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
orgqr self input2 =
  [C.block|Tensor* {
    return new Tensor(VariableType::orgqr(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* input2)));
   }|] >>= newForeignPtr deleteTensor


--  orgqr_out orgqr_out
--
orgqr_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
orgqr_out out self input2 =
  [C.block|void {
    VariableType::orgqr_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* input2));
   }|] >> pure (out)


--  ormqr ormqr
--
ormqr :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor)
ormqr self input2 input3 left transpose =
  [C.block|Tensor* {
    return new Tensor(VariableType::ormqr(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* input2), *$fptr-ptr:(Tensor* input3), $(bool left), $(bool transpose)));
   }|] >>= newForeignPtr deleteTensor


--  ormqr_out ormqr_out
--
ormqr_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor)
ormqr_out out self input2 input3 left transpose =
  [C.block|void {
    VariableType::ormqr_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* input2), *$fptr-ptr:(Tensor* input3), $(bool left), $(bool transpose));
   }|] >> pure (out)


--  pairwise_distance pairwise_distance
--
pairwise_distance :: ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> CBool -> IO (ForeignPtr CTensor)
pairwise_distance x1 x2 p eps keepdim =
  [C.block|Tensor* {
    return new Tensor(VariableType::pairwise_distance(*$fptr-ptr:(Tensor* x1), *$fptr-ptr:(Tensor* x2), $(double p), $(double eps), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  pdist pdist
--
pdist :: ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor)
pdist self p =
  [C.block|Tensor* {
    return new Tensor(VariableType::pdist(*$fptr-ptr:(Tensor* self), $(double p)));
   }|] >>= newForeignPtr deleteTensor


--  permute permute
--
permute :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
permute self dims =  V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in
  [C.block|Tensor* {
    return new Tensor(VariableType::permute(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


--  pin_memory pin_memory
--
pin_memory :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pin_memory self =
  [C.block|Tensor* {
    return new Tensor(VariableType::pin_memory(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  pinverse pinverse
--
pinverse :: ForeignPtr CTensor -> CDouble -> IO (ForeignPtr CTensor)
pinverse self rcond =
  [C.block|Tensor* {
    return new Tensor(VariableType::pinverse(*$fptr-ptr:(Tensor* self), $(double rcond)));
   }|] >>= newForeignPtr deleteTensor


--  pixel_shuffle pixel_shuffle
--
pixel_shuffle :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
pixel_shuffle self upscale_factor =
  [C.block|Tensor* {
    return new Tensor(VariableType::pixel_shuffle(*$fptr-ptr:(Tensor* self), $(int64_t upscale_factor)));
   }|] >>= newForeignPtr deleteTensor


--  poisson poisson
--
poisson :: ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
poisson self generator =
  [C.block|Tensor* {
    return new Tensor(VariableType::poisson(*$fptr-ptr:(Tensor* self), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


--  poisson_nll_loss poisson_nll_loss
--
poisson_nll_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
poisson_nll_loss input target log_input full eps reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::poisson_nll_loss(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* target), $(bool log_input), $(bool full), $(double eps), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  polygamma polygamma
--
polygamma :: Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
polygamma n self =
  [C.block|Tensor* {
    return new Tensor(VariableType::polygamma($(int64_t n), *$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  polygamma_ polygamma_
--
polygamma_ :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
polygamma_ self n =
  [C.block|void {
    VariableType::polygamma_(*$fptr-ptr:(Tensor* self), $(int64_t n));
   }|] >> pure self


--  polygamma_out polygamma_out
--
polygamma_out :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
polygamma_out out n self =
  [C.block|void {
    VariableType::polygamma_out(*$fptr-ptr:(Tensor* out), $(int64_t n), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  pow pow
--
pow :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
pow self exponent =
  [C.block|Tensor* {
    return new Tensor(VariableType::pow(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* exponent)));
   }|] >>= newForeignPtr deleteTensor


--  pow pow__1
--
pow__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pow__1 self exponent =
  [C.block|Tensor* {
    return new Tensor(VariableType::pow(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* exponent)));
   }|] >>= newForeignPtr deleteTensor


--  pow pow__2
--
pow__2 :: ForeignPtr CScalar -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pow__2 self exponent =
  [C.block|Tensor* {
    return new Tensor(VariableType::pow(*$fptr-ptr:(Scalar* self), *$fptr-ptr:(Tensor* exponent)));
   }|] >>= newForeignPtr deleteTensor


--  pow_ pow_
--
pow_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
pow_ self exponent =
  [C.block|void {
    VariableType::pow_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* exponent));
   }|] >> pure self


--  pow_ pow___1
--
pow___1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pow___1 self exponent =
  [C.block|void {
    VariableType::pow_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* exponent));
   }|] >> pure self


--  pow_out pow_out
--
pow_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
pow_out out self exponent =
  [C.block|void {
    VariableType::pow_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* exponent));
   }|] >> pure (out)


--  pow_out pow_out__1
--
pow_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pow_out__1 out self exponent =
  [C.block|void {
    VariableType::pow_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* exponent));
   }|] >> pure (out)


--  pow_out pow_out__2
--
pow_out__2 :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
pow_out__2 out self exponent =
  [C.block|void {
    VariableType::pow_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Scalar* self), *$fptr-ptr:(Tensor* exponent));
   }|] >> pure (out)


--  prelu prelu
--
prelu :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
prelu self weight =
  [C.block|Tensor* {
    return new Tensor(VariableType::prelu(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight)));
   }|] >>= newForeignPtr deleteTensor


--  prod prod
--
prod :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
prod self dtype =
  [C.block|Tensor* {
    return new Tensor(VariableType::prod(*$fptr-ptr:(Tensor* self), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  prod prod__1
--
prod__1 :: ForeignPtr CTensor -> Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
prod__1 self dim keepdim dtype =
  [C.block|Tensor* {
    return new Tensor(VariableType::prod(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  prod_out prod_out
--
prod_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
prod_out out self dim keepdim dtype =
  [C.block|void {
    VariableType::prod_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype)));
   }|] >> pure (out)


--  put_ put_
--
put_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
put_ self index source accumulate =
  [C.block|void {
    VariableType::put_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* source), $(bool accumulate));
   }|] >> pure self


--  q_scale q_scale
--
q_scale :: ForeignPtr CTensor -> IO (CDouble)
q_scale self =
  [C.block|double {
    return VariableType::q_scale(*$fptr-ptr:(Tensor* self));
   }|]


--  q_zero_point q_zero_point
--
q_zero_point :: ForeignPtr CTensor -> IO (Int64)
q_zero_point self =
  [C.block|int64_t {
    return VariableType::q_zero_point(*$fptr-ptr:(Tensor* self));
   }|]


--  qr qr
--
qr :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
qr self some =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::qr(*$fptr-ptr:(Tensor* self), $(bool some)));
   }|] >>= unTupleTensorTensor


--  qr_out qr_out
--
qr_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
qr_out q r self some =
  [C.block|void {
    VariableType::qr_out(*$fptr-ptr:(Tensor* q), *$fptr-ptr:(Tensor* r), *$fptr-ptr:(Tensor* self), $(bool some));
   }|] >> pure (q,r)


--  qscheme qscheme
--
qscheme :: ForeignPtr CTensor -> IO (Word8)
qscheme self =
  [C.block|uint8_t {
    return static_cast<uint8_t>(VariableType::qscheme(*$fptr-ptr:(Tensor* self)));
   }|]


--  quantize_linear quantize_linear
--
quantize_linear :: ForeignPtr CTensor -> CDouble -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
quantize_linear self scale zero_point dtype =
  [C.block|Tensor* {
    return new Tensor(VariableType::quantize_linear(*$fptr-ptr:(Tensor* self), $(double scale), $(int64_t zero_point), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  quantize_linear_per_channel quantize_linear_per_channel
--
quantize_linear_per_channel :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Int8 -> IO (ForeignPtr CTensor)
quantize_linear_per_channel self scales zero_points axis dtype =  V.unsafeWith axis $ \axis__array -> let axis__size = fromIntegral (V.length axis) in
  [C.block|Tensor* {
    return new Tensor(VariableType::quantize_linear_per_channel(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* scales), *$fptr-ptr:(Tensor* zero_points), ArrayRef<int64_t>($(int64_t* axis__array), $(size_t axis__size)), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  quantized_gru quantized_gru
--
quantized_gru :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
quantized_gru input hx params has_biases num_layers dropout train bidirectional batch_first =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::quantized_gru(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), $(bool batch_first)));
   }|] >>= unTupleTensorTensor


--  quantized_gru quantized_gru__1
--
quantized_gru__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
quantized_gru__1 dataX batch_sizes hx params has_biases num_layers dropout train bidirectional =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::quantized_gru(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional)));
   }|] >>= unTupleTensorTensor


--  quantized_gru_cell quantized_gru_cell
--
quantized_gru_cell :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
quantized_gru_cell input hx w_ih w_hh b_ih b_hh packed_ih packed_hh col_offsets_ih col_offsets_hh scale_ih scale_hh zero_point_ih zero_point_hh =
  [C.block|Tensor* {
    return new Tensor(VariableType::quantized_gru_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh), *$fptr-ptr:(Tensor* packed_ih), *$fptr-ptr:(Tensor* packed_hh), *$fptr-ptr:(Tensor* col_offsets_ih), *$fptr-ptr:(Tensor* col_offsets_hh), *$fptr-ptr:(Scalar* scale_ih), *$fptr-ptr:(Scalar* scale_hh), *$fptr-ptr:(Scalar* zero_point_ih), *$fptr-ptr:(Scalar* zero_point_hh)));
   }|] >>= newForeignPtr deleteTensor


--  quantized_lstm quantized_lstm
--
quantized_lstm :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> Int8 -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
quantized_lstm input hx params has_biases num_layers dropout train bidirectional batch_first dtype =  V.unsafeWith hx $ \hx__array -> let hx__size = fromIntegral (V.length hx) in V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::quantized_lstm(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** hx__array), $(size_t hx__size)), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), $(bool batch_first), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= unTupleTensorTensorTensor


--  quantized_lstm_cell quantized_lstm_cell
--
quantized_lstm_cell :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
quantized_lstm_cell input hx w_ih w_hh b_ih b_hh packed_ih packed_hh col_offsets_ih col_offsets_hh scale_ih scale_hh zero_point_ih zero_point_hh =  V.unsafeWith hx $ \hx__array -> let hx__size = fromIntegral (V.length hx) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::quantized_lstm_cell(*$fptr-ptr:(Tensor* input), pack_tensor_list($(Tensor** hx__array), $(size_t hx__size)), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh), *$fptr-ptr:(Tensor* packed_ih), *$fptr-ptr:(Tensor* packed_hh), *$fptr-ptr:(Tensor* col_offsets_ih), *$fptr-ptr:(Tensor* col_offsets_hh), *$fptr-ptr:(Scalar* scale_ih), *$fptr-ptr:(Scalar* scale_hh), *$fptr-ptr:(Scalar* zero_point_ih), *$fptr-ptr:(Scalar* zero_point_hh)));
   }|] >>= unTupleTensorTensor


--  quantized_max_pool2d quantized_max_pool2d
--
quantized_max_pool2d :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
quantized_max_pool2d self kernel_size stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::quantized_max_pool2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


--  quantized_rnn_relu_cell quantized_rnn_relu_cell
--
quantized_rnn_relu_cell :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
quantized_rnn_relu_cell input hx w_ih w_hh b_ih b_hh packed_ih packed_hh col_offsets_ih col_offsets_hh scale_ih scale_hh zero_point_ih zero_point_hh =
  [C.block|Tensor* {
    return new Tensor(VariableType::quantized_rnn_relu_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh), *$fptr-ptr:(Tensor* packed_ih), *$fptr-ptr:(Tensor* packed_hh), *$fptr-ptr:(Tensor* col_offsets_ih), *$fptr-ptr:(Tensor* col_offsets_hh), *$fptr-ptr:(Scalar* scale_ih), *$fptr-ptr:(Scalar* scale_hh), *$fptr-ptr:(Scalar* zero_point_ih), *$fptr-ptr:(Scalar* zero_point_hh)));
   }|] >>= newForeignPtr deleteTensor


--  quantized_rnn_tanh_cell quantized_rnn_tanh_cell
--
quantized_rnn_tanh_cell :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
quantized_rnn_tanh_cell input hx w_ih w_hh b_ih b_hh packed_ih packed_hh col_offsets_ih col_offsets_hh scale_ih scale_hh zero_point_ih zero_point_hh =
  [C.block|Tensor* {
    return new Tensor(VariableType::quantized_rnn_tanh_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh), *$fptr-ptr:(Tensor* packed_ih), *$fptr-ptr:(Tensor* packed_hh), *$fptr-ptr:(Tensor* col_offsets_ih), *$fptr-ptr:(Tensor* col_offsets_hh), *$fptr-ptr:(Scalar* scale_ih), *$fptr-ptr:(Scalar* scale_hh), *$fptr-ptr:(Scalar* zero_point_ih), *$fptr-ptr:(Scalar* zero_point_hh)));
   }|] >>= newForeignPtr deleteTensor


--  rand rand
--
rand :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
rand size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::rand(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  rand rand__1
--
rand__1 :: Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
rand__1 size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::rand(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  rand_like rand_like
--
rand_like :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rand_like self =
  [C.block|Tensor* {
    return new Tensor(VariableType::rand_like(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  rand_like rand_like__1
--
rand_like__1 :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
rand_like__1 self options =
  [C.block|Tensor* {
    return new Tensor(VariableType::rand_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  rand_out rand_out
--
rand_out :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
rand_out out size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::rand_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure (out)


--  rand_out rand_out__1
--
rand_out__1 :: ForeignPtr CTensor -> Vector Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
rand_out__1 out size generator =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::rand_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator));
   }|] >> pure (out)


--  randint randint
--
randint :: Int64 -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint high size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::randint($(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  randint randint__1
--
randint__1 :: Int64 -> Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint__1 high size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::randint($(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  randint randint__2
--
randint__2 :: Int64 -> Int64 -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint__2 low high size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::randint($(int64_t low), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  randint randint__3
--
randint__3 :: Int64 -> Int64 -> Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint__3 low high size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::randint($(int64_t low), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  randint_like randint_like
--
randint_like :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
randint_like self high =
  [C.block|Tensor* {
    return new Tensor(VariableType::randint_like(*$fptr-ptr:(Tensor* self), $(int64_t high)));
   }|] >>= newForeignPtr deleteTensor


--  randint_like randint_like__1
--
randint_like__1 :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
randint_like__1 self low high =
  [C.block|Tensor* {
    return new Tensor(VariableType::randint_like(*$fptr-ptr:(Tensor* self), $(int64_t low), $(int64_t high)));
   }|] >>= newForeignPtr deleteTensor


--  randint_like randint_like__2
--
randint_like__2 :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint_like__2 self high options =
  [C.block|Tensor* {
    return new Tensor(VariableType::randint_like(*$fptr-ptr:(Tensor* self), $(int64_t high), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  randint_like randint_like__3
--
randint_like__3 :: ForeignPtr CTensor -> Int64 -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randint_like__3 self low high options =
  [C.block|Tensor* {
    return new Tensor(VariableType::randint_like(*$fptr-ptr:(Tensor* self), $(int64_t low), $(int64_t high), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  randint_out randint_out
--
randint_out :: ForeignPtr CTensor -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
randint_out out high size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::randint_out(*$fptr-ptr:(Tensor* out), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure (out)


--  randint_out randint_out__1
--
randint_out__1 :: ForeignPtr CTensor -> Int64 -> Vector Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
randint_out__1 out high size generator =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::randint_out(*$fptr-ptr:(Tensor* out), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator));
   }|] >> pure (out)


--  randint_out randint_out__2
--
randint_out__2 :: ForeignPtr CTensor -> Int64 -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
randint_out__2 out low high size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::randint_out(*$fptr-ptr:(Tensor* out), $(int64_t low), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure (out)


--  randint_out randint_out__3
--
randint_out__3 :: ForeignPtr CTensor -> Int64 -> Int64 -> Vector Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
randint_out__3 out low high size generator =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::randint_out(*$fptr-ptr:(Tensor* out), $(int64_t low), $(int64_t high), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator));
   }|] >> pure (out)


--  randn randn
--
randn :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randn size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::randn(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  randn randn__1
--
randn__1 :: Vector Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randn__1 size generator options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::randn(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  randn_like randn_like
--
randn_like :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
randn_like self =
  [C.block|Tensor* {
    return new Tensor(VariableType::randn_like(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  randn_like randn_like__1
--
randn_like__1 :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randn_like__1 self options =
  [C.block|Tensor* {
    return new Tensor(VariableType::randn_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  randn_out randn_out
--
randn_out :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
randn_out out size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::randn_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure (out)


--  randn_out randn_out__1
--
randn_out__1 :: ForeignPtr CTensor -> Vector Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
randn_out__1 out size generator =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::randn_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(Generator* generator));
   }|] >> pure (out)


--  random_ random_
--
random_ :: ForeignPtr CTensor -> Int64 -> Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
random_ self from to generator =
  [C.block|void {
    VariableType::random_(*$fptr-ptr:(Tensor* self), $(int64_t from), $(int64_t to), $(Generator* generator));
   }|] >> pure self


--  random_ random___1
--
random___1 :: ForeignPtr CTensor -> Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
random___1 self to generator =
  [C.block|void {
    VariableType::random_(*$fptr-ptr:(Tensor* self), $(int64_t to), $(Generator* generator));
   }|] >> pure self


--  random_ random___2
--
random___2 :: ForeignPtr CTensor -> Ptr CGenerator -> IO (ForeignPtr CTensor)
random___2 self generator =
  [C.block|void {
    VariableType::random_(*$fptr-ptr:(Tensor* self), $(Generator* generator));
   }|] >> pure self


--  randperm randperm
--
randperm :: Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randperm n options =
  [C.block|Tensor* {
    return new Tensor(VariableType::randperm($(int64_t n), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  randperm randperm__1
--
randperm__1 :: Int64 -> Ptr CGenerator -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
randperm__1 n generator options =
  [C.block|Tensor* {
    return new Tensor(VariableType::randperm($(int64_t n), $(Generator* generator), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  randperm_out randperm_out
--
randperm_out :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
randperm_out out n =
  [C.block|void {
    VariableType::randperm_out(*$fptr-ptr:(Tensor* out), $(int64_t n));
   }|] >> pure (out)


--  randperm_out randperm_out__1
--
randperm_out__1 :: ForeignPtr CTensor -> Int64 -> Ptr CGenerator -> IO (ForeignPtr CTensor)
randperm_out__1 out n generator =
  [C.block|void {
    VariableType::randperm_out(*$fptr-ptr:(Tensor* out), $(int64_t n), $(Generator* generator));
   }|] >> pure (out)


--  range range
--
range :: ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
range start end step options =
  [C.block|Tensor* {
    return new Tensor(VariableType::range(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(Scalar* step), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  range range__1
--
range__1 :: ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
range__1 start end options =
  [C.block|Tensor* {
    return new Tensor(VariableType::range(*$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  range_out range_out
--
range_out :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
range_out out start end step =
  [C.block|void {
    VariableType::range_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Scalar* start), *$fptr-ptr:(Scalar* end), *$fptr-ptr:(Scalar* step));
   }|] >> pure (out)


--  reciprocal reciprocal
--
reciprocal :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
reciprocal self =
  [C.block|Tensor* {
    return new Tensor(VariableType::reciprocal(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  reciprocal_ reciprocal_
--
reciprocal_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
reciprocal_ self =
  [C.block|void {
    VariableType::reciprocal_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  reciprocal_out reciprocal_out
--
reciprocal_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
reciprocal_out out self =
  [C.block|void {
    VariableType::reciprocal_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  reflection_pad1d reflection_pad1d
--
reflection_pad1d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
reflection_pad1d self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::reflection_pad1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


--  reflection_pad1d_out reflection_pad1d_out
--
reflection_pad1d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
reflection_pad1d_out out self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|void {
    VariableType::reflection_pad1d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


--  reflection_pad2d reflection_pad2d
--
reflection_pad2d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
reflection_pad2d self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::reflection_pad2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


--  reflection_pad2d_out reflection_pad2d_out
--
reflection_pad2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
reflection_pad2d_out out self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|void {
    VariableType::reflection_pad2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


--  relu relu
--
relu :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
relu self =
  [C.block|Tensor* {
    return new Tensor(VariableType::relu(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  relu_ relu_
--
relu_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
relu_ self =
  [C.block|void {
    VariableType::relu_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  remainder remainder
--
remainder :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
remainder self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::remainder(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other)));
   }|] >>= newForeignPtr deleteTensor


--  remainder remainder__1
--
remainder__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
remainder__1 self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::remainder(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  remainder_ remainder_
--
remainder_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
remainder_ self other =
  [C.block|void {
    VariableType::remainder_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure self


--  remainder_ remainder___1
--
remainder___1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
remainder___1 self other =
  [C.block|void {
    VariableType::remainder_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure self


--  remainder_out remainder_out
--
remainder_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
remainder_out out self other =
  [C.block|void {
    VariableType::remainder_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other));
   }|] >> pure (out)


--  remainder_out remainder_out__1
--
remainder_out__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
remainder_out__1 out self other =
  [C.block|void {
    VariableType::remainder_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other));
   }|] >> pure (out)


--  renorm renorm
--
renorm :: ForeignPtr CTensor -> ForeignPtr CScalar -> Int64 -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
renorm self p dim maxnorm =
  [C.block|Tensor* {
    return new Tensor(VariableType::renorm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), $(int64_t dim), *$fptr-ptr:(Scalar* maxnorm)));
   }|] >>= newForeignPtr deleteTensor


--  renorm_ renorm_
--
renorm_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> Int64 -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
renorm_ self p dim maxnorm =
  [C.block|void {
    VariableType::renorm_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), $(int64_t dim), *$fptr-ptr:(Scalar* maxnorm));
   }|] >> pure self


--  renorm_out renorm_out
--
renorm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> Int64 -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
renorm_out out self p dim maxnorm =
  [C.block|void {
    VariableType::renorm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* p), $(int64_t dim), *$fptr-ptr:(Scalar* maxnorm));
   }|] >> pure (out)


--  repeat repeat
--
repeat :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
repeat self repeats =  V.unsafeWith repeats $ \repeats__array -> let repeats__size = fromIntegral (V.length repeats) in
  [C.block|Tensor* {
    return new Tensor(VariableType::repeat(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* repeats__array), $(size_t repeats__size))));
   }|] >>= newForeignPtr deleteTensor


--  repeat_interleave repeat_interleave
--
repeat_interleave :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
repeat_interleave repeats =
  [C.block|Tensor* {
    return new Tensor(VariableType::repeat_interleave(*$fptr-ptr:(Tensor* repeats)));
   }|] >>= newForeignPtr deleteTensor


--  repeat_interleave repeat_interleave__1
--
repeat_interleave__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Maybe Int64 -> IO (ForeignPtr CTensor)
repeat_interleave__1 self repeats dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in
  [C.block|Tensor* {
    return new Tensor(VariableType::repeat_interleave(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* repeats), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


--  repeat_interleave repeat_interleave__2
--
repeat_interleave__2 :: ForeignPtr CTensor -> Int64 -> Maybe Int64 -> IO (ForeignPtr CTensor)
repeat_interleave__2 self repeats dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in
  [C.block|Tensor* {
    return new Tensor(VariableType::repeat_interleave(*$fptr-ptr:(Tensor* self), $(int64_t repeats), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt)));
   }|] >>= newForeignPtr deleteTensor


--  replication_pad1d replication_pad1d
--
replication_pad1d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
replication_pad1d self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::replication_pad1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


--  replication_pad1d_out replication_pad1d_out
--
replication_pad1d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
replication_pad1d_out out self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|void {
    VariableType::replication_pad1d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


--  replication_pad2d replication_pad2d
--
replication_pad2d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
replication_pad2d self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::replication_pad2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


--  replication_pad2d_out replication_pad2d_out
--
replication_pad2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
replication_pad2d_out out self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|void {
    VariableType::replication_pad2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


--  replication_pad3d replication_pad3d
--
replication_pad3d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
replication_pad3d self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::replication_pad3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


--  replication_pad3d_out replication_pad3d_out
--
replication_pad3d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
replication_pad3d_out out self padding =  V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|void {
    VariableType::replication_pad3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


--  reshape reshape
--
reshape :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
reshape self shape =  V.unsafeWith shape $ \shape__array -> let shape__size = fromIntegral (V.length shape) in
  [C.block|Tensor* {
    return new Tensor(VariableType::reshape(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* shape__array), $(size_t shape__size))));
   }|] >>= newForeignPtr deleteTensor


--  reshape_as reshape_as
--
reshape_as :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
reshape_as self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::reshape_as(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  resize_ resize_
--
resize_ :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
resize_ self size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::resize_(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure self


--  resize_as_ resize_as_
--
resize_as_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
resize_as_ self the_template =
  [C.block|void {
    VariableType::resize_as_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* the_template));
   }|] >> pure self


--  rfft rfft
--
rfft :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
rfft self signal_ndim normalized onesided =
  [C.block|Tensor* {
    return new Tensor(VariableType::rfft(*$fptr-ptr:(Tensor* self), $(int64_t signal_ndim), $(bool normalized), $(bool onesided)));
   }|] >>= newForeignPtr deleteTensor


--  rnn_relu rnn_relu
--
rnn_relu :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
rnn_relu input hx params has_biases num_layers dropout train bidirectional batch_first =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::rnn_relu(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), $(bool batch_first)));
   }|] >>= unTupleTensorTensor


--  rnn_relu rnn_relu__1
--
rnn_relu__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
rnn_relu__1 dataX batch_sizes hx params has_biases num_layers dropout train bidirectional =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::rnn_relu(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional)));
   }|] >>= unTupleTensorTensor


--  rnn_relu_cell rnn_relu_cell
--
rnn_relu_cell :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rnn_relu_cell input hx w_ih w_hh b_ih b_hh =
  [C.block|Tensor* {
    return new Tensor(VariableType::rnn_relu_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh)));
   }|] >>= newForeignPtr deleteTensor


--  rnn_tanh rnn_tanh
--
rnn_tanh :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
rnn_tanh input hx params has_biases num_layers dropout train bidirectional batch_first =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::rnn_tanh(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional), $(bool batch_first)));
   }|] >>= unTupleTensorTensor


--  rnn_tanh rnn_tanh__1
--
rnn_tanh__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector (Ptr CTensor) -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
rnn_tanh__1 dataX batch_sizes hx params has_biases num_layers dropout train bidirectional =  V.unsafeWith params $ \params__array -> let params__size = fromIntegral (V.length params) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::rnn_tanh(*$fptr-ptr:(Tensor* dataX), *$fptr-ptr:(Tensor* batch_sizes), *$fptr-ptr:(Tensor* hx), pack_tensor_list($(Tensor** params__array), $(size_t params__size)), $(bool has_biases), $(int64_t num_layers), $(double dropout), $(bool train), $(bool bidirectional)));
   }|] >>= unTupleTensorTensor


--  rnn_tanh_cell rnn_tanh_cell
--
rnn_tanh_cell :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rnn_tanh_cell input hx w_ih w_hh b_ih b_hh =
  [C.block|Tensor* {
    return new Tensor(VariableType::rnn_tanh_cell(*$fptr-ptr:(Tensor* input), *$fptr-ptr:(Tensor* hx), *$fptr-ptr:(Tensor* w_ih), *$fptr-ptr:(Tensor* w_hh), *$fptr-ptr:(Tensor* b_ih), *$fptr-ptr:(Tensor* b_hh)));
   }|] >>= newForeignPtr deleteTensor


--  roll roll
--
roll :: ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
roll self shifts dims =  V.unsafeWith shifts $ \shifts__array -> let shifts__size = fromIntegral (V.length shifts) in V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in
  [C.block|Tensor* {
    return new Tensor(VariableType::roll(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* shifts__array), $(size_t shifts__size)), ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


--  rot90 rot90
--
rot90 :: ForeignPtr CTensor -> Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
rot90 self k dims =  V.unsafeWith dims $ \dims__array -> let dims__size = fromIntegral (V.length dims) in
  [C.block|Tensor* {
    return new Tensor(VariableType::rot90(*$fptr-ptr:(Tensor* self), $(int64_t k), ArrayRef<int64_t>($(int64_t* dims__array), $(size_t dims__size))));
   }|] >>= newForeignPtr deleteTensor


--  round round
--
round :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
round self =
  [C.block|Tensor* {
    return new Tensor(VariableType::round(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  round_ round_
--
round_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
round_ self =
  [C.block|void {
    VariableType::round_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  round_out round_out
--
round_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
round_out out self =
  [C.block|void {
    VariableType::round_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  rrelu rrelu
--
rrelu :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
rrelu self lower upper training generator =
  [C.block|Tensor* {
    return new Tensor(VariableType::rrelu(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* lower), *$fptr-ptr:(Scalar* upper), $(bool training), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


--  rrelu_ rrelu_
--
rrelu_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
rrelu_ self lower upper training generator =
  [C.block|void {
    VariableType::rrelu_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* lower), *$fptr-ptr:(Scalar* upper), $(bool training), $(Generator* generator));
   }|] >> pure self


--  rrelu_with_noise rrelu_with_noise
--
rrelu_with_noise :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
rrelu_with_noise self noise lower upper training generator =
  [C.block|Tensor* {
    return new Tensor(VariableType::rrelu_with_noise(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* noise), *$fptr-ptr:(Scalar* lower), *$fptr-ptr:(Scalar* upper), $(bool training), $(Generator* generator)));
   }|] >>= newForeignPtr deleteTensor


--  rrelu_with_noise_ rrelu_with_noise_
--
rrelu_with_noise_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
rrelu_with_noise_ self noise lower upper training generator =
  [C.block|void {
    VariableType::rrelu_with_noise_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* noise), *$fptr-ptr:(Scalar* lower), *$fptr-ptr:(Scalar* upper), $(bool training), $(Generator* generator));
   }|] >> pure self


--  rrelu_with_noise_out rrelu_with_noise_out
--
rrelu_with_noise_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> CBool -> Ptr CGenerator -> IO (ForeignPtr CTensor)
rrelu_with_noise_out out self noise lower upper training generator =
  [C.block|void {
    VariableType::rrelu_with_noise_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* noise), *$fptr-ptr:(Scalar* lower), *$fptr-ptr:(Scalar* upper), $(bool training), $(Generator* generator));
   }|] >> pure (out)


--  rsqrt rsqrt
--
rsqrt :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rsqrt self =
  [C.block|Tensor* {
    return new Tensor(VariableType::rsqrt(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  rsqrt_ rsqrt_
--
rsqrt_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rsqrt_ self =
  [C.block|void {
    VariableType::rsqrt_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  rsqrt_out rsqrt_out
--
rsqrt_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
rsqrt_out out self =
  [C.block|void {
    VariableType::rsqrt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  rsub rsub
--
rsub :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
rsub self other alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::rsub(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  rsub rsub__1
--
rsub__1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
rsub__1 self other alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::rsub(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  s_native_addmm s_native_addmm
--
s_native_addmm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
s_native_addmm self mat1 mat2 beta alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::s_native_addmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  s_native_addmm_ s_native_addmm_
--
s_native_addmm_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
s_native_addmm_ self mat1 mat2 beta alpha =
  [C.block|void {
    VariableType::s_native_addmm_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  s_native_addmm_out s_native_addmm_out
--
s_native_addmm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
s_native_addmm_out out self mat1 mat2 beta alpha =
  [C.block|void {
    VariableType::s_native_addmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  scalar_tensor scalar_tensor
--
scalar_tensor :: ForeignPtr CScalar -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
scalar_tensor s options =
  [C.block|Tensor* {
    return new Tensor(VariableType::scalar_tensor(*$fptr-ptr:(Scalar* s), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  scatter scatter
--
scatter :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
scatter self dim index src =
  [C.block|Tensor* {
    return new Tensor(VariableType::scatter(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* src)));
   }|] >>= newForeignPtr deleteTensor


--  scatter scatter__1
--
scatter__1 :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
scatter__1 self dim index value =
  [C.block|Tensor* {
    return new Tensor(VariableType::scatter(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


--  scatter_ scatter_
--
scatter_ :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
scatter_ self dim index src =
  [C.block|void {
    VariableType::scatter_(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* src));
   }|] >> pure self


--  scatter_ scatter___1
--
scatter___1 :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
scatter___1 self dim index value =
  [C.block|void {
    VariableType::scatter_(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


--  scatter_add scatter_add
--
scatter_add :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
scatter_add self dim index src =
  [C.block|Tensor* {
    return new Tensor(VariableType::scatter_add(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* src)));
   }|] >>= newForeignPtr deleteTensor


--  scatter_add_ scatter_add_
--
scatter_add_ :: ForeignPtr CTensor -> Int64 -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
scatter_add_ self dim index src =
  [C.block|void {
    VariableType::scatter_add_(*$fptr-ptr:(Tensor* self), $(int64_t dim), *$fptr-ptr:(Tensor* index), *$fptr-ptr:(Tensor* src));
   }|] >> pure self


--  select select
--
select :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
select self dim index =
  [C.block|Tensor* {
    return new Tensor(VariableType::select(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(int64_t index)));
   }|] >>= newForeignPtr deleteTensor


--  selu selu
--
selu :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
selu self =
  [C.block|Tensor* {
    return new Tensor(VariableType::selu(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  selu_ selu_
--
selu_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
selu_ self =
  [C.block|void {
    VariableType::selu_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  set_ set_
--
set_ :: ForeignPtr CTensor -> Ptr CStorage -> IO (ForeignPtr CTensor)
set_ self source =
  [C.block|void {
    VariableType::set_(*$fptr-ptr:(Tensor* self), *$(Storage* source));
   }|] >> pure self


--  set_ set___1
--
set___1 :: ForeignPtr CTensor -> Ptr CStorage -> Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
set___1 self source storage_offset size stride =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in
  [C.block|void {
    VariableType::set_(*$fptr-ptr:(Tensor* self), *$(Storage* source), $(int64_t storage_offset), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)));
   }|] >> pure self


--  set_ set___2
--
set___2 :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
set___2 self source =
  [C.block|void {
    VariableType::set_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* source));
   }|] >> pure self


--  set_ set___3
--
set___3 :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
set___3 self =
  [C.block|void {
    VariableType::set_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  set_data set_data
--
set_data :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO ()
set_data self new_data =
  [C.block|void {
    return VariableType::set_data(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* new_data));
   }|]


--  sigmoid sigmoid
--
sigmoid :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sigmoid self =
  [C.block|Tensor* {
    return new Tensor(VariableType::sigmoid(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  sigmoid_ sigmoid_
--
sigmoid_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sigmoid_ self =
  [C.block|void {
    VariableType::sigmoid_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  sigmoid_out sigmoid_out
--
sigmoid_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sigmoid_out out self =
  [C.block|void {
    VariableType::sigmoid_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  sign sign
--
sign :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sign self =
  [C.block|Tensor* {
    return new Tensor(VariableType::sign(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  sign_ sign_
--
sign_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sign_ self =
  [C.block|void {
    VariableType::sign_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  sign_out sign_out
--
sign_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sign_out out self =
  [C.block|void {
    VariableType::sign_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  sin sin
--
sin :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sin self =
  [C.block|Tensor* {
    return new Tensor(VariableType::sin(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  sin_ sin_
--
sin_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sin_ self =
  [C.block|void {
    VariableType::sin_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  sin_out sin_out
--
sin_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sin_out out self =
  [C.block|void {
    VariableType::sin_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  sinh sinh
--
sinh :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sinh self =
  [C.block|Tensor* {
    return new Tensor(VariableType::sinh(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  sinh_ sinh_
--
sinh_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sinh_ self =
  [C.block|void {
    VariableType::sinh_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  sinh_out sinh_out
--
sinh_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sinh_out out self =
  [C.block|void {
    VariableType::sinh_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  size size
--
size :: ForeignPtr CTensor -> Int64 -> IO (Int64)
size self dim =
  [C.block|int64_t {
    return VariableType::size(*$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|]


--  slice slice
--
slice :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
slice self dim start end step =
  [C.block|Tensor* {
    return new Tensor(VariableType::slice(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(int64_t start), $(int64_t end), $(int64_t step)));
   }|] >>= newForeignPtr deleteTensor


--  slogdet slogdet
--
slogdet :: ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
slogdet self =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::slogdet(*$fptr-ptr:(Tensor* self)));
   }|] >>= unTupleTensorTensor


--  smm smm
--
smm :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
smm self mat2 =
  [C.block|Tensor* {
    return new Tensor(VariableType::smm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat2)));
   }|] >>= newForeignPtr deleteTensor


--  smooth_l1_loss smooth_l1_loss
--
smooth_l1_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
smooth_l1_loss self target reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::smooth_l1_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  smooth_l1_loss_out smooth_l1_loss_out
--
smooth_l1_loss_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
smooth_l1_loss_out out self target reduction =
  [C.block|void {
    VariableType::smooth_l1_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction));
   }|] >> pure (out)


--  soft_margin_loss soft_margin_loss
--
soft_margin_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
soft_margin_loss self target reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::soft_margin_loss(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  soft_margin_loss_out soft_margin_loss_out
--
soft_margin_loss_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
soft_margin_loss_out out self target reduction =
  [C.block|void {
    VariableType::soft_margin_loss_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* target), $(int64_t reduction));
   }|] >> pure (out)


--  softmax softmax
--
softmax :: ForeignPtr CTensor -> Int64 -> Int8 -> IO (ForeignPtr CTensor)
softmax self dim dtype =
  [C.block|Tensor* {
    return new Tensor(VariableType::softmax(*$fptr-ptr:(Tensor* self), $(int64_t dim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  softplus softplus
--
softplus :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
softplus self beta threshold =
  [C.block|Tensor* {
    return new Tensor(VariableType::softplus(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* threshold)));
   }|] >>= newForeignPtr deleteTensor


--  softplus_out softplus_out
--
softplus_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
softplus_out out self beta threshold =
  [C.block|void {
    VariableType::softplus_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* threshold));
   }|] >> pure (out)


--  softshrink softshrink
--
softshrink :: ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
softshrink self lambd =
  [C.block|Tensor* {
    return new Tensor(VariableType::softshrink(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* lambd)));
   }|] >>= newForeignPtr deleteTensor


--  softshrink_out softshrink_out
--
softshrink_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
softshrink_out out self lambd =
  [C.block|void {
    VariableType::softshrink_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* lambd));
   }|] >> pure (out)


--  solve solve
--
solve :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
solve self a =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::solve(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a)));
   }|] >>= unTupleTensorTensor


--  solve_out solve_out
--
solve_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
solve_out solution lu self a =
  [C.block|void {
    VariableType::solve_out(*$fptr-ptr:(Tensor* solution), *$fptr-ptr:(Tensor* lu), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a));
   }|] >> pure (solution,lu)


--  sort sort
--
sort :: ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
sort self dim descending =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::sort(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool descending)));
   }|] >>= unTupleTensorTensor


--  sort_out sort_out
--
sort_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
sort_out values indices self dim descending =
  [C.block|void {
    VariableType::sort_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool descending));
   }|] >> pure (values,indices)


--  sparse_coo_tensor sparse_coo_tensor
--
sparse_coo_tensor :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
sparse_coo_tensor size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::sparse_coo_tensor(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  sparse_coo_tensor sparse_coo_tensor__1
--
sparse_coo_tensor__1 :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
sparse_coo_tensor__1 indices values options =
  [C.block|Tensor* {
    return new Tensor(VariableType::sparse_coo_tensor(*$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* values), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  sparse_coo_tensor sparse_coo_tensor__2
--
sparse_coo_tensor__2 :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
sparse_coo_tensor__2 indices values size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::sparse_coo_tensor(*$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* values), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  sparse_dim sparse_dim
--
sparse_dim :: ForeignPtr CTensor -> IO (Int64)
sparse_dim self =
  [C.block|int64_t {
    return VariableType::sparse_dim(*$fptr-ptr:(Tensor* self));
   }|]


--  sparse_mask sparse_mask
--
sparse_mask :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sparse_mask self mask =
  [C.block|Tensor* {
    return new Tensor(VariableType::sparse_mask(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mask)));
   }|] >>= newForeignPtr deleteTensor


--  sparse_resize_ sparse_resize_
--
sparse_resize_ :: ForeignPtr CTensor -> Vector Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
sparse_resize_ self size sparse_dim dense_dim =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::sparse_resize_(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(int64_t sparse_dim), $(int64_t dense_dim));
   }|] >> pure self


--  sparse_resize_and_clear_ sparse_resize_and_clear_
--
sparse_resize_and_clear_ :: ForeignPtr CTensor -> Vector Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
sparse_resize_and_clear_ self size sparse_dim dense_dim =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::sparse_resize_and_clear_(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), $(int64_t sparse_dim), $(int64_t dense_dim));
   }|] >> pure self


--  split split
--
split :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (Vector (Ptr CTensor))
split self split_size dim =
  [C.block|void* {
    return (void*)new std::vector<Tensor>(VariableType::split(*$fptr-ptr:(Tensor* self), $(int64_t split_size), $(int64_t dim)));
   }|] >>= unVectorTensor


--  split_with_sizes split_with_sizes
--
split_with_sizes :: ForeignPtr CTensor -> Vector Int64 -> Int64 -> IO (Vector (Ptr CTensor))
split_with_sizes self split_sizes dim =  V.unsafeWith split_sizes $ \split_sizes__array -> let split_sizes__size = fromIntegral (V.length split_sizes) in
  [C.block|void* {
    return (void*)new std::vector<Tensor>(VariableType::split_with_sizes(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* split_sizes__array), $(size_t split_sizes__size)), $(int64_t dim)));
   }|] >>= unVectorTensor


--  sqrt sqrt
--
sqrt :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sqrt self =
  [C.block|Tensor* {
    return new Tensor(VariableType::sqrt(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  sqrt_ sqrt_
--
sqrt_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sqrt_ self =
  [C.block|void {
    VariableType::sqrt_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  sqrt_out sqrt_out
--
sqrt_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
sqrt_out out self =
  [C.block|void {
    VariableType::sqrt_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  squeeze squeeze
--
squeeze :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
squeeze self =
  [C.block|Tensor* {
    return new Tensor(VariableType::squeeze(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  squeeze squeeze__1
--
squeeze__1 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
squeeze__1 self dim =
  [C.block|Tensor* {
    return new Tensor(VariableType::squeeze(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  squeeze_ squeeze_
--
squeeze_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
squeeze_ self =
  [C.block|void {
    VariableType::squeeze_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  squeeze_ squeeze___1
--
squeeze___1 :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
squeeze___1 self dim =
  [C.block|void {
    VariableType::squeeze_(*$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|] >> pure self


--  sspaddmm sspaddmm
--
sspaddmm :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sspaddmm self mat1 mat2 beta alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::sspaddmm(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  sspaddmm_out sspaddmm_out
--
sspaddmm_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sspaddmm_out out self mat1 mat2 beta alpha =
  [C.block|void {
    VariableType::sspaddmm_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* mat1), *$fptr-ptr:(Tensor* mat2), *$fptr-ptr:(Scalar* beta), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  stack stack
--
stack :: Vector (Ptr CTensor) -> Int64 -> IO (ForeignPtr CTensor)
stack tensors dim =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in
  [C.block|Tensor* {
    return new Tensor(VariableType::stack(pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size)), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  stack_out stack_out
--
stack_out :: ForeignPtr CTensor -> Vector (Ptr CTensor) -> Int64 -> IO (ForeignPtr CTensor)
stack_out out tensors dim =  V.unsafeWith tensors $ \tensors__array -> let tensors__size = fromIntegral (V.length tensors) in
  [C.block|void {
    VariableType::stack_out(*$fptr-ptr:(Tensor* out), pack_tensor_list($(Tensor** tensors__array), $(size_t tensors__size)), $(int64_t dim));
   }|] >> pure (out)


--  std std
--
std :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
std self unbiased =
  [C.block|Tensor* {
    return new Tensor(VariableType::std(*$fptr-ptr:(Tensor* self), $(bool unbiased)));
   }|] >>= newForeignPtr deleteTensor


--  std std__1
--
std__1 :: ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
std__1 self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::std(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  std_mean std_mean
--
std_mean :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
std_mean self unbiased =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::std_mean(*$fptr-ptr:(Tensor* self), $(bool unbiased)));
   }|] >>= unTupleTensorTensor


--  std_mean std_mean__1
--
std_mean__1 :: ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
std_mean__1 self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::std_mean(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


--  std_out std_out
--
std_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
std_out out self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|void {
    VariableType::std_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim));
   }|] >> pure (out)


--  stft stft
--
stft :: ForeignPtr CTensor -> Int64 -> Maybe Int64 -> Maybe Int64 -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor)
stft self n_fft hop_length win_length window normalized onesided =  let (hop_length__is_present, hop_length__value) = splitMaybe hop_length 0 in let (win_length__is_present, win_length__value) = splitMaybe win_length 0 in
  [C.block|Tensor* {
    return new Tensor(VariableType::stft(*$fptr-ptr:(Tensor* self), $(int64_t n_fft), ($(bool hop_length__is_present) ? make_optional($(int64_t hop_length__value)) : c10::nullopt), ($(bool win_length__is_present) ? make_optional($(int64_t win_length__value)) : c10::nullopt), *$fptr-ptr:(Tensor* window), $(bool normalized), $(bool onesided)));
   }|] >>= newForeignPtr deleteTensor


--  stride stride
--
stride :: ForeignPtr CTensor -> Int64 -> IO (Int64)
stride self dim =
  [C.block|int64_t {
    return VariableType::stride(*$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|]


--  sub sub
--
sub :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sub self other alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::sub(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  sub sub__1
--
sub__1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sub__1 self other alpha =
  [C.block|Tensor* {
    return new Tensor(VariableType::sub(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other), *$fptr-ptr:(Scalar* alpha)));
   }|] >>= newForeignPtr deleteTensor


--  sub_ sub_
--
sub_ :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sub_ self other alpha =
  [C.block|void {
    VariableType::sub_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  sub_ sub___1
--
sub___1 :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sub___1 self other alpha =
  [C.block|void {
    VariableType::sub_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure self


--  sub_out sub_out
--
sub_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
sub_out out self other alpha =
  [C.block|void {
    VariableType::sub_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), *$fptr-ptr:(Scalar* alpha));
   }|] >> pure (out)


--  sum sum
--
sum :: ForeignPtr CTensor -> Int8 -> IO (ForeignPtr CTensor)
sum self dtype =
  [C.block|Tensor* {
    return new Tensor(VariableType::sum(*$fptr-ptr:(Tensor* self), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  sum sum__1
--
sum__1 :: ForeignPtr CTensor -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
sum__1 self dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::sum(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype))));
   }|] >>= newForeignPtr deleteTensor


--  sum_out sum_out
--
sum_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> Int8 -> IO (ForeignPtr CTensor)
sum_out out self dim keepdim dtype =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|void {
    VariableType::sum_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool keepdim), static_cast<ScalarType>($(int8_t dtype)));
   }|] >> pure (out)


--  sum_to_size sum_to_size
--
sum_to_size :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
sum_to_size self size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::sum_to_size(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size))));
   }|] >>= newForeignPtr deleteTensor


--  svd svd
--
svd :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
svd self some compute_uv =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::svd(*$fptr-ptr:(Tensor* self), $(bool some), $(bool compute_uv)));
   }|] >>= unTupleTensorTensorTensor


--  svd_out svd_out
--
svd_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
svd_out u s v self some compute_uv =
  [C.block|void {
    VariableType::svd_out(*$fptr-ptr:(Tensor* u), *$fptr-ptr:(Tensor* s), *$fptr-ptr:(Tensor* v), *$fptr-ptr:(Tensor* self), $(bool some), $(bool compute_uv));
   }|] >> pure (u,s,v)


--  symeig symeig
--
symeig :: ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
symeig self eigenvectors upper =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::symeig(*$fptr-ptr:(Tensor* self), $(bool eigenvectors), $(bool upper)));
   }|] >>= unTupleTensorTensor


--  symeig_out symeig_out
--
symeig_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
symeig_out e v self eigenvectors upper =
  [C.block|void {
    VariableType::symeig_out(*$fptr-ptr:(Tensor* e), *$fptr-ptr:(Tensor* v), *$fptr-ptr:(Tensor* self), $(bool eigenvectors), $(bool upper));
   }|] >> pure (e,v)


--  t t
--
t :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
t self =
  [C.block|Tensor* {
    return new Tensor(VariableType::t(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  t_ t_
--
t_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
t_ self =
  [C.block|void {
    VariableType::t_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  take take
--
take :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
take self index =
  [C.block|Tensor* {
    return new Tensor(VariableType::take(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* index)));
   }|] >>= newForeignPtr deleteTensor


--  take_out take_out
--
take_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
take_out out self index =
  [C.block|void {
    VariableType::take_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* index));
   }|] >> pure (out)


--  tan tan
--
tan :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tan self =
  [C.block|Tensor* {
    return new Tensor(VariableType::tan(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  tan_ tan_
--
tan_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tan_ self =
  [C.block|void {
    VariableType::tan_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  tan_out tan_out
--
tan_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tan_out out self =
  [C.block|void {
    VariableType::tan_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  tanh tanh
--
tanh :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tanh self =
  [C.block|Tensor* {
    return new Tensor(VariableType::tanh(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  tanh_ tanh_
--
tanh_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tanh_ self =
  [C.block|void {
    VariableType::tanh_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  tanh_out tanh_out
--
tanh_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
tanh_out out self =
  [C.block|void {
    VariableType::tanh_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  tensordot tensordot
--
tensordot :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
tensordot self other dims_self dims_other =  V.unsafeWith dims_self $ \dims_self__array -> let dims_self__size = fromIntegral (V.length dims_self) in V.unsafeWith dims_other $ \dims_other__array -> let dims_other__size = fromIntegral (V.length dims_other) in
  [C.block|Tensor* {
    return new Tensor(VariableType::tensordot(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), ArrayRef<int64_t>($(int64_t* dims_self__array), $(size_t dims_self__size)), ArrayRef<int64_t>($(int64_t* dims_other__array), $(size_t dims_other__size))));
   }|] >>= newForeignPtr deleteTensor


--  thnn_conv2d thnn_conv2d
--
thnn_conv2d :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv2d self weight kernel_size bias stride padding =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::thnn_conv2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


--  thnn_conv2d_out thnn_conv2d_out
--
thnn_conv2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv2d_out out self weight kernel_size bias stride padding =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|void {
    VariableType::thnn_conv2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


--  thnn_conv3d thnn_conv3d
--
thnn_conv3d :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv3d self weight kernel_size bias stride padding =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|Tensor* {
    return new Tensor(VariableType::thnn_conv3d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size))));
   }|] >>= newForeignPtr deleteTensor


--  thnn_conv3d_out thnn_conv3d_out
--
thnn_conv3d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv3d_out out self weight kernel_size bias stride padding =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in
  [C.block|void {
    VariableType::thnn_conv3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)));
   }|] >> pure (out)


--  thnn_conv_depthwise2d thnn_conv_depthwise2d
--
thnn_conv_depthwise2d :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv_depthwise2d self weight kernel_size bias stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|Tensor* {
    return new Tensor(VariableType::thnn_conv_depthwise2d(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size))));
   }|] >>= newForeignPtr deleteTensor


--  thnn_conv_depthwise2d_out thnn_conv_depthwise2d_out
--
thnn_conv_depthwise2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> ForeignPtr CTensor -> Vector Int64 -> Vector Int64 -> Vector Int64 -> IO (ForeignPtr CTensor)
thnn_conv_depthwise2d_out out self weight kernel_size bias stride padding dilation =  V.unsafeWith kernel_size $ \kernel_size__array -> let kernel_size__size = fromIntegral (V.length kernel_size) in V.unsafeWith stride $ \stride__array -> let stride__size = fromIntegral (V.length stride) in V.unsafeWith padding $ \padding__array -> let padding__size = fromIntegral (V.length padding) in V.unsafeWith dilation $ \dilation__array -> let dilation__size = fromIntegral (V.length dilation) in
  [C.block|void {
    VariableType::thnn_conv_depthwise2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* weight), ArrayRef<int64_t>($(int64_t* kernel_size__array), $(size_t kernel_size__size)), *$fptr-ptr:(Tensor* bias), ArrayRef<int64_t>($(int64_t* stride__array), $(size_t stride__size)), ArrayRef<int64_t>($(int64_t* padding__array), $(size_t padding__size)), ArrayRef<int64_t>($(int64_t* dilation__array), $(size_t dilation__size)));
   }|] >> pure (out)


--  threshold threshold
--
threshold :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
threshold self threshold value =
  [C.block|Tensor* {
    return new Tensor(VariableType::threshold(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* threshold), *$fptr-ptr:(Scalar* value)));
   }|] >>= newForeignPtr deleteTensor


--  threshold_ threshold_
--
threshold_ :: ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
threshold_ self threshold value =
  [C.block|void {
    VariableType::threshold_(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* threshold), *$fptr-ptr:(Scalar* value));
   }|] >> pure self


--  threshold_out threshold_out
--
threshold_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CScalar -> ForeignPtr CScalar -> IO (ForeignPtr CTensor)
threshold_out out self threshold value =
  [C.block|void {
    VariableType::threshold_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Scalar* threshold), *$fptr-ptr:(Scalar* value));
   }|] >> pure (out)


--  to to
--
to :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> CBool -> CBool -> IO (ForeignPtr CTensor)
to self options non_blocking copy =
  [C.block|Tensor* {
    return new Tensor(VariableType::to(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options), $(bool non_blocking), $(bool copy)));
   }|] >>= newForeignPtr deleteTensor


--  to to__1
--
to__1 :: ForeignPtr CTensor -> Ptr CDevice -> Int8 -> CBool -> CBool -> IO (ForeignPtr CTensor)
to__1 self device dtype non_blocking copy =
  [C.block|Tensor* {
    return new Tensor(VariableType::to(*$fptr-ptr:(Tensor* self), *$(Device* device), static_cast<ScalarType>($(int8_t dtype)), $(bool non_blocking), $(bool copy)));
   }|] >>= newForeignPtr deleteTensor


--  to to__2
--
to__2 :: ForeignPtr CTensor -> Int8 -> CBool -> CBool -> IO (ForeignPtr CTensor)
to__2 self dtype non_blocking copy =
  [C.block|Tensor* {
    return new Tensor(VariableType::to(*$fptr-ptr:(Tensor* self), static_cast<ScalarType>($(int8_t dtype)), $(bool non_blocking), $(bool copy)));
   }|] >>= newForeignPtr deleteTensor


--  to to__3
--
to__3 :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> IO (ForeignPtr CTensor)
to__3 self other non_blocking copy =
  [C.block|Tensor* {
    return new Tensor(VariableType::to(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other), $(bool non_blocking), $(bool copy)));
   }|] >>= newForeignPtr deleteTensor


--  to_dense to_dense
--
to_dense :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
to_dense self =
  [C.block|Tensor* {
    return new Tensor(VariableType::to_dense(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  to_mkldnn to_mkldnn
--
to_mkldnn :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
to_mkldnn self =
  [C.block|Tensor* {
    return new Tensor(VariableType::to_mkldnn(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  to_sparse to_sparse
--
to_sparse :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
to_sparse self sparse_dim =
  [C.block|Tensor* {
    return new Tensor(VariableType::to_sparse(*$fptr-ptr:(Tensor* self), $(int64_t sparse_dim)));
   }|] >>= newForeignPtr deleteTensor


--  to_sparse to_sparse__1
--
to_sparse__1 :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
to_sparse__1 self =
  [C.block|Tensor* {
    return new Tensor(VariableType::to_sparse(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  topk topk
--
topk :: ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
topk self k dim largest sorted =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::topk(*$fptr-ptr:(Tensor* self), $(int64_t k), $(int64_t dim), $(bool largest), $(bool sorted)));
   }|] >>= unTupleTensorTensor


--  topk_out topk_out
--
topk_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
topk_out values indices self k dim largest sorted =
  [C.block|void {
    VariableType::topk_out(*$fptr-ptr:(Tensor* values), *$fptr-ptr:(Tensor* indices), *$fptr-ptr:(Tensor* self), $(int64_t k), $(int64_t dim), $(bool largest), $(bool sorted));
   }|] >> pure (values,indices)


--  trace trace
--
trace :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
trace self =
  [C.block|Tensor* {
    return new Tensor(VariableType::trace(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  transpose transpose
--
transpose :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
transpose self dim0 dim1 =
  [C.block|Tensor* {
    return new Tensor(VariableType::transpose(*$fptr-ptr:(Tensor* self), $(int64_t dim0), $(int64_t dim1)));
   }|] >>= newForeignPtr deleteTensor


--  transpose_ transpose_
--
transpose_ :: ForeignPtr CTensor -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
transpose_ self dim0 dim1 =
  [C.block|void {
    VariableType::transpose_(*$fptr-ptr:(Tensor* self), $(int64_t dim0), $(int64_t dim1));
   }|] >> pure self


--  trapz trapz
--
trapz :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
trapz y x dim =
  [C.block|Tensor* {
    return new Tensor(VariableType::trapz(*$fptr-ptr:(Tensor* y), *$fptr-ptr:(Tensor* x), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  trapz trapz__1
--
trapz__1 :: ForeignPtr CTensor -> CDouble -> Int64 -> IO (ForeignPtr CTensor)
trapz__1 y dx dim =
  [C.block|Tensor* {
    return new Tensor(VariableType::trapz(*$fptr-ptr:(Tensor* y), $(double dx), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  triangular_solve triangular_solve
--
triangular_solve :: ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
triangular_solve self a upper transpose unitriangular =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::triangular_solve(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a), $(bool upper), $(bool transpose), $(bool unitriangular)));
   }|] >>= unTupleTensorTensor


--  triangular_solve_out triangular_solve_out
--
triangular_solve_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
triangular_solve_out x m self a upper transpose unitriangular =
  [C.block|void {
    VariableType::triangular_solve_out(*$fptr-ptr:(Tensor* x), *$fptr-ptr:(Tensor* m), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* a), $(bool upper), $(bool transpose), $(bool unitriangular));
   }|] >> pure (x,m)


--  tril tril
--
tril :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
tril self diagonal =
  [C.block|Tensor* {
    return new Tensor(VariableType::tril(*$fptr-ptr:(Tensor* self), $(int64_t diagonal)));
   }|] >>= newForeignPtr deleteTensor


--  tril_ tril_
--
tril_ :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
tril_ self diagonal =
  [C.block|void {
    VariableType::tril_(*$fptr-ptr:(Tensor* self), $(int64_t diagonal));
   }|] >> pure self


--  tril_indices tril_indices
--
tril_indices :: Int64 -> Int64 -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
tril_indices row col offset options =
  [C.block|Tensor* {
    return new Tensor(VariableType::tril_indices($(int64_t row), $(int64_t col), $(int64_t offset), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  tril_out tril_out
--
tril_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
tril_out out self diagonal =
  [C.block|void {
    VariableType::tril_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t diagonal));
   }|] >> pure (out)


--  triplet_margin_loss triplet_margin_loss
--
triplet_margin_loss :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> CDouble -> CDouble -> CDouble -> CBool -> Int64 -> IO (ForeignPtr CTensor)
triplet_margin_loss anchor positive negative margin p eps swap reduction =
  [C.block|Tensor* {
    return new Tensor(VariableType::triplet_margin_loss(*$fptr-ptr:(Tensor* anchor), *$fptr-ptr:(Tensor* positive), *$fptr-ptr:(Tensor* negative), $(double margin), $(double p), $(double eps), $(bool swap), $(int64_t reduction)));
   }|] >>= newForeignPtr deleteTensor


--  triu triu
--
triu :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
triu self diagonal =
  [C.block|Tensor* {
    return new Tensor(VariableType::triu(*$fptr-ptr:(Tensor* self), $(int64_t diagonal)));
   }|] >>= newForeignPtr deleteTensor


--  triu_ triu_
--
triu_ :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
triu_ self diagonal =
  [C.block|void {
    VariableType::triu_(*$fptr-ptr:(Tensor* self), $(int64_t diagonal));
   }|] >> pure self


--  triu_indices triu_indices
--
triu_indices :: Int64 -> Int64 -> Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
triu_indices row col offset options =
  [C.block|Tensor* {
    return new Tensor(VariableType::triu_indices($(int64_t row), $(int64_t col), $(int64_t offset), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  triu_out triu_out
--
triu_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
triu_out out self diagonal =
  [C.block|void {
    VariableType::triu_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), $(int64_t diagonal));
   }|] >> pure (out)


--  trunc trunc
--
trunc :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
trunc self =
  [C.block|Tensor* {
    return new Tensor(VariableType::trunc(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  trunc_ trunc_
--
trunc_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
trunc_ self =
  [C.block|void {
    VariableType::trunc_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  trunc_out trunc_out
--
trunc_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
trunc_out out self =
  [C.block|void {
    VariableType::trunc_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self));
   }|] >> pure (out)


--  type_as type_as
--
type_as :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
type_as self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::type_as(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  unbind unbind
--
unbind :: ForeignPtr CTensor -> Int64 -> IO (Vector (Ptr CTensor))
unbind self dim =
  [C.block|void* {
    return (void*)new std::vector<Tensor>(VariableType::unbind(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= unVectorTensor


--  unfold unfold
--
unfold :: ForeignPtr CTensor -> Int64 -> Int64 -> Int64 -> IO (ForeignPtr CTensor)
unfold self dimension size step =
  [C.block|Tensor* {
    return new Tensor(VariableType::unfold(*$fptr-ptr:(Tensor* self), $(int64_t dimension), $(int64_t size), $(int64_t step)));
   }|] >>= newForeignPtr deleteTensor


--  uniform_ uniform_
--
uniform_ :: ForeignPtr CTensor -> CDouble -> CDouble -> Ptr CGenerator -> IO (ForeignPtr CTensor)
uniform_ self from to generator =
  [C.block|void {
    VariableType::uniform_(*$fptr-ptr:(Tensor* self), $(double from), $(double to), $(Generator* generator));
   }|] >> pure self


--  unique_consecutive unique_consecutive
--
unique_consecutive :: ForeignPtr CTensor -> CBool -> CBool -> Maybe Int64 -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
unique_consecutive self return_inverse return_counts dim =  let (dim__is_present, dim__value) = splitMaybe dim 0 in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::unique_consecutive(*$fptr-ptr:(Tensor* self), $(bool return_inverse), $(bool return_counts), ($(bool dim__is_present) ? make_optional($(int64_t dim__value)) : c10::nullopt)));
   }|] >>= unTupleTensorTensorTensor


--  unique_dim unique_dim
--
unique_dim :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
unique_dim self dim sorted return_inverse return_counts =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::unique_dim(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool sorted), $(bool return_inverse), $(bool return_counts)));
   }|] >>= unTupleTensorTensorTensor


--  unique_dim_consecutive unique_dim_consecutive
--
unique_dim_consecutive :: ForeignPtr CTensor -> Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor, ForeignPtr CTensor)
unique_dim_consecutive self dim return_inverse return_counts =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor,Tensor>(VariableType::unique_dim_consecutive(*$fptr-ptr:(Tensor* self), $(int64_t dim), $(bool return_inverse), $(bool return_counts)));
   }|] >>= unTupleTensorTensorTensor


--  unsqueeze unsqueeze
--
unsqueeze :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
unsqueeze self dim =
  [C.block|Tensor* {
    return new Tensor(VariableType::unsqueeze(*$fptr-ptr:(Tensor* self), $(int64_t dim)));
   }|] >>= newForeignPtr deleteTensor


--  unsqueeze_ unsqueeze_
--
unsqueeze_ :: ForeignPtr CTensor -> Int64 -> IO (ForeignPtr CTensor)
unsqueeze_ self dim =
  [C.block|void {
    VariableType::unsqueeze_(*$fptr-ptr:(Tensor* self), $(int64_t dim));
   }|] >> pure self


--  upsample_bicubic2d upsample_bicubic2d
--
upsample_bicubic2d :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
upsample_bicubic2d self output_size align_corners =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::upsample_bicubic2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners)));
   }|] >>= newForeignPtr deleteTensor


--  upsample_bicubic2d_out upsample_bicubic2d_out
--
upsample_bicubic2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
upsample_bicubic2d_out out self output_size align_corners =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::upsample_bicubic2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners));
   }|] >> pure (out)


--  upsample_bilinear2d upsample_bilinear2d
--
upsample_bilinear2d :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
upsample_bilinear2d self output_size align_corners =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::upsample_bilinear2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners)));
   }|] >>= newForeignPtr deleteTensor


--  upsample_bilinear2d_out upsample_bilinear2d_out
--
upsample_bilinear2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
upsample_bilinear2d_out out self output_size align_corners =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::upsample_bilinear2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners));
   }|] >> pure (out)


--  upsample_linear1d upsample_linear1d
--
upsample_linear1d :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
upsample_linear1d self output_size align_corners =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::upsample_linear1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners)));
   }|] >>= newForeignPtr deleteTensor


--  upsample_linear1d_out upsample_linear1d_out
--
upsample_linear1d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
upsample_linear1d_out out self output_size align_corners =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::upsample_linear1d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners));
   }|] >> pure (out)


--  upsample_nearest1d upsample_nearest1d
--
upsample_nearest1d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
upsample_nearest1d self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::upsample_nearest1d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


--  upsample_nearest1d_out upsample_nearest1d_out
--
upsample_nearest1d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
upsample_nearest1d_out out self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::upsample_nearest1d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out)


--  upsample_nearest2d upsample_nearest2d
--
upsample_nearest2d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
upsample_nearest2d self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::upsample_nearest2d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


--  upsample_nearest2d_out upsample_nearest2d_out
--
upsample_nearest2d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
upsample_nearest2d_out out self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::upsample_nearest2d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out)


--  upsample_nearest3d upsample_nearest3d
--
upsample_nearest3d :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
upsample_nearest3d self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::upsample_nearest3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size))));
   }|] >>= newForeignPtr deleteTensor


--  upsample_nearest3d_out upsample_nearest3d_out
--
upsample_nearest3d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
upsample_nearest3d_out out self output_size =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::upsample_nearest3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)));
   }|] >> pure (out)


--  upsample_trilinear3d upsample_trilinear3d
--
upsample_trilinear3d :: ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
upsample_trilinear3d self output_size align_corners =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::upsample_trilinear3d(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners)));
   }|] >>= newForeignPtr deleteTensor


--  upsample_trilinear3d_out upsample_trilinear3d_out
--
upsample_trilinear3d_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> IO (ForeignPtr CTensor)
upsample_trilinear3d_out out self output_size align_corners =  V.unsafeWith output_size $ \output_size__array -> let output_size__size = fromIntegral (V.length output_size) in
  [C.block|void {
    VariableType::upsample_trilinear3d_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* output_size__array), $(size_t output_size__size)), $(bool align_corners));
   }|] >> pure (out)


--  values values
--
values :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
values self =
  [C.block|Tensor* {
    return new Tensor(VariableType::values(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  var var
--
var :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor)
var self unbiased =
  [C.block|Tensor* {
    return new Tensor(VariableType::var(*$fptr-ptr:(Tensor* self), $(bool unbiased)));
   }|] >>= newForeignPtr deleteTensor


--  var var__1
--
var__1 :: ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
var__1 self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|Tensor* {
    return new Tensor(VariableType::var(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim)));
   }|] >>= newForeignPtr deleteTensor


--  var_mean var_mean
--
var_mean :: ForeignPtr CTensor -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
var_mean self unbiased =
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::var_mean(*$fptr-ptr:(Tensor* self), $(bool unbiased)));
   }|] >>= unTupleTensorTensor


--  var_mean var_mean__1
--
var_mean__1 :: ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor, ForeignPtr CTensor)
var_mean__1 self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|void* {
    return (void*)new std::tuple<Tensor,Tensor>(VariableType::var_mean(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim)));
   }|] >>= unTupleTensorTensor


--  var_out var_out
--
var_out :: ForeignPtr CTensor -> ForeignPtr CTensor -> Vector Int64 -> CBool -> CBool -> IO (ForeignPtr CTensor)
var_out out self dim unbiased keepdim =  V.unsafeWith dim $ \dim__array -> let dim__size = fromIntegral (V.length dim) in
  [C.block|void {
    VariableType::var_out(*$fptr-ptr:(Tensor* out), *$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* dim__array), $(size_t dim__size)), $(bool unbiased), $(bool keepdim));
   }|] >> pure (out)


--  view view
--
view :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
view self size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::view(*$fptr-ptr:(Tensor* self), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size))));
   }|] >>= newForeignPtr deleteTensor


--  view_as view_as
--
view_as :: ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
view_as self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::view_as(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  where whereX
--
whereX :: ForeignPtr CTensor -> ForeignPtr CTensor -> ForeignPtr CTensor -> IO (ForeignPtr CTensor)
whereX condition self other =
  [C.block|Tensor* {
    return new Tensor(VariableType::where(*$fptr-ptr:(Tensor* condition), *$fptr-ptr:(Tensor* self), *$fptr-ptr:(Tensor* other)));
   }|] >>= newForeignPtr deleteTensor


--  where where__1
--
where__1 :: ForeignPtr CTensor -> IO (Vector (Ptr CTensor))
where__1 condition =
  [C.block|void* {
    return (void*)new std::vector<Tensor>(VariableType::where(*$fptr-ptr:(Tensor* condition)));
   }|] >>= unVectorTensor


--  zero_ zero_
--
zero_ :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
zero_ self =
  [C.block|void {
    VariableType::zero_(*$fptr-ptr:(Tensor* self));
   }|] >> pure self


--  zeros zeros
--
zeros :: Vector Int64 -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
zeros size options =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|Tensor* {
    return new Tensor(VariableType::zeros(ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  zeros_like zeros_like
--
zeros_like :: ForeignPtr CTensor -> IO (ForeignPtr CTensor)
zeros_like self =
  [C.block|Tensor* {
    return new Tensor(VariableType::zeros_like(*$fptr-ptr:(Tensor* self)));
   }|] >>= newForeignPtr deleteTensor


--  zeros_like zeros_like__1
--
zeros_like__1 :: ForeignPtr CTensor -> ForeignPtr CTensorOptions -> IO (ForeignPtr CTensor)
zeros_like__1 self options =
  [C.block|Tensor* {
    return new Tensor(VariableType::zeros_like(*$fptr-ptr:(Tensor* self), *$fptr-ptr:(TensorOptions* options)));
   }|] >>= newForeignPtr deleteTensor


--  zeros_out zeros_out
--
zeros_out :: ForeignPtr CTensor -> Vector Int64 -> IO (ForeignPtr CTensor)
zeros_out out size =  V.unsafeWith size $ \size__array -> let size__size = fromIntegral (V.length size) in
  [C.block|void {
    VariableType::zeros_out(*$fptr-ptr:(Tensor* out), ArrayRef<int64_t>($(int64_t* size__array), $(size_t size__size)));
   }|] >> pure (out)

