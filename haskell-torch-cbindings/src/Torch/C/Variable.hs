{-# LANGUAGE FlexibleContexts, QuasiQuotes, ScopedTypeVariables, TemplateHaskell #-}

module Torch.C.Variable where
import           Control.Monad.Extra
import qualified Data.ByteString        as BS
import qualified Data.ByteString.Unsafe as BS
import           Data.Coerce
import           Data.Monoid            ((<>))
import           Data.Text              (Text)
import qualified Data.Text              as T
import qualified Data.Text.Encoding     as T
import qualified Data.Text.Foreign      as T
import qualified Data.Text.IO           as T
import           Data.Vector.Storable   (Vector)
import qualified Data.Vector.Storable   as V
import           Foreign.C.String
import           Foreign.C.Types
import           Foreign.ForeignPtr
import           Foreign.Marshal.Alloc
import           Foreign.Marshal.Array
import           Foreign.Ptr
import           GHC.Int
import qualified Language.C.Inline      as C
import qualified Language.C.Inline.Cpp  as C
import           Torch.C.Tensor         (deleteTensor, unVectorTensor)
import           Torch.C.Types

C.context (tensorCtx <> C.funCtx)

C.include "<torch/csrc/autograd/variable.h>"
C.include "<torch/csrc/autograd/function.h>"
C.include "<torch/csrc/autograd/generated/VariableType.h>"
C.include "<torch/csrc/autograd/grad_mode.h>"
C.include "<torch/csrc/jit/tracer.h>"
C.include "<torch/csrc/jit/ir.h>"
C.include "<torch/csrc/jit/export.h>"
C.include "<torch/csrc/jit/import.h>"

C.using "namespace std"
C.using "namespace torch::autograd"
C.using "namespace at"

C.verbatim "using TracingState    = torch::jit::tracer::TracingState;"
C.verbatim "using JitNode         = torch::jit::Node;"
C.verbatim "using JitValue        = torch::jit::Value;"
C.verbatim "using JitIValue       = torch::jit::IValue;"
C.verbatim "using JitBlock        = torch::jit::Block;"
C.verbatim "using JitType         = ::c10::Type;"
C.verbatim "using JitScriptModule = torch::jit::script::Module;"

C.verbatim "extern \"C\" void delete_variable(Variable* o) { delete o; }"
C.verbatim "extern \"C\" void delete_tracing_state(shared_ptr<TracingState>* o) { delete o; }"
C.verbatim "extern \"C\" void delete_jit_module(JitScriptModule* o) { delete o; }"

foreign import ccall "&delete_variable" deleteVariable :: FunPtr (Ptr CVariable -> IO ())
foreign import ccall "&delete_tracing_state" deleteTracingState :: FunPtr (Ptr CTracingState -> IO ())
foreign import ccall "&delete_jit_module" deleteJitModule :: FunPtr (Ptr CJitScriptModule -> IO ())

textPeekCString :: CString -> IO Text
textPeekCString cs = do
  bs <- BS.unsafePackCString cs
  return $! T.decodeUtf8 bs

withTextCString :: Text -> (CString -> IO a) -> IO a
withTextCString t f = BS.useAsCString (T.encodeUtf8 t) f

set_grad_enabled :: CBool -> IO ()
set_grad_enabled b = [C.exp|void { GradMode::set_enabled($(bool b)) }|]

grad_enabled :: IO CBool
grad_enabled = [C.exp|bool { GradMode::is_enabled() }|]

delete :: Coercible a (ForeignPtr CTensor) => a -> IO ()
delete v = [C.exp|void { delete ((Variable*) $fptr-ptr:(Tensor *v)); }|]

version :: Coercible a (ForeignPtr CTensor) => a -> IO CInt
version v = [C.exp|int { ((Variable*) $fptr-ptr:(Tensor *v))->current_version() }|]

gradient_function :: Coercible a (ForeignPtr CTensor) => a -> IO (Ptr CNode)
gradient_function v = [C.exp|Node* { ((Variable*) $fptr-ptr:(Tensor *v))->grad_fn().get() }|]

function_name f = do
  s <- [C.exp|char* { strdup($(Node *f)->name().c_str()) }|]
  s' <- peekCString s
  free s
  pure s'

function_nr_inputs f = [C.exp|int { $(Node *f)->num_inputs() }|]
function_nr_outputs f = [C.exp|int { $(Node *f)->num_outputs() }|]

unVectorEdge :: Ptr () -> IO (Vector (Ptr CEdge))
unVectorEdge ptr = do
  s <- [C.exp|size_t { ((std::vector<Edge>*)$(void* ptr))->size() }|]
  r <- V.generateM (fromIntegral s)
    (\i -> let i' = fromIntegral i
          in [C.exp|Edge *{ &(((std::vector<Edge>*)$(void* ptr))->at($(int i'))) }|])
  pure r

C.verbatim "using Stack = std::vector<JitIValue>;"
C.verbatim "Stack pack_variable_list(Variable** arr, size_t len) { std::vector<JitIValue> v; for(size_t i = 0; i < len; i++) { v.push_back(JitIValue(*(arr[i]))); }; return v; }"

unVectorVariable :: Ptr () -> IO (Vector (Ptr CVariable))
unVectorVariable ptr = do
  s <- [C.exp|size_t { ((std::vector<JitIValue>*)$(void* ptr))->size() }|]
  r <- V.generateM (fromIntegral s)
    (\i -> let i' = fromIntegral i
          in [C.exp|Variable *{ new Variable(((std::vector<JitIValue>*)$(void* ptr))->at($(int i')).toTensor()) }|])
  [C.exp|void { delete ((std::vector<JitIValue>*)$(void* ptr)) }|]
  pure r

enter_trace :: Vector (Ptr CVariable) -> IO (Ptr CTracingState, Vector (Ptr CVariable))
enter_trace vs =
  V.unsafeWith vs
  $ \variables__array ->
      let variables__size = fromIntegral (V.length vs) in
        do
          ps <- mallocArray 2
          [C.block|void {
      void **ps = $(void **ps);
      shared_ptr<TracingState> state;
      Stack trace_vars_in;
      Stack input_vars = pack_variable_list($(Variable** variables__array), $(size_t variables__size));
      std::vector<TypePtr> input_types;
      input_types.reserve(input_vars.size());
      for (size_t i = 0; i < input_vars.size(); i++) {
         input_types.push_back(TensorType::get());
      }
      auto input_typeptr = TupleType::create(std::move(input_types));
      std::tie(state, trace_vars_in) = torch::jit::tracer::enter(torch::jit::tracer::TypedStack(input_vars, input_typeptr));
      cout << "State has N references " << state.use_count() << endl;
      shared_ptr<TracingState> *statePtr =
            new shared_ptr<TracingState>(state);
      cout << "State has N references " << state.use_count() << endl;
      ps[0] = state.get();
      ps[1] = new Stack(trace_vars_in);
      return;
          }|]
          tracingState <- [C.exp|TracingState* { (TracingState*) $(void **ps)[0] }|]
          traceVarsIn <- [C.exp|JitIValue* { (JitIValue*) $(void **ps)[1] }|]
          ts <- unVectorVariable (castPtr traceVarsIn)
          pure (tracingState, ts)

exit_trace :: Vector (Ptr CVariable) -> IO ()
exit_trace vs = do
  V.unsafeWith vs
  $ \variables__array ->
      let variables__size = fromIntegral (V.length vs) in
        [C.exp|void { torch::jit::tracer::exit(pack_variable_list($(Variable** variables__array), $(size_t variables__size))) } |]

print_tracing_state_graph :: Ptr CTracingState -> IO ()
print_tracing_state_graph ts =
  [C.block|void{
      cout << *$(TracingState* ts)->graph << endl;
  }|]

-- TODO https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/export.h
-- TORCH_API std::tuple<std::string, RawDataExportMap> export_onnx(
--     const std::shared_ptr<Graph>& graph,
--     const std::map<std::string, at::Tensor>& initializers,
--     int64_t onnx_opset_version,
--     const std::unordered_map<std::string, std::unordered_map<int64_t, std::string>>& dynamic_axes,
--     bool defer_weight_export = false,
--     ::torch::onnx::OperatorExportTypes operator_export_type =
--         ::torch::onnx::OperatorExportTypes::ONNX,
--     bool strip_doc_string = true,
--     bool keep_initializers_as_inputs = true);

print_tracing_state_graph_onnx :: Ptr CTracingState -> Vector (Ptr CVariable) -> CBool -> CInt -> IO ()
print_tracing_state_graph_onnx ts vs googleFormat opset = do
  V.unsafeWith vs
  $ \variables__array ->
      let variables__size = fromIntegral (V.length vs) in
         [C.block|void{
           std::map<std::string, at::Tensor> initializers;
           for(int i = 0; i < $(int variables__size); i++)
              initializers[std::to_string(i)] = *$(Variable** variables__array)[i];
           cout << pretty_print_onnx(
           $(TracingState* ts)->graph,
           initializers,
           $(int opset),
           true,
           ::torch::onnx::OperatorExportTypes::ONNX_ATEN,
           $(bool googleFormat)) << endl;
         }|]

-- TODO
-- TORCH_API std::string pretty_print_onnx(
--     const std::shared_ptr<Graph>& graph,
--     const std::map<std::string, at::Tensor>& initializers,
--     int64_t onnx_opset_version,
--     bool defer_weight_export,
--     ::torch::onnx::OperatorExportTypes operator_export_type =
--         ::torch::onnx::OperatorExportTypes::ONNX,
--     bool google_printer = false);

unVectorNode :: Ptr () -> IO (Vector (Ptr CNode))
unVectorNode ptr = do
  s <- [C.exp|size_t { ((std::vector<Node>*)$(void* ptr))->size() }|]
  r <- V.generateM (fromIntegral s)
    (\i -> let i' = fromIntegral i
          in [C.exp|Node *{ &((std::vector<Node>*)$(void* ptr))->at($(int i')) }|])
  [C.exp|void { delete ((std::vector<Node>*)$(void* ptr)) }|]
  pure r

pointerArrayToVector :: V.Storable a => Ptr a -> Int -> Vector (Ptr a)
pointerArrayToVector ptr n = V.generate n (\i -> advancePtr ptr i)

-- https://github.com/pytorch/pytorch/blob/f3fdbba666ee5106d999f7f46f34fab0e0855cab/torch/csrc/jit/ir.h
tracing_state_graph :: Ptr CTracingState -> IO (Vector (Ptr CJitValue)
                                             ,Vector (Ptr CJitNode)
                                             ,Vector (Ptr CJitValue)
                                             ,Ptr CJitNode
                                             ,Ptr CJitBlock)
tracing_state_graph s = do
  nr_is <- fromIntegral <$> [C.exp|int { $(TracingState* s)->graph->inputs().size() }|]
  is_ptr <- [C.exp|JitValue **{ const_cast<JitValue**>($(TracingState* s)->graph->inputs().data()) }|]
  nr_ns <- fromIntegral <$> [C.block|int {
    int i = 0;
    for(auto n : $(TracingState* s)->graph->nodes()) { i++; }
    return i;
}|]
  ns_ptr <- castPtr <$> [C.block|JitNode **{
    std::vector<JitNode*> *ns = new std::vector<JitNode*>;
    for(auto n : $(TracingState* s)->graph->nodes()) { ns->push_back(&*n); }
    return ns->data();
}|]
  nr_os <- fromIntegral <$> [C.exp|int { $(TracingState* s)->graph->outputs().size() }|]
  os_ptr <- [C.exp|JitValue **{ const_cast<JitValue**>($(TracingState* s)->graph->outputs().data()) }|]
  is_ptr' <- newForeignPtr_ is_ptr
  ns_ptr' <- newForeignPtr_ ns_ptr
  os_ptr' <- newForeignPtr_ os_ptr
  ret_ptr <- [C.exp|JitNode *{ $(TracingState* s)->graph->return_node() }|]
  block_ptr <- [C.exp|JitBlock *{ $(TracingState* s)->graph->block() }|]
  pure (V.unsafeFromForeignPtr0 is_ptr' nr_is
       ,V.unsafeFromForeignPtr0 ns_ptr' nr_ns
       ,V.unsafeFromForeignPtr0 os_ptr' nr_os
       ,ret_ptr
       ,block_ptr)

node_inputs :: Ptr CJitNode -> IO (Vector (Ptr CJitValue))
node_inputs n = do
  nr <- [C.exp|size_t { $(JitNode *n)->inputs().size() }|]
  ptr <- [C.exp|JitValue **{ const_cast<JitValue**>($(JitNode *n)->inputs().data()) }|]
  ptr' <- newForeignPtr_ ptr
  pure $ V.unsafeFromForeignPtr0 ptr' (fromIntegral nr)

node_outputs :: Ptr CJitNode -> IO (Vector (Ptr CJitValue))
node_outputs n = do
  nr <- [C.exp|size_t { $(JitNode *n)->outputs().size() }|]
  ptr <- [C.exp|JitValue **{ const_cast<JitValue**>($(JitNode *n)->outputs().data()) }|]
  ptr' <- newForeignPtr_ ptr
  pure $ V.unsafeFromForeignPtr0 ptr' (fromIntegral nr)

node_kind :: Ptr CJitNode -> IO String
node_kind n = [C.exp|char *{ (char*)$(JitNode *n)->kind().toQualString() }|] >>= peekCString

node_kind' :: Ptr CJitNode -> IO CInt
node_kind' n = [C.exp|int { $(JitNode *n)->kind() }|]

node_has_attribute :: Ptr CJitNode -> String -> IO CBool
node_has_attribute n str =
  withCString str (\s -> [C.exp|bool { $(JitNode *n)->hasAttributeS($(char *s)) }|])

node_attribute_kind :: Ptr CJitNode -> String -> IO AttributeKind
node_attribute_kind n str =
  toEnum . fromIntegral <$> withCString str (\s -> [C.exp|int { (int)$(JitNode *n)->kindOfS($(char *s)) }|])

node_get_attribute_float :: Ptr CJitNode -> String -> IO CFloat
node_get_attribute_float n str =
  withCString str (\s -> [C.exp|float { $(JitNode *n)->f(Symbol::attr($(char *s))) }|])

node_get_attribute_int :: Ptr CJitNode -> String -> IO CInt
node_get_attribute_int n str =
  withCString str (\s -> [C.exp|int { $(JitNode *n)->i(Symbol::attr($(char *s))) }|])

node_get_attribute_string :: Ptr CJitNode -> String -> IO String
node_get_attribute_string n str =
  withCString str (\s -> [C.exp|char *{ (char*)$(JitNode *n)->s(Symbol::attr($(char *s))).c_str() }|]) >>= peekCString

node_get_attribute_tensor :: Ptr CJitNode -> String -> IO (ForeignPtr CTensor)
node_get_attribute_tensor n str =
  withCString str (\s -> [C.exp|Tensor *{ new Tensor($(JitNode *n)->t(Symbol::attr($(char *s)))) }|])
  >>= newForeignPtr deleteTensor

-- TODO Reenable this one day
value_name :: Ptr CJitValue -> IO String
value_name v = [C.exp|char *{ (char*)$(JitValue *v)->debugName().c_str() }|] >>= peekCString

value_is_tensor :: Ptr CJitValue -> IO CBool
value_is_tensor v = [C.exp|bool { $(JitValue *v)->isCompleteTensor() }|]

-- TODO Handle these cases!

-- Only use on tensors!
value_sizes :: Ptr CJitValue -> IO (Vector Int64)
value_sizes v = do
  s <- [C.exp|int { ((CompleteTensorType*)&(*$(JitValue *v)->type()))->sizes().size() }|]
  d <- [C.exp|int64_t *{ const_cast<int64_t*>(((CompleteTensorType*)&(*$(JitValue *v)->type()))->sizes().data()) }|] >>= newForeignPtr_
  pure $ V.unsafeFromForeignPtr0 d (fromIntegral s)

value_scalar_type :: Ptr CJitValue -> IO ScalarType
value_scalar_type v = do
  (CInt s) <- [C.exp|int { (int) ((CompleteTensorType*)&(*$(JitValue *v)->type()))->scalarType() }|]
  pure $ toEnum $ fromIntegral s

value_type_kind :: Ptr CJitValue -> IO TypeKind
value_type_kind v = toEnum . fromIntegral <$> [C.exp|int { (int) $(JitValue *v)->type()->kind() }|]

value_type :: Ptr CJitValue -> IO (Ptr CType)
value_type v = [C.exp|JitType *{ $(JitValue *v)->type().get() }|]

type_scalar_type :: Ptr CType -> IO ScalarType
type_scalar_type t = do
  (CInt s) <- [C.exp|int { (int) ((CompleteTensorType*)($(JitType *t)))->scalarType() }|]
  pure $ toEnum $ fromIntegral s

type_kind :: Ptr CType -> IO TypeKind
type_kind t = toEnum . fromIntegral <$> [C.exp|int { (int) $(JitType *t)->kind() }|]

type_string :: Ptr CType -> IO String
type_string t = [C.exp|char* { (char*)$(JitType *t)->str().c_str() }|] >>= peekCString

-- Only use on tensors!
type_sizes :: Ptr CType -> IO (Vector Int64)
type_sizes t = do
  s <- [C.exp|int { ((CompleteTensorType*)$(JitType *t))->sizes().size() }|]
  d <- [C.exp|int64_t *{ const_cast<int64_t*>(((CompleteTensorType*)$(JitType *t))->sizes().data()) }|] >>= newForeignPtr_
  pure $ V.unsafeFromForeignPtr0 d (fromIntegral s)

type_device_type :: Ptr CType -> IO Backend
type_device_type t = toEnum . fromIntegral <$> [C.exp|int { (int)((CompleteTensorType*)$(JitType *t))->device().type() }|]

type_requires_grad :: Ptr CType -> IO Bool
type_requires_grad t = do
  (CBool b) <- [C.exp|bool { ((CompleteTensorType*)($(JitType *t)))->requires_grad() }|]
  pure (b /= 0)

type_contained :: Ptr CType -> IO (Vector (Ptr CType))
type_contained t = do
  s <- [C.exp|int { $(JitType *t)->containedTypes().size() }|]
  r <- V.generateM (fromIntegral s)
    (\i -> let i' = fromIntegral i
          in [C.exp|JitType *{ $(JitType *t)->containedTypes().at($(int i')).get() }|])
  pure r
-- toEnum . fromIntegral <$> [C.exp|int { (int) $(JitValue *v)->type()->contained() }|]

value_type_contained :: Ptr CJitValue -> IO (Vector (Ptr CType))
value_type_contained v = do
  s <- [C.exp|int { $(JitValue *v)->type()->containedTypes().size() }|]
  r <- V.generateM (fromIntegral s)
    (\i -> let i' = fromIntegral i
          in [C.exp|JitType *{ $(JitValue *v)->type()->containedTypes().at($(int i')).get() }|])
  pure r
-- toEnum . fromIntegral <$> [C.exp|int { (int) $(JitValue *v)->type()->contained() }|]

-- TODO Remove this
debug_print_unique_name :: Ptr CJitValue -> IO ()
debug_print_unique_name v = do
  [C.exp|void { cout << "JitValue name:" << $(JitValue *v)->debugName() << endl << $(JitValue *v)->type()->str() << endl; }|]

-- Test code:
  -- std::shared_ptr<TracingState> state;
  -- variable_list trace_vars_in;
  -- std::tie(state, trace_vars_in) = torch::jit::tracer::enter(vars_in, 1);
  -- auto trace_vars_out = test(trace_vars_in);
  -- torch::jit::tracer::exit(trace_vars_out);
  -- return state->graph;

-- Only alive while the function is alive!
function_next_edges f = [C.exp|void* { (void*)&($(Node *f)->next_edges()) }|] >>= unVectorEdge

edge_function :: Ptr CEdge -> IO (Ptr CNode)
edge_function e = [C.exp|Node* { $(Edge *e)->function.get() }|]

edge_input_nr :: Ptr CEdge -> IO CInt
edge_input_nr e = [C.exp|int { $(Edge *e)->input_nr }|]

-- TODO clear gradient function
-- clear_gradient_function v = [cblock|void { ((Variable*) $fptr-ptr:(Tensor *v))->get()->_grad_fn = nullptr; }|]

is_leaf :: Coercible a (ForeignPtr CVariable) => a -> IO CBool
is_leaf v = [C.exp|bool { $fptr-ptr:(Variable *v)->is_leaf() }|]

-- This is not a memory leak, grad is cleaned up when the variable dies
grad :: Coercible a (ForeignPtr CTensor) => a -> IO (ForeignPtr CTensor)
grad v = newForeignPtr_ =<< [C.exp|Tensor *{ &$fptr-ptr:(Tensor *v)->grad() }|]

detach :: Coercible a (ForeignPtr CTensor) => a -> IO (ForeignPtr CVariable)
detach v = newForeignPtr deleteVariable
           =<< [C.exp|Variable *{ new Variable(((Variable*) $fptr-ptr:(Tensor *v))->detach()) }|]

output_nr :: Coercible a (ForeignPtr CTensor) => a -> IO CInt
output_nr v = [C.exp|int { ((Variable*) $fptr-ptr:(Tensor *v))->output_nr() }|]

requires_grad :: Coercible a (ForeignPtr CTensor) => a -> IO CBool
requires_grad v = [C.exp|bool { ((Variable*) $fptr-ptr:(Tensor *v))->requires_grad() }|]

set_requires_grad :: (Coercible a (ForeignPtr CVariable), Coercible a (ForeignPtr CTensor)) => a -> CBool -> IO ()
set_requires_grad v b =
  ifM ((/= 0) <$> is_leaf v)
      [C.exp|void { ($fptr-ptr:(Tensor *v))->set_requires_grad($(bool b)) }|]
      (error "Can only change requires_grad on leaf variables")

name :: Coercible a (ForeignPtr CTensor) => a -> IO (Ptr CChar)
name v = [C.exp|const char *{ ((Variable*) $fptr-ptr:(Tensor *v))->name().c_str() }|]

-- TODO Hooks!
-- -- I think i == 0 always
-- add_backwards_hook v f i =
--   [cblock|void *{
--          auto hook = std::shared_ptr<HsFunctionPreHook>(
--             new HsFunctionPreHook($fun:(Variable* (*f)(Variable*)), $(int i)));
--          ((Variable*) $fptr-ptr:(Tensor *v))->hooks().emplace_back(hook);
--          return (void*)hook.get();
--          }|]

-- TODO Hooks!
-- remove_backwards_hook v h =
--   [cblock|bool {
--          auto hooks = ((Variable*) $fptr-ptr:(Tensor *v))->hooks();
--          for(int i = 0; i < hooks.size(); i++) {
--             if(hooks.at(i).get() == $(void* h)) {
--               hooks.erase(hooks.begin() + i);
--               return true;
--             }
--          }
--          return false; }|]

-- TODO Hooks!
-- clear_backwards_hooks v = [C.exp|void { ((Variable*) $fptr-ptr:(Tensor *v))->hooks().clear() }|]

base :: Coercible a (ForeignPtr CTensor) => a -> IO (Ptr CVariable)
base v = [C.exp|Variable *{ const_cast<Variable*>(&((Variable*) $fptr-ptr:(Tensor *v))->base()) }|]

shape :: Coercible a (ForeignPtr CTensor) => a -> IO (Vector CLong)
shape v = do
  -- TODO Storage issues, when is this available?
  d <- newForeignPtr_ =<< [C.exp|const long int *{ ((Variable*) $fptr-ptr:(Tensor *v))->sizes().data() }|]
  s <- [C.exp|int { ((Variable*) $fptr-ptr:(Tensor *v))->sizes().size() }|]
  pure $ V.unsafeFromForeignPtr0 d (fromIntegral s)

stride :: Coercible a (ForeignPtr CTensor) => a -> IO (Vector CLong)
stride v = do
  -- TODO Storage issues, when is this available?
  d <- newForeignPtr_ =<< [C.exp|const long int *{ ((Variable*) $fptr-ptr:(Tensor *v))->strides().data() }|]
  s <- [C.exp|int { ((Variable*) $fptr-ptr:(Tensor *v))->strides().size() }|]
  pure $ V.unsafeFromForeignPtr0 d (fromIntegral s)

variableData :: (Coercible b (Ptr ()), Coercible a (ForeignPtr CTensor)) => a -> IO b
variableData v =
  -- TODO what does retain do here?
  coerce <$> [C.exp|void *{ ((Variable*) $fptr-ptr:(Tensor *v))->unsafeGetTensorImpl() }|]

mkVar :: Coercible a (ForeignPtr CTensor) => a -> CBool -> IO (ForeignPtr CVariable)
mkVar tensor needsGrad =
  newForeignPtr deleteVariable
  =<< [C.exp|Variable *{ new Variable(std::move(make_variable(*((Variable*)$fptr-ptr:(Tensor *tensor)), $(bool needsGrad)))) }|]

-- * IO

mkJitModule :: IO (ForeignPtr CJitScriptModule)
mkJitModule = [C.exp|JitScriptModule *{ new JitScriptModule("__main__") }|] >>= newForeignPtr deleteJitModule

jitModuleAddTensor mod key tensor =
  withTextCString key (\k -> [C.exp|void { $fptr-ptr:(JitScriptModule *mod)->register_parameter($(char *k), *$fptr-ptr:(Tensor *tensor), false) }|])

jitModuleAddModule mod key mod' =
  withTextCString key (\k -> [C.exp|void { $fptr-ptr:(JitScriptModule *mod)->register_module($(char *k), *$fptr-ptr:(JitScriptModule *mod')) }|])

jitModueWriteToFile mod filename =
  withTextCString filename (\fname -> [C.exp|void { torch::jit::ExportModule(*$fptr-ptr:(JitScriptModule *mod), $(char *fname)) }|])

jitModuleReadFromFile filename = do
  m <- withTextCString filename (\fname -> [C.block|JitScriptModule *{
                                     try
                                     {
                                       return new JitScriptModule(torch::jit::load($(char *fname)));
                                     } catch(...)
                                     {
                                       return nullptr;
                                     }
                                     }|])
  if m == nullPtr then
    pure Nothing else
    Just <$> newForeignPtr deleteJitModule m

jitModuleHasTensor mod key =
  withTextCString key (\k -> [C.exp|bool {$fptr-ptr:(JitScriptModule *mod)->find_parameter($(char *k)) ? true : false }|])

jitModuleReadTensor mod key =
  withTextCString key (\k -> [C.exp|Tensor *{ new Tensor($fptr-ptr:(JitScriptModule *mod)->find_parameter($(char *k))->value().toTensor()) }|])
  >>= newForeignPtr deleteTensor

jitModuleReadModule mod key =
  withTextCString key (\k -> [C.exp|JitScriptModule *{ new JitScriptModule($fptr-ptr:(JitScriptModule *mod)->get_module($(char *k))) }|])
  >>= newForeignPtr deleteJitModule

jitModuleNumberOfSlots mod = [C.exp|int { $fptr-ptr:(JitScriptModule *mod)->num_slots() }|]

jitModueToBackend mod backend = [C.exp|void { $fptr-ptr:(JitScriptModule *mod)->to(at::Device(static_cast<c10::DeviceType>($(int backend)), -1)) }|]

jitModuleSlotName mod slotNr =
  [C.exp|char *{ (char*)$fptr-ptr:(JitScriptModule *mod)->get_slot($(int slotNr)).name().c_str() }|] >>= textPeekCString

jitModuleSlotType :: ForeignPtr CJitScriptModule -> CInt -> IO ModuleEntityType
jitModuleSlotType mod slotNr =
  toEnum . fromIntegral <$> [C.exp|int{ (int)$fptr-ptr:(JitScriptModule *mod)->get_slot($(int slotNr)).entity_type() }|]
