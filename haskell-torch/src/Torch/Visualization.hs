{-# LANGUAGE AllowAmbiguousTypes, ConstraintKinds, DataKinds, ExtendedDefaultRules, FlexibleContexts, FlexibleInstances, GADTs        #-}
{-# LANGUAGE KindSignatures, MultiParamTypeClasses, OverloadedStrings, PolyKinds, QuasiQuotes, RankNTypes, ScopedTypeVariables        #-}
{-# LANGUAGE TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Torch.Visualization where
import           Control.Monad
import qualified Data.List            as L
import qualified Data.Map.Strict      as M
import           Data.Text            (Text)
import qualified Data.Text            as T
import qualified Data.Text.IO         as T
import           Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import           Foreign.ForeignPtr
import           Foreign.Ptr
import           GHC.Int
import qualified Shelly               as S
import qualified Torch.C.Tensor       as C
import qualified Torch.C.Types        as C
import qualified Torch.C.Variable     as C
import           Torch.Tensor
import           Torch.Types
import           Foreign.ForeignPtr.Unsafe


data TraceValue = TraceValueTensor { tvTensorName       :: T.Text
                                   , tvTensorSizes      :: Vector Int64
                                   , tvtensorScalarType :: C.ScalarType }
                deriving (Show, Eq, Ord)

data TraceNode = TraceNode { tnKind    :: T.Text
                           , tnInputs  :: [TraceValue]
                           , tnOutputs :: [TraceValue] }
                deriving (Show, Eq, Ord)

data TraceGraph = TraceGraph { traceGraphInputs  :: [TraceValue]
                             , traceGraphNodes   :: [TraceNode]
                             , traceGraphOutputs :: [TraceValue] }
                deriving (Show, Eq, Ord)

type TracePtr = ForeignPtr C.CTracingState

withTracing :: (TensorConstraints ty ki sz) => [AnyTensor] -> IO (Tensor ty ki sz) -> IO (Tensor ty ki sz, TracePtr)
withTracing ts f = do
  -- TODO This has a constant, 1
  cfn <- C.mkTraceable (\_ -> castPtr . unsafeForeignPtrToPtr . tensorPtr <$> f) -- TODO Tensor to Variable cast
  (state, vars) <- withForeignPtrList (map (\(AnyTensor (Tensor p _)) -> p) ts)
                                      (\ps -> C.trace (V.map castPtr ps) 1 cfn)
  -- FIXME What about this?
  -- state' <- newForeignPtr C.deleteTracingState state
  state' <- newForeignPtr_ state
  ptr <- newForeignPtr C.deleteTensor  $ castPtr $ vars V.! 0
  pure (Tensor ptr Nothing, state') -- TODO Nothing might instead be vars

-- | This prints the trace in the pytorch/onnx graph format.
printTrace :: TracePtr -> IO ()
printTrace s = withForeignPtr s C.print_tracing_state_graph

-- | This prints the trace in the onnx graph format.
printTraceONNX :: TracePtr -> [AnyTensor] -> Bool -> Int -> IO ()
printTraceONNX s ts googleFormat opset =
  withForeignPtrList (map (\(AnyTensor (Tensor p _)) -> p) ts)
  (\ps -> withForeignPtr s (\s -> C.print_tracing_state_graph_onnx s (V.map castPtr ps) (boolc googleFormat) (fromIntegral opset)))

tnIsConstant :: TraceNode -> Bool
tnIsConstant tn | tnKind tn == "prim::Constant" = True
                | otherwise = False

showsz :: (V.Storable a, Show a) => Vector a -> Text
showsz x =
  case V.length x of
    0 -> "Scalar"
    _ -> T.pack $ show x

data TraceType = TraceTypeSimple Bool C.TypeKind
               | TraceTypeList   Bool [TraceType]
               | TraceTypeTuple  Bool [TraceType]
               | TraceTypeTensor Bool (Maybe (Vector Int64)) (Maybe C.ScalarType) (Maybe C.Backend)
               deriving (Show, Eq)

convertCType :: Ptr C.CType -> IO TraceType
convertCType t = do
  k <- C.type_kind t
  grad <- C.type_requires_grad t
  case k of
    C.TypeKindTensor -> do
      sizes <- C.type_sizes t
      scalar <- C.type_scalar_type t
      device <- C.type_device_type t
      pure $ TraceTypeTensor grad (Just sizes) (Just scalar) (Just device)
      -- TODO cleanup
      -- C.TypeKindCompleteTensor -> do
      -- pure $ TraceTypeTensor grad Nothing Nothing Nothing
    C.TypeKindList -> do
      cs <- C.type_contained t
      cs' <- mapM convertCType $ V.toList cs
      pure $ TraceTypeList grad cs'
    C.TypeKindTuple -> do
      cs <- C.type_contained t
      cs' <- mapM convertCType $ V.toList cs
      pure $ TraceTypeTuple grad cs'
    C.TypeKindInt -> do
      pure $ TraceTypeSimple grad k
    C.TypeKindFloat -> do
      pure $ TraceTypeSimple grad k
    C.TypeKindString -> do
      pure $ TraceTypeSimple grad k
    C.TypeKindNone -> do
      pure $ TraceTypeSimple grad k
    C.TypeKindGenerator -> do
      pure $ TraceTypeSimple grad k
    C.TypeKindBool -> do
      pure $ TraceTypeSimple grad k
    _ -> do
      print =<< C.type_string t
      error "Unsupported variable type, see line above"

parseTrace :: ForeignPtr C.CTracingState -> IO TraceGraph
parseTrace trace =
  parseGraph =<< withForeignPtr trace C.tracing_state_graph
  where parseGraph (inputs, nodes, outputs, _, _) = do
          is <- mapM parseValue $ V.toList inputs
          ns <- mapM parseNode $ V.toList nodes
          os <- mapM parseValue $ V.toList outputs
          pure $ TraceGraph is ns os
        parseValue x = do
          b <- cbool <$> C.check_value_tensor x
          if b then do
            sz <- C.value_sizes x
            ty <- C.value_scalar_type x
            na <- C.value_name x
            pure $ TraceValueTensor (T.pack na) sz ty
            else do
            -- TODO Figure out more stuff here
            na <- C.value_name x
            pure $ TraceValueTensor (T.pack na) V.empty C.ScalarTypeUndefined
            -- DEBUGGING TODO
            -- error "Other values are not yet supported"
        parseNode x = do
          ki <- C.node_kind x
          is <- mapM parseValue . V.toList =<< C.node_inputs x
          os <- mapM parseValue . V.toList =<< C.node_outputs x
          pure $ TraceNode (T.pack ki) is os

countTraceParameters :: TraceGraph -> Int64
countTraceParameters trace = countGraph trace
  where countGraph tr = Prelude.sum $ map countNode $ traceGraphNodes tr
        countNode n = if tnIsConstant n then
                        Prelude.sum (map countValue (tnOutputs n))
                        else 0
        countValue v = case v of
          TraceValueTensor n s t -> V.product s

summarizeTrace :: TraceGraph -> IO ()
summarizeTrace trace = do
  T.putStrLn (T.justifyRight 25 ' ' "Input shape"
             <> T.justifyRight 35 ' ' (T.intercalate " " (map showsz (map tvTensorSizes (traceGraphInputs trace)))))
  T.putStrLn (T.replicate 79 "=")
  T.putStrLn (T.justifyRight 25 ' ' "Function"
             <>T.justifyRight 35 ' '"Output shape"
             <>T.justifyRight 19 ' ' "# Params")
  T.putStrLn (T.replicate 79 "-")
  mapM (\e -> do
           if tnIsConstant e then
             pure () else
             T.putStrLn
             (T.justifyRight 25 ' ' (tnKind e)
              <>T.justifyRight 35 ' '
               (T.intercalate " "
                 (map showsz (map tvTensorSizes (tnOutputs e))))
               <>T.justifyRight 19 ' '
               (T.pack
                (show
                 (Prelude.sum (map (\i ->
                                    case M.lookup (tvTensorName i) constants of
                                      Just _  -> V.product $ tvTensorSizes i
                                      Nothing -> 0)
                             (tnInputs e)))))))
    (traceGraphNodes trace)
  T.putStrLn (T.replicate 79 "=")
  T.putStrLn $ "Total number of parameters: " <> T.pack (show (countTraceParameters trace))
  T.putStrLn (T.replicate 79 "-")
  where constants = M.fromList
                    $ filter (\(_,x) -> not (elem x (traceGraphInputs trace)))
                    $ concatMap (\x ->
                           if tnIsConstant x then
                             map (\x -> (case x of
                                           TraceValueTensor{} -> tvTensorName x
                                       , x)) (tnOutputs x)
                             else [])
                    (traceGraphNodes trace)

showTraceGraph :: TracePtr -> Bool -> IO ()
showTraceGraph tptr noConstants = do
  trace <- parseTrace tptr
  let filename = "/tmp/a.dot"
  let outfilename = "/tmp/a.png"
  T.writeFile (T.unpack filename) (traceToDot trace noConstants)
  c <- S.shelly $ S.verbosely $ do
    S.run_ "dot" [filename,"-Tpng","-o",outfilename]
    S.lastExitCode
  c <- S.shelly $ S.verbosely $ do
    S.run_ "feh" [outfilename]
    S.lastExitCode
  pure ()

data TraceGraphState = TGS { tgsStr     :: Text
                           , tgsMap     :: M.Map Text Int
                           , tgsNext    :: Int
                           , tgsOutputs :: M.Map TraceValue [Int] }
  deriving (Show)

traceToDot :: TraceGraph -> Bool -> Text
traceToDot trace noConstants =
  let state = loopGraph (TGS "" M.empty 0 M.empty) trace
  in "digraph trace {\n"
     <> "node [style=rounded]" <> "\n"
     <> tgsStr state
     <> T.unlines (map (\(v,is) -> mkOutputNode v is) (M.toList (tgsOutputs state)))
     <> "}"
  where
    withNextNode s f = f (s { tgsNext = tgsNext s + 1 }) (tgsNext s)
    mkInputNode v s i =
      s { tgsStr = tgsStr s
                 <> T.pack (show i) <> " [shape=egg,color=green,label=\"In:" <> showsz (tvTensorSizes v) <> "\"]" <> "\n"
        , tgsMap = M.insert (tvTensorName v) i (tgsMap s) }
    mkNode n s i | any (\x -> T.isPrefixOf (T.pack x) (tnKind n) || T.isSuffixOf (T.pack x) (tnKind n)) censoredNodes = s
                 | noConstants && tnIsConstant n = s
                 | otherwise =
      s { tgsStr = tgsStr s
                 <> T.pack (show i) <> " [shape=box,"
                                    <> (if tnIsConstant n then
                                          "color=orange," else
                                          "")
                                    <> "label=\""
                                    <> (if tnIsConstant n then
                                         "Const:" <> (T.intercalate " "
                                                     (map showsz (map tvTensorSizes (tnOutputs n)))) else
                                       tnKind n)
                                  <>"\"]" <> "\n"
                 <> T.unlines
                       (map (\x ->
                               case M.lookup (tvTensorName x) (tgsMap s) of
                                 Nothing ->
                                   if noConstants then
                                     "" else
                                     error $ "Can't find an input? This is a bug! " <> show x <> show n
                                 Just x -> T.pack (show x) <> " -> "  <> T.pack (show i))
                            (tnInputs n))
        , tgsMap = L.foldl' (\m v -> M.insert (tvTensorName v) i m) (tgsMap s) (tnOutputs n)
        , tgsOutputs =
            (L.foldl' (\m v ->
                       case M.lookup v m of
                         Just x  -> M.adjust (i:) v m
                         Nothing -> m)
              -- delete any seen outputs; they're outputs of a subgraph
              (L.foldl' (flip M.delete) (tgsOutputs s) (tnInputs n))
              (tnOutputs n))
        }
    mkOutputNode v is =
         "O" <> (tvTensorName v) <> " [shape=hexagon,color=blue,label=\"Out:" <> showsz (tvTensorSizes v) <> "\"]" <> "\n"
      <> T.unlines (map (\i -> T.pack (show i) <> " -> O" <> (tvTensorName v)) is)
    loopGraph state g =
      -- Invent nodes for the inputs if needed
      let state' = L.foldl' (\s v ->
                            case M.lookup (tvTensorName v) (tgsMap s) of
                              Nothing -> withNextNode s (mkInputNode v)
                              Just _  -> s)
                          state
                          (traceGraphInputs trace)
      -- Invent nodes for the outputs if needed
      in let state'' = L.foldl' (\s v -> s { tgsOutputs = M.insert v [] (tgsOutputs s) } )
                                state'
                                (traceGraphOutputs trace)
         in L.foldl' loopNode state'' (traceGraphNodes g)
    loopNode s n = withNextNode s (mkNode n)
    censoredNodes = ["aten::_convolution_nogroup"
                    ,"_forward"
                    ,"aten::thnn_"]
