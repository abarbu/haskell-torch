{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, DefaultSignatures, DeriveAnyClass, DeriveGeneric, FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances, FunctionalDependencies, GADTs, OverloadedLabels, OverloadedLists, OverloadedStrings                   #-}
{-# LANGUAGE PartialTypeSignatures, PolyKinds, QuasiQuotes, RankNTypes, RecordWildCards, ScopedTypeVariables, StandaloneDeriving      #-}
{-# LANGUAGE TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -pgmP cc -optP -E -optP -undef -optP -std=c89 #-}

module Torch.Initialization where
import           Control.Monad
import           Data.Default
import           Data.IORef
import           Data.Maybe
import           Data.Singletons
import           Data.Singletons.Prelude   as SP
import           Data.Singletons.TH
import           Data.String.InterpolateIO
import           Data.String.ShowIO
import qualified Data.Vector               as VB
import qualified Data.Vector.Storable      as V
import           Foreign.Storable
import           Generics.Eot
import           GHC.TypeLits              as TL
import           Prelude                   as P
import qualified Torch.C.Tensor            as C
import           Torch.Inplace
import           Torch.Misc
import           Torch.Tensor
import           Torch.Types

-- * Initialization schemes

kaimingUniform :: forall sz ty ki d. (TensorConstraints ty ki sz, SingI (IsFloatTy ty))
               => Maybe Double -> GainNonlinearity -> FanMode -> IO (Tensor ty ki sz)
kaimingUniform a gianMode fanMode = zeros >>= \t -> kaimingUniform_ t a gianMode fanMode

kaimingUniformBias :: forall sz ty ki d. (TensorConstraints ty ki sz, SingI (IsFloatTy ty))
                   => IO (Tensor ty ki sz)
kaimingUniformBias = zeros >>= \t -> kaimingUniformBias_ t

kaimingNormal :: forall sz ty ki d. (TensorConstraints ty ki sz, SingI (IsFloatTy ty))
              => Maybe Double -> GainNonlinearity -> FanMode -> IO (Tensor ty ki sz)
kaimingNormal a gainMode fanMode = zeros >>= \t -> kaimingNormal_ t a gainMode fanMode

xavierUniform :: forall sz ty ki d. (TensorConstraints ty ki sz, SingI (IsFloatTy ty))
              => Maybe Double -> IO (Tensor ty ki sz)
xavierUniform gain = zeros >>= \t -> xavierUniform_ t gain

xavierNormal :: forall sz ty ki d. (TensorConstraints ty ki sz, SingI (IsFloatTy ty))
             => Maybe Double -> IO (Tensor ty ki sz)
xavierNormal gain = zeros >>= \t -> xavierNormal_ t gain

randn2 :: (TensorConstraints ty ki sz, TensorConstraints ty ki sz', IsFloatTy ty ~ 'True)
       => IO (Tensor ty ki sz
            ,Maybe (Tensor ty ki sz'))
randn2 = do
  w <- randn
  b <- randn
  pure (w, Just b)

initUniform :: forall sz ty ki d. (TensorConstraints ty ki sz, SingI (IsFloatTy ty))
             => Double -> IO (Tensor ty ki sz)
initUniform stdv =
  case (sing :: Sing (IsFloatTy ty)) of
    STrue -> uniform (- stdv) stdv
    _     -> toType =<< uniform @TDouble (- stdv) stdv

-- * Extract tensors

-- | Extract all of the tensors from inside a model or a layer. This is like to
-- ToParameters but gives you all of the tensors, even if the ones which are not
-- parameters.
--
-- For your own structures, just derive from Generic and ToTensors.

class ToTensors a where
  toTensors :: a -> IO [AnyTensor]
  default toTensors :: (HasEot a, EotToTensors (Eot a)) => a -> IO [AnyTensor]
  toTensors = eotToTensors . toEot

class EotToTensors eot where
  eotToTensors :: eot -> IO [AnyTensor]

instance (EotToTensors fields) => EotToTensors (Either fields Void) where
  eotToTensors (Left fields) = eotToTensors fields

instance (ToTensors x, EotToTensors xs) =>
  EotToTensors (x, xs) where
  eotToTensors (x, xs) = do
    ps <- toTensors x
    ps' <- eotToTensors xs
    pure $ ps ++ ps'

instance EotToTensors () where
  eotToTensors _ = pure []

instance (TensorConstraints ty ki sz) => ToTensors (Tensor ty ki sz) where
  toTensors t = pure [AnyTensor t]

instance (TensorConstraints ty ki '[outF, inF], SingI outF)
       => ToTensors (LinearParam ty ki inF outF) where
  toTensors (LinearParam t mt') =
    pure $ [AnyTensor t] ++ maybe [] ((\x -> [x]) . AnyTensor) mt'

instance (TensorConstraints ty ki otherSz, SingI outChans)
       => ToTensors (ConvParam ty ki outChans otherSz) where
  toTensors (ConvParam t mt') =
    pure $ [AnyTensor t] ++ maybe [] ((\x -> [x]) . AnyTensor) mt'

instance (TensorConstraints ty ki sz)
       => ToTensors (AffineParam ty ki sz) where
  toTensors (AffineParam t mt') =
    pure $ [AnyTensor t] ++ maybe [] ((\x -> [x]) . AnyTensor) mt'

instance (TensorConstraints ty ki '[], SingI nrEmbeddings, SingI embeddingDim)
       => ToTensors (EmbeddingParam ty ki nrEmbeddings embeddingDim) where
  toTensors (EmbeddingParam t) = pure [AnyTensor t]

instance (TensorConstraints ty ki '[], KnownNat nrLayers
         ,KnownNat hiddenF, KnownNat inF, SingI isBidirectional, SingI gateSize
         ,KnownNat (NrOfRNNDirections isBidirectional), SingI (gateSize TL.* hiddenF)
         ,SingI (NrOfRNNDirections isBidirectional TL.* hiddenF))
       => ToTensors (GenericRNNParam ty ki gateSize inF hiddenF nrLayers isBidirectional batchFirst) where
  toTensors GenericRNNParam{..} =
    pure $ ( map AnyTensor $ toListDirection grnnParamWih0)
           ++ concat (VB.toList (VB.map (\x -> map AnyTensor (toListDirection x)) grnnParamWihN1))
           ++ maybe [] (concat . VB.toList . VB.map (\x -> map AnyTensor (toListDirection x))) grnnParamBihN
           ++ concat (VB.toList (VB.map (\x -> map AnyTensor (toListDirection x)) grnnParamWhhN))
           ++ maybe [] (concat . VB.toList . VB.map (\x -> map AnyTensor (toListDirection x))) grnnParamBhhN

instance (ToTensors (GenericRNNParam ty ki 1 inF hiddenF nrLayers isBidirectional batchFirst))
       => ToTensors (RNNParams ty ki inF hiddenF nrLayers isBidirectional batchFirst) where
  toTensors (RNNParams g) = toTensors g

instance (ToTensors (GenericRNNParam ty ki 3 inF hiddenF nrLayers isBidirectional batchFirst))
       => ToTensors (GRUParams ty ki inF hiddenF nrLayers isBidirectional batchFirst) where
  toTensors (GRUParams g) = toTensors g

instance (ToTensors (GenericRNNParam ty ki 4 inF hiddenF nrLayers isBidirectional batchFirst))
       => ToTensors (LSTMParams ty ki inF hiddenF nrLayers isBidirectional batchFirst) where
  toTensors (LSTMParams g) = toTensors g

instance (TensorConstraints ty ki '[hiddenF, inF], SingI hiddenF)
       => ToTensors (RNNCellParam ty ki inF hiddenF) where
  toTensors (RNNCellParam wih bih whh bhh) =
    pure ([AnyTensor wih] ++ maybe [] ((\x -> [x]) . AnyTensor) bih
           ++ [AnyTensor whh] ++ maybe [] ((\x -> [x]) . AnyTensor) bhh)

instance (TensorConstraints ty ki sz) => ToTensors (BatchNormState ty ki sz) where
  toTensors (BatchNormState s) = do
    BatchNormData m v <- readIORef s
    pure $ map AnyTensor $ catMaybes [m,v]

instance (ToTensors a1, ToTensors a2) => ToTensors (a1,a2)
instance (ToTensors a1, ToTensors a2, ToTensors a3) => ToTensors (a1,a2,a3)
instance (ToTensors a1, ToTensors a2, ToTensors a3, ToTensors a4) => ToTensors (a1,a2,a3,a4)
instance (ToTensors a1, ToTensors a2, ToTensors a3, ToTensors a4, ToTensors a5) => ToTensors (a1,a2,a3,a4,a5)
instance (ToTensors a1, ToTensors a2, ToTensors a3, ToTensors a4, ToTensors a5, ToTensors a6) => ToTensors (a1,a2,a3,a4,a5,a6)

instance (SingI ty, SingI ki, SingI nrLayers, KnownNat (NrOfRNNDirections isBidirectional)
         ,SingI (NrOfRNNDirections isBidirectional TL.* nrLayers), SingI batch, SingI hiddenF
         ,Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
         ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty))
       => ToTensors (RNNState ty ki batch isBidirectional nrLayers hiddenF) where
  toTensors (RNNState a) = pure [AnyTensor a]

instance (SingI ty, SingI ki, SingI nrLayers, KnownNat (NrOfRNNDirections isBidirectional)
         ,SingI (NrOfRNNDirections isBidirectional TL.* nrLayers), SingI batch, SingI hiddenF
         ,Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
         ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty))
       => ToTensors (RNNStateBatchFirst ty ki batch isBidirectional nrLayers hiddenF) where
  toTensors (RNNStateBatchFirst a) = pure [AnyTensor a]

instance (SingI ty, SingI ki, SingI nrLayers, KnownNat (NrOfRNNDirections isBidirectional)
         ,SingI (NrOfRNNDirections isBidirectional TL.* nrLayers), SingI batch, SingI hiddenF
         ,Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
         ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty))
       => ToTensors (LSTMState ty ki batch isBidirectional nrLayers hiddenF) where
  toTensors (LSTMState a b) = pure [AnyTensor a, AnyTensor b]

instance (SingI ty, SingI ki, SingI nrLayers, KnownNat (NrOfRNNDirections isBidirectional)
         ,SingI (NrOfRNNDirections isBidirectional TL.* nrLayers), SingI batch, SingI hiddenF
         ,Num (TensorTyToHs ty), Num (TensorTyToHsC ty)
         ,Storable (TensorTyToHs ty), Storable (TensorTyToHsC ty))
       => ToTensors (LSTMStateBatchFirst ty ki batch isBidirectional nrLayers hiddenF) where
  toTensors (LSTMStateBatchFirst a b) = pure [AnyTensor a, AnyTensor b]

-- * Extract parameters

-- | Extract parameters from a layer or some other structure. You may also
-- notice we have ToTensors. Not all tensors are parameters, for example, the
-- state of an RNN is generally not considered an optimizable parameter, instead
-- it's the state of a system. Somehow, if you have a larger structure, you will
-- want to pick out the pieces that are relevant to an optimizer, that's what
-- this does. We might have been able to get away with just ToTensors if
-- StoredModule had been kinder to us. It doesn't keep the required_gradient
-- flag correctly all the time, so we don't know if a tensor in a stored model
-- needed a gradient or not. That means when we load a module, we need to get
-- all of its parameters, and enable gradients for them --- but not do so for
-- state-like things.
--
-- For your own structures, just derive from Generic and ToParameters.
class ToParameters a where
  -- | Convert your layer or structure into its parameters
  toParameters :: a -> IO [AnyTensor]
  default toParameters :: (HasEot a, EotToParameters (Eot a)) => a -> IO [AnyTensor]
  toParameters = eotToParameters . toEot

class EotToParameters eot where
  eotToParameters :: eot -> IO [AnyTensor]

instance (EotToParameters fields) => EotToParameters (Either fields Void) where
  eotToParameters (Left fields) = eotToParameters fields

instance (ToParameters x, EotToParameters xs) =>
  EotToParameters (x, xs) where
  eotToParameters (x, xs) = do
    ps <- toParameters x
    ps' <- eotToParameters xs
    pure $ ps ++ ps'

instance EotToParameters () where
  eotToParameters _ = pure []

instance (ToTensors (Tensor ty ki sz))
       => ToParameters (Tensor ty ki sz) where
  toParameters x = filterM requiresGradAny =<< toTensors x

instance (ToTensors (LinearParam ty ki inF outF))
       => ToParameters (LinearParam ty ki inF outF) where
  toParameters x = filterM requiresGradAny =<< toTensors x

instance (ToTensors (ConvParam ty ki outChans otherSz))
       => ToParameters (ConvParam ty ki outChans otherSz) where
  toParameters x = filterM requiresGradAny =<< toTensors x

instance (ToTensors (AffineParam ty ki sz))
       => ToParameters (AffineParam ty ki sz) where
  toParameters x = filterM requiresGradAny =<< toTensors x

instance (ToTensors (EmbeddingParam ty ki nrEmbeddings embeddingDim))
       => ToParameters (EmbeddingParam ty ki nrEmbeddings embeddingDim) where
  toParameters x = filterM requiresGradAny =<< toTensors x

instance (ScalarConstraints ty ki, KnownNat nrLayers
         ,KnownNat hiddenF, KnownNat inF, SingI isBidirectional, SingI gateSize
         ,KnownNat (NrOfRNNDirections isBidirectional), SingI (gateSize TL.* hiddenF)
         ,SingI (NrOfRNNDirections isBidirectional TL.* hiddenF))
       => ToParameters (GenericRNNParam ty ki gateSize inF hiddenF nrLayers isBidirectional batchFirst) where
  toParameters x = filterM requiresGradAny =<< toTensors x

instance (ToParameters (GenericRNNParam ty ki 1 inF hiddenF nrLayers isBidirectional batchFirst))
       => ToParameters (RNNParams ty ki inF hiddenF nrLayers isBidirectional batchFirst) where
  toParameters (RNNParams g) = toParameters g

instance (ToParameters (GenericRNNParam ty ki 3 inF hiddenF nrLayers isBidirectional batchFirst))
       => ToParameters (GRUParams ty ki inF hiddenF nrLayers isBidirectional batchFirst) where
  toParameters (GRUParams g) = toParameters g

instance (ToParameters (GenericRNNParam ty ki 4 inF hiddenF nrLayers isBidirectional batchFirst))
       => ToParameters (LSTMParams ty ki inF hiddenF nrLayers isBidirectional batchFirst) where
  toParameters (LSTMParams g) = toParameters g

instance (ToTensors (RNNCellParam wih bih whh bhh))
       => ToParameters (RNNCellParam wih bih whh bhh) where
  toParameters x = filterM requiresGradAny =<< toTensors x

instance (ToParameters a1, ToParameters a2) => ToParameters (a1,a2)
instance (ToParameters a1, ToParameters a2, ToParameters a3) => ToParameters (a1,a2,a3)
instance (ToParameters a1, ToParameters a2, ToParameters a3, ToParameters a4) => ToParameters (a1,a2,a3,a4)
instance (ToParameters a1, ToParameters a2, ToParameters a3, ToParameters a4, ToParameters a5) => ToParameters (a1,a2,a3,a4,a5)
instance (ToParameters a1, ToParameters a2, ToParameters a3, ToParameters a4, ToParameters a5, ToParameters a6) => ToParameters (a1,a2,a3,a4,a5,a6)

-- These instances are special. They don't follow the default rule, which is
-- "all differentiable tensors are parameters".

instance ToParameters (BatchNormState ty ki sz) where
  toParameters _ = pure []

instance ToParameters (RNNState ty ki batch isBidirectional nrLayers hiddenF) where
  toParameters _ = pure []

instance ToParameters (RNNStateBatchFirst ty ki batch isBidirectional nrLayers hiddenF) where
  toParameters _ = pure []

instance ToParameters (LSTMState ty ki batch isBidirectional nrLayers hiddenF) where
  toParameters _ = pure []

instance ToParameters (LSTMStateBatchFirst ty ki batch isBidirectional nrLayers hiddenF) where
  toParameters _ = pure []

-- * Initializing layers

-- | This is handy for initializing your own layers and structures with the
-- default settings, with or without a gradient. Derive from this class and
-- Generic in order to take advantage of this mechanism for your data types.  If
-- you want custom initializers for a layer, you're best off wrapping that layer
-- in a newtype and then creating an instance of this class that does the
-- initialization the way you want it.
class Initialize a where
  defP :: Bool -> IO a
  default defP :: (HasEot a, EotInitialize (Eot a)) => Bool -> IO a
  defP b = fromEot <$> eotDefP b
  gradP :: IO a
  gradP = defP True
  noGradP :: IO a
  noGradP = defP False

class EotInitialize eot where
  eotDefP :: Bool -> IO eot

instance (EotInitialize fields) => EotInitialize (Either fields Void) where
  eotDefP b = Left <$> eotDefP b

instance (Initialize x, EotInitialize xs) => EotInitialize (x, xs) where
  eotDefP b = do
    y <- defP b
    ys <- eotDefP b
    pure (y, ys)

instance EotInitialize () where
  eotDefP _ = pure ()

instance (TensorConstraints ty ki sz, SingI (IsFloatTy ty))
       => Initialize (Tensor ty ki sz) where
  defP b = setRequiresGrad b =<< kaimingUniform Nothing def def

instance (TensorConstraints ty ki '[outF, inF], SingI outF, SingI (IsFloatTy ty))
       => Initialize (LinearParam ty ki inF outF) where
  defP b = do
    r <- setRequiresGrad b =<< kaimingUniform (Just $ P.sqrt 5) def def
    r' <- setRequiresGrad b =<< kaimingUniformBias
    pure $ LinearParam r (Just r')

instance (TensorConstraints ty ki otherSz, SingI (IsFloatTy ty), SingI outChans)
       => Initialize (ConvParam ty ki outChans otherSz) where
  defP b = do
    r <- setRequiresGrad b =<< kaimingUniform (Just $ P.sqrt 5) def def
    r' <- setRequiresGrad b =<< kaimingUniformBias
    pure $ ConvParam r (Just r')

-- TODO This only allows initialization when the ty is float!?
instance (TensorConstraints ty ki sz, IsFloatTy ty ~ True)
       => Initialize (AffineParam ty ki sz) where
  defP b = do
    r <- setRequiresGrad b =<< uniform 0 1
    r' <- setRequiresGrad b =<< zeros
    pure $ AffineParam r (Just r')

instance (TensorConstraints ty ki '[nrEmbeddings, embeddingDim], SingI nrEmbeddings, SingI embeddingDim
         ,IsFloatTy ty ~ True)
       => Initialize (EmbeddingParam ty ki nrEmbeddings embeddingDim) where
  defP b = EmbeddingParam <$> (setRequiresGrad b =<< uniform 0 1)

instance (TensorConstraints ty ki '[hiddenF, inF], SingI inF, SingI hiddenF, KnownNat hiddenF
         ,SingI nrLayers, SingI isBidirectional, SingI gateSize, KnownNat gateSize, SingI batchFirst
         ,SingI (IsFloatTy ty), SingI '[gateSize TL.* hiddenF, inF]
         ,KnownNat (NrOfRNNDirections isBidirectional)
         ,SingI '[hiddenF, NrOfRNNDirections isBidirectional TL.* hiddenF])
       => Initialize (GenericRNNParam ty ki gateSize inF hiddenF nrLayers isBidirectional batchFirst) where
  defP b = do
    let k = 1 / (P.sqrt $ demoteN @hiddenF)
    wih0  <- repeatMDirection @isBidirectional (setRequiresGrad b =<< initUniform k)
    wihN1 <- VB.generateM (demoteN @nrLayers - 1) (\_ -> repeatMDirection @isBidirectional (setRequiresGrad b =<< initUniform k))
    bihN  <- VB.generateM (demoteN @nrLayers)     (\_ -> repeatMDirection @isBidirectional (setRequiresGrad b =<< initUniform k))
    whhN  <- VB.generateM (demoteN @nrLayers)     (\_ -> repeatMDirection @isBidirectional (setRequiresGrad b =<< initUniform k))
    bhhN  <- VB.generateM (demoteN @nrLayers)     (\_ -> repeatMDirection @isBidirectional (setRequiresGrad b =<< initUniform k))
    let rnn = GenericRNNParam wih0 wihN1 (Just bihN) whhN (Just bhhN)
    case demote @ki of
#if WITH_CUDA
      KCuda -> do
        -- We're totally abusing this function for something it was never meant
        -- to do. Internally, it places all of the weights into a vector in the
        -- right order. That's the piece we want here so that
        -- _cudnn_rnn_flatten_weight can make all of our weights contiguous.
        placeholder <- C.undefinedTensor
        genericRNN (\_ _ params has_bias nrLayers _ _ bidi batchFirst -> do
                      C._cudnn_rnn_flatten_weight params
                       -- This looks strange, but it's a reminder that we need
                       -- to set the stride correctly. If there is a bias, the
                       -- stride is 4, otherwise it's 2.
                              (if True then 4 else 2)
                              (demoteN @inF)
                      -- TODO This is atrocious and a terrible abuse. It
                      -- hardcodes these constants and it conflates RNN_RELU
                      -- with RNN_TANH (but I can't see how that would matter
                      -- here at all). In any case, we should clean it up.
                              (case demoteN @gateSize of
                                  1 -> 0 -- CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1
                                  3 -> 3 --  CUDNN_GRU = 3
                                  4 -> 2) -- CUDNN_LSTM = 2
                              (demoteN @hiddenF) (demoteN @nrLayers) batchFirst bidi)
                    (inF_ @inF) (hiddenF_ @hiddenF) (nrLayers_ @nrLayers) (isBidirectional_ @isBidirectional)
                    0 Train (demote @batchFirst) rnn placeholder placeholder
        pure ()
#endif
      _ -> pure ()
    pure rnn

instance (Initialize (GenericRNNParam ty ki 1 inF hiddenF nrLayers isBidirectional batchFirst))
       => Initialize (RNNParams ty ki inF hiddenF nrLayers isBidirectional batchFirst) where
  defP b = RNNParams <$> defP b

instance (Initialize (GenericRNNParam ty ki 3 inF hiddenF nrLayers isBidirectional batchFirst))
       => Initialize (GRUParams ty ki inF hiddenF nrLayers isBidirectional batchFirst) where
  defP b = GRUParams <$> defP b

instance (Initialize (GenericRNNParam ty ki 4 inF hiddenF nrLayers isBidirectional batchFirst))
       => Initialize (LSTMParams ty ki inF hiddenF nrLayers isBidirectional batchFirst) where
  defP b = LSTMParams <$> defP b

instance (TensorConstraints ty ki '[hiddenF, inF], SingI hiddenF, SingI (IsFloatTy ty))
       => Initialize (RNNCellParam ty ki inF hiddenF) where
  defP b = do
    let k = P.sqrt $ 1 / (fromIntegral $ demoteN @hiddenF)
    wih <- setRequiresGrad b =<< initUniform k
    bih <- setRequiresGrad b =<< initUniform k
    whh <- setRequiresGrad b =<< initUniform k
    bhh <- setRequiresGrad b =<< initUniform k
    pure $ RNNCellParam wih (Just bih) whh (Just bhh)

-- TODO This isn't technically legit. We never differentiate the
-- BatchNormState. But it's so handy!
instance Initialize (BatchNormState ty ki sz) where
  defP b = def

instance (Initialize a1, Initialize a2) => Initialize (a1,a2)
instance (Initialize a1, Initialize a2, Initialize a3) => Initialize (a1,a2,a3)
instance (Initialize a1, Initialize a2, Initialize a3, Initialize a4) => Initialize (a1,a2,a3,a4)
instance (Initialize a1, Initialize a2, Initialize a3, Initialize a4, Initialize a5) => Initialize (a1,a2,a3,a4,a5)
instance (Initialize a1, Initialize a2, Initialize a3, Initialize a4, Initialize a5, Initialize a6) => Initialize (a1,a2,a3,a4,a5,a6)

instance (TensorConstraints ty ki '[batch], KnownNat (NrOfRNNDirections isBidirectional), KnownNat nrLayers, KnownNat batch, KnownNat hiddenF)
       => Initialize (RNNState ty ki batch isBidirectional nrLayers hiddenF) where
  defP _ = RNNState <$> zeros

instance (TensorConstraints ty ki '[batch], KnownNat (NrOfRNNDirections isBidirectional), KnownNat nrLayers, KnownNat batch, KnownNat hiddenF)
       => Initialize (RNNStateBatchFirst ty ki batch isBidirectional nrLayers hiddenF) where
  defP _ = RNNStateBatchFirst <$> zeros

instance (TensorConstraints ty ki '[batch], KnownNat (NrOfRNNDirections isBidirectional), KnownNat nrLayers, KnownNat batch, KnownNat hiddenF)
       => Initialize (LSTMState ty ki batch isBidirectional nrLayers hiddenF) where
  defP _ = LSTMState <$> zeros <*> zeros

instance (TensorConstraints ty ki '[batch], KnownNat (NrOfRNNDirections isBidirectional), KnownNat nrLayers, KnownNat batch, KnownNat hiddenF)
       => Initialize (LSTMStateBatchFirst ty ki batch isBidirectional nrLayers hiddenF) where
  defP _ = LSTMStateBatchFirst <$> zeros <*> zeros

-- * Instances for ShowIO

instance {-# OVERLAPPING #-} (TensorConstraints ty ki outF, TensorConstraints ty ki inF)
       => ShowIO (LinearParam ty ki inF outF) where
  showIO (LinearParam x y) = [c|LinearParam #{x} #{y}|]

instance {-# OVERLAPPING #-} (TensorConstraints ty ki outF, TensorConstraints ty ki inF)
       => ShowIO (ConvParam ty ki inF outF) where
  showIO (ConvParam x y) = [c|ConvParam #{x} #{y}|]

instance {-# OVERLAPPING #-} (TensorConstraints ty ki sz)
        => ShowIO (AffineParam ty ki sz) where
  showIO (AffineParam x y) = [c|AffineParam #{x} #{y}|]

instance {-# OVERLAPPING #-} (TensorConstraints ty ki hiddenF, TensorConstraints ty ki inF)
       => ShowIO (RNNCellParam ty ki inF hiddenF) where
  showIO (RNNCellParam wih bih whh bhh) =
    [c|RNNCellParam Wih: #{wih} bih: #{bih} Whh: #{whh} bhh: #{bhh}|]
