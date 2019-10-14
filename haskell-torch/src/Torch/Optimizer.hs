{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, ExistentialQuantification, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies, GADTs, MultiParamTypeClasses, OverloadedLabels, OverloadedStrings, PartialTypeSignatures     #-}
{-# LANGUAGE PolyKinds, RankNTypes, RecordWildCards, ScopedTypeVariables, TypeApplications, TypeFamilies, TypeFamilyDependencies  #-}
{-# LANGUAGE TypeInType, TypeOperators, UndecidableInstances                                                                      #-}
{-# OPTIONS_GHC -pgmP cc -optP -E -optP -undef -optP -std=c89 #-}

-- | A collection of optimizers. Before adding optimizers here please ensure
-- that they are floating-point equivalent to the ones in PyTorch if they share
-- the same name.
--
-- You will notice that the optimizers only take as input a single tensor and
-- that we deal with a list of optimizers. Common optimizers today deal with
-- parameters independently from one another allowing us to factor this out and
-- simplify the code, there's no downside to doing this.

module Torch.Optimizer where
import           Control.Monad
import           Data.Default
import           Data.IORef
import           Data.Maybe
import           Data.Singletons
import qualified Data.Text          as T
import           Foreign.C.Types
import           Foreign.ForeignPtr
import           GHC.Float
import           Numeric.Half
import           Prelude            as P
import qualified Torch.C.Scalar     as C
import qualified Torch.C.Tensor     as C
import qualified Torch.C.Types      as C
import qualified Torch.C.Variable   as C
import           Torch.Inplace
import           Torch.Operators
import           Torch.Tensor       as T hiding (take)
import           Torch.Types

-- * General utilies for dealing with parameters

zeroGradients_ :: [AnyTensor] -> IO ()
zeroGradients_ ts = mapM_ (\(AnyTensor t) -> clearGradinet t) ts

clipGradNorm_ :: [AnyTensor] -> Double -> Double -> IO Double
clipGradNorm_ ts maxNorm pNorm = do
  totalNorm :: Double <-
    if isInfinite pNorm then
      P.maximum
        <$> mapM (\(AnyTensor t) -> do
                     g <- gradient t
                     case g of
                       Nothing -> pure 0
                       Just grad ->
                         toDouble <$> (fromScalar =<< T.max =<< T.abs grad)) ts
    else
      ((** (1 / pNorm)) . P.sum)
        <$> mapM (\(AnyTensor t) -> do
                   g <- gradient t
                   case g of
                     Nothing -> pure 0
                     Just grad -> ((** pNorm) . toDouble)
                                 <$> (fromScalar =<< norm grad (NormP pNorm))) ts
  let clipCoef = maxNorm / (totalNorm + 1e-6)
  when (clipCoef < 1) $
    mapM_ (\(AnyTensor t) -> do
              g <- gradient t
              case g of
                Nothing   -> pure ()
                Just grad -> grad .*=@ fromDouble clipCoef >> pure ()) ts
  pure totalNorm

clipGradValue_ :: [AnyTensor] -> Double -> IO ()
clipGradValue_ ts clipValue = do
  mapM_ (\(AnyTensor t) -> do
            g <- gradient t
            case g of
              Nothing -> pure ()
              Just grad -> clamp_ grad
                                 (fromDouble (-clipValue))
                                 (fromDouble clipValue) >> pure ())
        ts

-- | TODO Various quantities here are floats, for example learning rates, but some
-- tensors are integral. We use a shady floatToScalar to fix this up. There must
-- be a better way but it seems so wasteful to parameterize all of the learners
-- by the kind of tensor that is being tuned.
fromFloat :: forall ty. (SingI ty) => Float -> (TensorTyToHs ty)
fromFloat arg = case sing :: Sing ty of
              STBool   -> P.round arg
              STByte   -> P.round arg
              STChar   -> toEnum $ P.round arg
              STShort  -> P.round arg
              STInt    -> P.round arg
              STLong   -> P.round arg
              STHalf   -> toHalf arg
              STFloat  -> arg
              STDouble -> float2Double arg

-- TODO Remove me
debugTs :: [AnyTensor] -> IO ()
debugTs ts = mapM_ (\(AnyTensor t) -> do
                           case t of
                             (Tensor t _) -> print =<< C.shape t
                           briefPrint t
                           g <- gradient t
                           case g of
                             Nothing -> print "No gradient"
                             Just g' -> briefPrint g') ts
  where briefPrint t = do
          putStrLn =<< (T.unpack . T.take 100 <$> str t)
          print "   ..."
          putStrLn =<< (T.unpack . T.reverse . T.take 100 . T.reverse <$> str t)

instance Optimizer a => Optimizer [a] where
  step_ opts = mapM step_ opts

instance Optimizer a => Optimizer (IORef a) where
  step_ opt = do
    r <- readIORef opt
    r' <- step_ r
    writeIORef opt r'
    pure opt

instance OptimizerWithLR a => OptimizerWithLR [a] where
  -- Gets the mean learning rate for a collection of optimizers
  getLearningRate  opts     = mean <$> mapM getLearningRate opts
    where mean l = P.sum l / fromIntegral (length l)
  setLearningRate_ opts lr = mapM (\opt -> setLearningRate_ opt lr) opts

instance OptimizerWithLR a => OptimizerWithLR (IORef a) where
  -- Gets the mean learning rate for a collection of optimizers
  getLearningRate  opt     = getLearningRate =<< readIORef opt
  setLearningRate_ opt  lr = do
    o  <- readIORef opt
    o' <- setLearningRate_ o lr
    writeIORef opt o'
    pure opt

-- TODO Remove me
-- | TODO Various quantities here are floats, for example learning rates, but some
-- tensors are integral. We use a shady floatToScalar to fix this up. There must
-- be a better way but it seems so wasteful to parameterize all of the learners
-- by the kind of tensor that is being tuned.
floatToScalar :: forall (ty :: TensorType) (ki :: TensorKind). (SingI ki, SingI ty)
              => Float -> IO (ForeignPtr C.CScalar)
floatToScalar arg = (case (sing :: Sing ty, demote @ki) of
                   (STBool, KCpu)    -> C.mkScalarCPUBool (P.round arg)
                   (STByte, KCpu)    -> C.mkScalarCPUByte (P.round arg)
                   (STChar, KCpu)    -> C.mkScalarCPUChar (P.round arg)
                   (STShort, KCpu)   -> C.mkScalarCPUShort (P.round arg)
                   (STInt, KCpu)     -> C.mkScalarCPUInt (P.round arg)
                   (STLong, KCpu)    -> C.mkScalarCPULong (P.round arg)
                   (STHalf, KCpu)    -> C.mkScalarCPUHalf (P.round arg)
                   (STFloat, KCpu)   -> C.mkScalarCPUFloat (CFloat arg)
                   (STDouble, KCpu)  -> C.mkScalarCPUDouble (CDouble (float2Double arg))
#if WITH_CUDA
                   (STBool, KCuda)   -> C.mkScalarCUDABool (P.round arg)
                   (STByte, KCuda)   -> C.mkScalarCUDAByte (P.round arg)
                   (STChar, KCuda)   -> C.mkScalarCUDAChar (P.round arg)
                   (STShort, KCuda)  -> C.mkScalarCUDAShort (P.round arg)
                   (STInt, KCuda)    -> C.mkScalarCUDAInt (P.round arg)
                   (STLong, KCuda)   -> C.mkScalarCUDALong (P.round arg)
                   (STHalf, KCuda)   -> C.mkScalarCUDAHalf (P.round arg)
                   (STFloat, KCuda)  -> C.mkScalarCUDAFloat (CFloat arg)
                   (STDouble, KCuda) -> C.mkScalarCUDADouble (CDouble (float2Double arg))
#endif
                ) >>= newForeignPtr C.deleteScalar

-- TODO Removeme
floatToScalarAs :: forall ty ki sz. (TensorConstraints ty ki sz)
                => Tensor ty ki sz -> Float -> IO (ForeignPtr C.CScalar)
floatToScalarAs _ s = floatToScalar @ty @ki s

-- * Stochastic Gradient Descent (SGD)

data SGD t = SGD { sgdLearningRate   :: !Float
                 , sgdMomentum       :: !(Maybe Float)
                 , sgdMomentumBuffer :: !(Maybe t)
                 , sgdDampening      :: !(Maybe Float)
                 , sgdWeightDecay    :: !(Maybe Float)
                 , sgdNesterov       :: !Bool }

instance Default (SGD t) where
  def = SGD { sgdLearningRate   = 0.01
            , sgdMomentum       = Nothing
            , sgdMomentumBuffer = Nothing
            , sgdDampening      = Nothing
            , sgdWeightDecay    = Nothing
            , sgdNesterov       = False }

data AnySGD = forall ty ki sz. TensorConstraints ty ki sz =>
              AnySGD (SGD (Tensor ty ki sz)) (Tensor ty ki sz)

-- TODO Sanity checking for the parameters
sgd_ :: AnySGD -> IO AnySGD
sgd_ (AnySGD param@SGD{..} p) = do
  param' <-
    maybeM' (gradient p)
     (pure param)
     (\d_p -> do
      maybe' sgdWeightDecay
             (pure ())
             (\w -> (d_p ..+= p .*@ fromFloat w) >> pure ())
      (d_p, param) <-
        maybe' sgdMomentum
          (pure (d_p, param))
          (\momentum -> do
            (param, mbuf) <-
              case sgdMomentumBuffer of
                Nothing -> do
                  mbuf <- noGrad =<< detach =<< clone d_p
                  pure (param { sgdMomentumBuffer = Just mbuf }, mbuf)
                Just mbuf -> do
                  mbuf .*=@ fromFloat momentum
                  mbuf ..+= d_p .*@ (fromFloat $ 1 - fromMaybe 0 sgdDampening)
                  pure (param, mbuf)
            d_p <- if sgdNesterov then do
                    d_p ..+= mbuf .*@ fromFloat momentum
                    pure d_p
                  else
                    pure mbuf
            pure (d_p, param))
      withoutGrad (p ..+= d_p .*@ (fromFloat $ - sgdLearningRate))
      pure param)
  pure (AnySGD param' p)

maybe' :: Maybe a -> b -> (a -> b) -> b
maybe' Nothing  n _ = n
maybe' (Just x) _ f = f x

maybeM' :: Monad m => m (Maybe a) -> m b -> (a -> m b) -> m b
maybeM' m n f = do
  m' <- m
  maybe' m' n f

instance Optimizer AnySGD where
  step_ = sgd_

instance OptimizerWithLR AnySGD where
  getLearningRate  (AnySGD s _)    = pure $ sgdLearningRate s
  setLearningRate_ (AnySGD s t) lr = pure $ AnySGD (s { sgdLearningRate = lr }) t

sgd' :: forall x. SGD x -> [AnyTensor] -> [AnyOptimizerWithLR]
sgd' p = map AnyOptimizerWithLR . sgd' p

sgd :: forall x. SGD x -> [AnyTensor] -> [AnySGD]
sgd params ts =
  map (\(AnyTensor t) -> mkSGD params t) ts
  where mkSGD params = AnySGD (SGD { sgdLearningRate   = sgdLearningRate params
                                   , sgdMomentum       = sgdMomentum params
                                   , sgdMomentumBuffer = Nothing
                                   , sgdDampening      = sgdDampening params
                                   , sgdWeightDecay    = sgdWeightDecay params
                                   , sgdNesterov       = sgdNesterov params })

-- * Adam

data Adam t = Adam { adamLearningRate :: !Float
                   , adamBeta1        :: !Float
                   , adamBeta2        :: !Float
                   , adamEps          :: !Float
                   , adamWeightDecay  :: !(Maybe Float)
                   , adamAMSGrad      :: !Bool
                   , adamStep         :: !Int
                   , adamExpAvg       :: !(Maybe t)
                   , adamExpAvgSq     :: !(Maybe t)
                   , adamMaxExpAvgSq  :: !(Maybe t)
                   }

instance Default (Adam t) where
  def = Adam { adamLearningRate = 1e-3
             , adamBeta1        = 0.9
             , adamBeta2        = 0.999
             , adamEps          = 1e-8
             , adamWeightDecay  = Nothing
             , adamAMSGrad      = False
             , adamStep         = 0
             , adamExpAvg       = Nothing
             , adamExpAvgSq     = Nothing
             , adamMaxExpAvgSq  = Nothing
             }

data AnyAdam = forall ty ki sz d. TensorConstraints ty ki sz =>
              AnyAdam (Adam (Tensor ty ki sz)) (Tensor ty ki sz)

instance Optimizer AnyAdam where
  step_ = adam_

instance OptimizerWithLR AnyAdam where
  getLearningRate  (AnyAdam s _)    = pure $ adamLearningRate s
  setLearningRate_ (AnyAdam s t) lr = pure $ AnyAdam (s { adamLearningRate = lr }) t

-- TODO Sanity checking for the parameters
adam_ :: AnyAdam -> IO AnyAdam
adam_ (AnyAdam param p) = withoutGrad $ do
  param' <-
    maybeM' (gradient p)
    (pure param)
    (\grad -> do
      -- initialization
      param <- case adamExpAvg param of
                Nothing -> do
                   expAvg <- noGrad =<< zeros
                   pure $ param { adamExpAvg = Just expAvg }
                _ -> pure param
      param <- case adamExpAvgSq param of
                Nothing -> do
                   expAvgSq <- noGrad =<< zeros
                   pure $ param { adamExpAvgSq = Just expAvgSq }
                _ -> pure param
      param <- case (adamAMSGrad param, adamMaxExpAvgSq param) of
                (True, Nothing) -> do
                   maxExpAvgSq <- noGrad =<< zeros
                   pure $ param { adamMaxExpAvgSq = Just maxExpAvgSq }
                _ -> pure param
      param <- pure $ param { adamStep = adamStep param + 1 }
      --
      maybe' (adamWeightDecay param)
             (pure ())
             (\w -> (grad ..+= p .*@ fromFloat w) >> pure ())
      -- update
      case (adamExpAvg param, adamExpAvgSq param, param) of
        (Just expAvg, Just expAvgSq, Adam{..}) -> do
          expAvg .*=@ fromFloat adamBeta1
          expAvg ..+= grad .*@ fromFloat (1 - adamBeta1)
          --
          expAvgSq .*=@ fromFloat adamBeta2
          addcmul_ expAvgSq (1 - fromFloat adamBeta2) grad grad
          --
          let biasCorrection1 = 1 - adamBeta1 ** (fromIntegral adamStep)
          let biasCorrection2 = 1 - adamBeta2 ** (fromIntegral adamStep)
          denom <- case (adamAMSGrad, adamMaxExpAvgSq) of
                    (True, Just maxExpAvgSq) -> do
                      -- Maintains the maximum of all 2nd moment running avg. till now
                      C.max_out__1 (tensorPtr maxExpAvgSq)
                                   (tensorPtr expAvgSq)
                                   (tensorPtr maxExpAvgSq)
                      -- Use the max. for normalizing running avg. of gradient
                      T.sqrt maxExpAvgSq
                    _ -> T.sqrt expAvgSq
          denom .+=@ fromFloat adamEps
          --
          let stepSize = adamLearningRate * P.sqrt biasCorrection2 / biasCorrection1
          addcdiv_ p (fromFloat (- stepSize)) expAvg denom
          pure param
        _ -> error "Adam bug; should have been initialized by now!")
  pure (AnyAdam param' p)

adam' :: forall x. Adam x -> [AnyTensor] -> [AnyOptimizerWithLR]
adam' p = map AnyOptimizerWithLR . adam p

adam :: forall x. Adam x -> [AnyTensor] -> [AnyAdam]
adam params ts =
  map (\(AnyTensor t) -> mkAdam params t) ts
  where mkAdam params = AnyAdam (Adam { adamLearningRate = adamLearningRate params
                                      , adamBeta1        = adamBeta1 params
                                      , adamBeta2        = adamBeta2 params
                                      , adamEps          = adamEps params
                                      , adamWeightDecay  = adamWeightDecay params
                                      , adamAMSGrad      = adamAMSGrad params
                                      , adamStep         = adamStep params
                                      , adamExpAvg       = Nothing
                                      , adamExpAvgSq     = Nothing
                                      , adamMaxExpAvgSq  = Nothing
                                      })
