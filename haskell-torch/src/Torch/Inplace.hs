{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, FlexibleContexts, FlexibleInstances, FunctionalDependencies, GADTs #-}
{-# LANGUAGE KindSignatures, MultiParamTypeClasses, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies, TypeInType, TypeOperators  #-}
{-# LANGUAGE UndecidableInstances                                                                                                     #-}
{-# OPTIONS_GHC -pgmP cc -optP -E -optP -undef -optP -std=c89 #-}

-- | Operations that mutate tensors.

module Torch.Inplace where
import           Data.Coerce
import           Data.Default
import           Data.Maybe
import           Data.Singletons
import           Foreign.C.Types
import           GHC.Int
import           Prelude         as P
import qualified Torch.C.Tensor  as C
import           Torch.Misc
import           Torch.Tensor
import           Torch.Types

-- * Mathematical operations

mul_ :: forall ty ki sz sz'.
      (SingI (BroadcastSizes sz sz'))
      => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor ty ki sz)
mul_ x@(Tensor{}) y@(Tensor{}) = do
  generic x y
  pure x
  where
    generic x@(Tensor t _) y@(Tensor t' _) = do
      let szExpanded = demoteNv @(BroadcastSizes sz sz')
      t'e <- C.expand_mab t' szExpanded (boolc True)
      C.mul__mt t t'e

mulScalar_ :: forall ty ki sz.
             Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
mulScalar_ x@(Tensor t _) alpha = do
  s <- toCScalar @ty @ki $ hsScalarToC alpha
  C.mul__ms t s
  pure x

addcmul_ :: forall ty ki sz sz' sz''.
          (SingI (BroadcastSizes sz' sz''), sz ~ (BroadcastSizes sz' sz''))
        => Tensor ty ki sz
        -> TensorTyToHs ty
        -> Tensor ty ki sz'
        -> Tensor ty ki sz''
        -> IO (Tensor ty ki sz)
addcmul_ (Tensor i _) val (Tensor t1 _) (Tensor t2 _) = do
  let szExpanded = demoteNv @(BroadcastSizes sz (BroadcastSizes sz' sz''))
  t1'  <- C.expand_mab t1 szExpanded (boolc True)
  t2'  <- C.expand_mab t2 szExpanded (boolc True)
  val' <- toCScalar @ty @ki $ hsScalarToC val
  rt <- C.addcmul__mtts i t1' t2' val'
  pure $ Tensor rt Nothing

div_ :: forall ty ki sz sz'.
      (SingI (BroadcastSizes sz sz'))
      => Tensor ty ki sz -> Tensor ty ki sz' -> IO (Tensor ty ki sz)
div_ x@(Tensor{}) y@(Tensor{}) = do
  generic x y
  pure x
  where
    generic x@(Tensor t _) y@(Tensor t' _) = do
      let szExpanded = demoteNv @(BroadcastSizes sz sz')
      t'e <- C.expand_mab t' szExpanded (boolc True)
      C.div__mt t t'e

divScalar_ :: forall ty ki sz.
             Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
divScalar_ x@(Tensor t _) alpha = do
  s <- toCScalar @ty @ki $ hsScalarToC alpha
  C.div__ms t s
  pure x

addcdiv_ :: forall ty ki sz sz' sz''.
          (SingI (BroadcastSizes sz' sz''), sz ~ (BroadcastSizes sz' sz''))
        => Tensor ty ki sz
        -> TensorTyToHs ty
        -> Tensor ty ki sz'
        -> Tensor ty ki sz''
        -> IO (Tensor ty ki sz)
addcdiv_ (Tensor i _) val (Tensor t1 _) (Tensor t2 _) = do
  let szExpanded = demoteNv @(BroadcastSizes sz (BroadcastSizes sz' sz''))
  t1'  <- C.expand_mab t1 szExpanded (boolc True)
  t2'  <- C.expand_mab t2 szExpanded (boolc True)
  val' <- toCScalar @ty @ki $ hsScalarToC val
  rt <- C.addcdiv__mtts i t1' t2' val'
  pure $ Tensor rt Nothing

add_ :: forall ty ki sz.
       Num (TensorTyToHs ty) =>
       Tensor ty ki sz -> Tensor ty ki sz -> IO (Tensor ty ki sz)
add_ x y = add_' x y 1

-- | x + alpha * y
add_' :: forall ty ki sz.
       Tensor ty ki sz -> Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
add_' x@(Tensor p _) y@(Tensor p' _) alpha = do
  s <- toCScalar @ty @ki $ hsScalarToC alpha
  C.add__mts p p' s
  pure x

addScalar_ :: forall ty ki sz.
       Num (TensorTyToHs ty) =>
       Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
addScalar_ x y = addScalar_' x y 1

-- | x + alpha * y
addScalar_' :: forall ty ki sz.
       Tensor ty ki sz -> TensorTyToHs ty -> TensorTyToHs ty -> IO (Tensor ty ki sz)
addScalar_' x@(Tensor p _) y alpha = do
  y' <- toCScalar @ty @ki $ hsScalarToC y
  alpha' <- toCScalar @ty @ki $ hsScalarToC alpha
  C.add__mss p y' alpha'
  pure x

sub_ :: forall ty ki sz.
       Num (TensorTyToHs ty) =>
       Tensor ty ki sz -> Tensor ty ki sz -> IO (Tensor ty ki sz)
sub_ x y = sub_' x y 1

-- | x + alpha * y
sub_' :: forall ty ki sz.
       Tensor ty ki sz -> Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
sub_' x@(Tensor p _) y@(Tensor p' _) alpha = do
  s <- toCScalar @ty @ki $ hsScalarToC alpha
  C.sub__mts p p' s
  pure x

subScalar_ :: forall ty ki sz.
       Num (TensorTyToHs ty) =>
       Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
subScalar_ x y = subScalar_' x y 1

-- | x + alpha * y
subScalar_' :: forall ty ki sz.
       Tensor ty ki sz -> TensorTyToHs ty -> TensorTyToHs ty -> IO (Tensor ty ki sz)
subScalar_' x@(Tensor p _) y alpha = do
  y' <- toCScalar @ty @ki $ hsScalarToC y
  alpha' <- toCScalar @ty @ki $ hsScalarToC alpha
  C.sub__mss p y' alpha'
  pure x

abs_ :: (SingI ty, SingI ki, SingI sz) =>
       Tensor ty ki sz -> IO (Tensor ty ki sz)
abs_ x@(Tensor t a) = do
  C.abs___t t
  pure x

copy_ :: Tensor ty ki sz -> Tensor ty ki sz -> IO (Tensor ty ki sz)
copy_ t@(Tensor dst _) (Tensor src _) = do
  C.copy__mtb dst src (boolc False)
  pure t

sin_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
sin_ x@(Tensor t _) = C.sin___t t >> pure x

sinh_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
sinh_ x@(Tensor t _) = C.sinh___t t >> pure x

asin_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
asin_ x@(Tensor t _) = C.asin___t t >> pure x

cos_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
cos_ x@(Tensor t _) = C.cos___t t >> pure x

cosh_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
cosh_ x@(Tensor t _) = C.cosh___t t >> pure x

acos_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
acos_ x@(Tensor t _) = C.acos___t t >> pure x

tan_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
tan_ x@(Tensor t _) = C.tan___t t >> pure x

tanh_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
tanh_ x@(Tensor t _) = C.tanh___t t >> pure x

atan_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
atan_ x@(Tensor t _) = C.atan___t t >> pure x

ceil_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
ceil_ x@(Tensor t _) = C.ceil___t t >> pure x

floor_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
floor_ x@(Tensor t _) = C.floor___t t >> pure x

clamp_ :: forall ty ki sz. TensorTyToHs ty -> TensorTyToHs ty -> Tensor ty ki sz -> IO (Tensor ty ki sz)
clamp_ lower upper x@(Tensor t _) = do
  l <- toCScalar @ty @ki (hsScalarToC lower)
  u <- toCScalar @ty @ki (hsScalarToC upper)
  C.clamp___tss t l u
  pure x

clampMax_ :: forall ty ki sz. TensorTyToHs ty -> Tensor ty ki sz -> IO (Tensor ty ki sz)
clampMax_ upper x@(Tensor t _) = do
  u <- toCScalar @ty @ki (hsScalarToC upper)
  C.clamp_max___ts t u
  pure x

clampMin_ :: forall ty ki sz. TensorTyToHs ty -> Tensor ty ki sz -> IO (Tensor ty ki sz)
clampMin_ lower x@(Tensor t _) = do
  l <- toCScalar @ty @ki (hsScalarToC lower)
  C.clamp_min___ts t l
  pure x

atan2_ :: forall ty ki sz sz'.
        Tensor ty ki sz -> Tensor ty ki sz -> IO (Tensor ty ki sz)
atan2_ x@(Tensor p _) (Tensor p' _) = do
  C.atan2__mt p p'
  pure x

digamma_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
digamma_ x@(Tensor t _) = C.digamma__m t >> pure x

erf_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
erf_ x@(Tensor t _) = C.erf__m t >> pure x

erfc_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
erfc_ x@(Tensor t _) = C.erfc__m t >> pure x

erfinv_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
erfinv_ x@(Tensor t _) = C.erfinv__m t >> pure x

exp_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
exp_ x@(Tensor t _) = C.exp__m t >> pure x

expm1_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
expm1_ x@(Tensor t _) = C.expm1__m t >> pure x

fmod_ :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
fmod_ x@(Tensor t _) div = do
  d <- toCScalar @ty @ki (hsScalarToC div)
  C.fmod__ms t d
  pure x

frac_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
frac_ x@(Tensor t _) = C.frac__m t >> pure x

log_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
log_ x@(Tensor t _) = C.log__m t >> pure x

log10_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
log10_ x@(Tensor t _) = C.log10__m t >> pure x

log1p_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
log1p_ x@(Tensor t _) = C.log1p__m t >> pure x

log2_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
log2_ x@(Tensor t _) = C.log2__m t >> pure x

mvlgamma_ :: forall ty ki sz. (SingI sz, TensorConstraints ty ki sz)
         => Tensor ty ki sz -> Int64 -> IO (Tensor ty ki sz)
mvlgamma_ t@(Tensor ptr _) dim = do
  C.mvlgamma__m6 ptr dim
  pure t

neg_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
neg_ x@(Tensor t _) = C.neg__m t >> pure x

reciprocal_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
reciprocal_ x@(Tensor t _) = C.reciprocal__m t >> pure x

remainder_ :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
remainder_ x@(Tensor t _) div = do
  d <- toCScalar @ty @ki (hsScalarToC div)
  C.remainder__ms t d >> pure x

round_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
round_ x@(Tensor t _) = C.round__m t >> pure x

sqrt_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
sqrt_ x@(Tensor t _) = C.sqrt__m t >> pure x

rsqrt_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
rsqrt_ x@(Tensor t _) = C.rsqrt__m t >> pure x

sign_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
sign_ x@(Tensor t _) = C.sign__m t >> pure x

trunc_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
trunc_ x@(Tensor t _) = C.trunc__m t >> pure x

-- * Non-linear activation functions

relu_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
relu_ x@(Tensor t a) = wrapTensorM (C.relu__m t) a

threshold_ :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> TensorTyToHs ty -> IO (Tensor ty ki sz)
threshold_ x@(Tensor t a) threshold value = do
  x <- toCScalar @ty @ki (hsScalarToC threshold)
  y <- toCScalar @ty @ki (hsScalarToC value)
  wrapTensorM (C.threshold___tss t x y) a

hardtanh_ :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> TensorTyToHs ty -> IO (Tensor ty ki sz)
hardtanh_ x@(Tensor t a) min max = do
  x <- toCScalar @ty @ki (hsScalarToC min)
  y <- toCScalar @ty @ki (hsScalarToC max)
  wrapTensorM (C.hardtanh___tss t x y) a

relu6_ :: Num (TensorTyToHs ty) => Tensor ty ki sz -> IO (Tensor ty ki sz)
relu6_ t = hardtanh_ t 0 6

elu_ :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> TensorTyToHs ty -> IO (Tensor ty ki sz)
elu_ x@(Tensor t a) alpha scale = do
  x <- toCScalar @ty @ki (hsScalarToC alpha)
  y <- toCScalar @ty @ki (hsScalarToC scale)
  s <- toCScalar @ty @ki 1
  wrapTensorM (C.elu___tsss t x y s) a

selu_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
selu_ x@(Tensor t a) = wrapTensorM (C.selu___t t) a

celu_ :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
celu_ x@(Tensor t a) alpha = do
  x <- toCScalar @ty @ki (hsScalarToC alpha)
  wrapTensorM (C.celu___ts t x) a

leakyRelu_ :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
leakyRelu_ x@(Tensor t a) negativeSlope = do
  x <- toCScalar @ty @ki (hsScalarToC negativeSlope)
  wrapTensorM (C.leaky_relu___ts t x) a

rrelu_ :: forall ty ki sz. Tensor ty ki sz -> TensorTyToHs ty -> TensorTyToHs ty -> DataPurpose -> IO (Tensor ty ki sz)
rrelu_ x@(Tensor t a) lower upper dataPurpose = do
  x <- toCScalar @ty @ki (hsScalarToC lower)
  y <- toCScalar @ty @ki (hsScalarToC upper)
  gen <- generatorFor (demote @ki)
  wrapTensorM (C.rrelu___tssbg t x y (boolc (dataPurpose == Train)) gen) a

sigmoid_ :: Tensor ty ki sz -> IO (Tensor ty ki sz)
sigmoid_ x@(Tensor t a) = wrapTensorM (C.sigmoid___t t) a

-- * Initialization

ones_ :: forall ty ki sz. (TensorConstraints ty ki sz)
      => Tensor ty ki sz -> IO (Tensor ty ki sz)
ones_ t@(Tensor ptr _) = do
  C.ones_out__ta ptr (demoteNv @sz)
  pure t

zeros_ :: forall ty ki sz. (TensorConstraints ty ki sz)
       => Tensor ty ki sz -> IO (Tensor ty ki sz)
zeros_ t@(Tensor ptr _) = do
  C.zeros_out__ta ptr (demoteNv @sz)
  pure t

eye_ :: forall ty ki sz0 sz1. (TensorConstraints ty ki sz0, SingI sz1)
     => Tensor ty ki '[sz0,sz1] -> IO (Tensor ty ki '[sz0,sz1])
eye_ t@(Tensor ptr _) = do
  C.eye_out__t66 ptr (demoteN @sz0) (demoteN @sz1)
  pure t

data FanMode = FanInMode | FanOutMode
  deriving (Show, Eq)

instance Default FanMode where
  def = FanInMode

calculateFanInOut :: [Int] -> (Int, Int)
calculateFanInOut []                    = error "Cannot compute fan in or out with fewer than two dimensions"
-- The PyTorch code doesn't have this case, but it simplfies so much downstream code
calculateFanInOut [fin]                 = (fin,fin)
calculateFanInOut [fin,fout]            = (fin,fout)
calculateFanInOut (outMaps:inMaps:rest) = (inMaps * product rest, outMaps * product rest)

calculateFan t FanInMode  = fst $ calculateFanInOut t
calculateFan t FanOutMode = snd $ calculateFanInOut t

-- | PyTorch calls this a gain computed for a nonlinearity. That's kind of
-- strange.
data GainNonlinearity = GainLinear
                      | GainConv
                      | GainSigmoid
                      | GainTanh
                      | GainRelu
                      | GainLeakyRelu
  deriving (Show, Eq)

instance Default GainNonlinearity where
  def = GainLeakyRelu

calculateGain :: GainNonlinearity -> Maybe Double -> Double
calculateGain GainLinear    _     = 1
calculateGain GainConv      _     = 1
calculateGain GainSigmoid   _     = 1
calculateGain GainTanh      _     = 5/3
calculateGain GainRelu      _     = P.sqrt 2
calculateGain GainLeakyRelu param = P.sqrt(2.0 / (1 + (fromMaybe 0.01 param) ** 2))

kaimingUniform_ :: forall sz ty ki. (TensorConstraints ty ki sz, SingI (IsFloatTy ty))
               => Tensor ty ki sz -> Maybe Double -> GainNonlinearity -> FanMode -> IO (Tensor ty ki sz)
kaimingUniform_ t a gainMode fanMode = do
  let fan   = calculateFan (demoteNs @sz) fanMode
  let gain  = calculateGain gainMode a
  let std   = gain / P.sqrt (fromIntegral fan)
  let bound = P.sqrt 3 * std -- Calculate uniform bounds from standard deviation
  withoutGrad $ uniform_ t (- bound) bound

kaimingUniformBias_ :: forall sz ty ki. (TensorConstraints ty ki sz, SingI (IsFloatTy ty))
                    => Tensor ty ki sz -> IO (Tensor ty ki sz)
kaimingUniformBias_ t = do
  let (fanIn,_) = calculateFanInOut (demoteNs @sz)
  let bound = 1 / P.sqrt (fromIntegral fanIn)
  withoutGrad $ uniform_ t (- bound) bound

kaimingNormal_ :: forall sz ty ki. (TensorConstraints ty ki sz, SingI (IsFloatTy ty))
               => Tensor ty ki sz -> Maybe Double -> GainNonlinearity -> FanMode -> IO (Tensor ty ki sz)
kaimingNormal_ t a gainMode fanMode = do
  let fan   = calculateFan (demoteNs @sz) fanMode
  let gain  = calculateGain gainMode a
  let std   = gain / P.sqrt (fromIntegral fan)
  withoutGrad $ normal_ t 0 std

xavierUniform_ :: forall sz ty ki. (TensorConstraints ty ki sz, SingI (IsFloatTy ty))
               => Tensor ty ki sz -> Maybe Double -> IO (Tensor ty ki sz)
xavierUniform_ t gain = do
  let (fanIn,fanOut)   = calculateFanInOut (demoteNs @sz)
  let std = fromMaybe 1 gain * P.sqrt (2.0 / (fromIntegral $ fanIn + fanOut))
  let a = std * P.sqrt 3 -- Calculate uniform bounds from standard deviation
  withoutGrad $ uniform_ t (- a) a

xavierNormal_ :: forall sz ty ki. (TensorConstraints ty ki sz, SingI (IsFloatTy ty))
               => Tensor ty ki sz -> Maybe Double -> IO (Tensor ty ki sz)
xavierNormal_ t gain = do
  let (fanIn,fanOut)   = calculateFanInOut (demoteNs @sz)
  let std = fromMaybe 1 gain * P.sqrt (2.0 / (fromIntegral $ fanIn + fanOut))
  withoutGrad $ normal_ t 0 std

-- * Random operations

uniform_ :: forall ty ki sz. (SingI sz, TensorConstraints ty ki sz)
         => Tensor ty ki sz -> Double -> Double -> IO (Tensor ty ki sz)
uniform_ t@(Tensor ptr _) l h = do
  C.uniform__mddg ptr (coerce l) (coerce h) =<< generatorFor (demote @ki)
  pure t

random_ :: forall ty ki sz. (SingI sz, TensorConstraints ty ki sz)
         => Tensor ty ki sz -> Int64 -> Int64 -> IO (Tensor ty ki sz)
random_ t@(Tensor ptr _) l h = do
  C.random__m66g ptr (coerce l) (Just $ coerce h) =<< generatorFor (demote @ki)
  pure t

normal_ :: forall ty ki sz. (SingI sz, TensorConstraints ty ki sz)
        => Tensor ty ki sz -> Double -> Double -> IO (Tensor ty ki sz)
normal_ t@(Tensor ptr _) m v = do
  C.normal__mddg ptr (CDouble m) (CDouble v) =<< generatorFor (demote @ki)
  pure t

bernoulli_ :: forall ty ki sz. (SingI sz, TensorConstraints ty ki sz)
           => Tensor ty ki sz -> Double -> IO (Tensor ty ki sz)
bernoulli_ t@(Tensor ptr _) p = do
  C.bernoulli__mdg ptr (coerce p) =<< generatorFor (demote @ki)
  pure t

exponential_ :: forall ty ki sz. (SingI sz, TensorConstraints ty ki sz)
           => Tensor ty ki sz -> Double -> IO (Tensor ty ki sz)
exponential_ t@(Tensor ptr _) p = do
  C.exponential__mdg ptr (coerce p) =<< generatorFor (demote @ki)
  pure t

geometric_ :: forall ty ki sz. (SingI sz, TensorConstraints ty ki sz)
           => Tensor ty ki sz -> Double -> IO (Tensor ty ki sz)
geometric_ t@(Tensor ptr _) p = do
  C.geometric__mdg ptr (coerce p) =<< generatorFor (demote @ki)
  pure t

cauchy_ :: forall ty ki sz. (SingI sz, TensorConstraints ty ki sz)
        => Tensor ty ki sz -> Double -> Double -> IO (Tensor ty ki sz)
cauchy_ t@(Tensor ptr _) m s = do
  C.cauchy__mddg ptr (coerce m) (coerce s) =<< generatorFor (demote @ki)
  pure t

logNormal_ :: forall ty ki sz. (SingI sz, TensorConstraints ty ki sz)
        => Tensor ty ki sz -> Double -> Double -> IO (Tensor ty ki sz)
logNormal_ t@(Tensor ptr _) m s = do
  C.log_normal__mddg ptr (coerce m) (coerce s) =<< generatorFor (demote @ki)
  pure t

constant_ :: forall ty ki sz. (TensorConstraints ty ki sz)
          => Tensor ty ki sz -> TensorTyToHs ty -> IO (Tensor ty ki sz)
constant_ t@(Tensor ptr _) c = do
  fill <- toCScalar @ty @ki (hsScalarToC c)
  C.full_out__tas ptr (demoteNv @sz) fill
  pure t
