{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, FlexibleContexts, FlexibleInstances, FunctionalDependencies, GADTs #-}
{-# LANGUAGE KindSignatures, MultiParamTypeClasses, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies, TypeInType, TypeOperators  #-}
{-# LANGUAGE UndecidableInstances                                                                                                     #-}
{-# options_ghc -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -pgmP cc -optP -E -optP -undef -optP -std=c89 #-}

-- | Operators to make tensor manipulations more readable. Some examples:
--    .* multiplies two tensors
--    .+ adds two tensors
--    ..+ adds two tensors, both of them are in IO
--    .+@ adds a tensor and a scalar
--    @+. adds a scalar and a tensor
--    .> compares two tensors
--    ..> compares two tensors, both of them are in IO
--    .>@ compares a tensor and a scalar
--    .*# multiplies a vector by a matrix
--    #*# matrix multiplication
--    .*= inplace multiplication of two tensors
--    .+= inplace addition of two tensors
--    ..+= inplace addition of two tensors, the tensor on right is in IO
--    .+=@ inplace addition of a tensor and a scalar
--
--  The examples above should be enough to read any operator.
--
--  Note that in some cases you may be tempted to avoid operators for
--  efficiency. For example when adding and scaling a tensor inplace:
--    a ..+= b .* 2
--  compared to
--    add_' a b 2
--  Since code would look messy in this style we add rules to convert
--  the latter into the former. No need to optimize by hand.
--
--  Operators are named according to these guidelines:
--    - . represents an argument that is a tensor
--    - @ represents an argument that is a scalar
--    - # represents a matrix when this is relevant, eg., matrix vector
--      multiplication is #*., the opposite is .*#
--    - any operator that takes as input a tensor is prefixed with .
--    - any operator that takes a scalar as its second argument is postfixed with @
--    - flipping the operator flips the markings
--    - repeated operators indicate that the operation accepts arguments in IO.
--      This is a convenience to avoid constantly having to use monad notation everywhere.

module Torch.Operators where
import           Torch.Inplace
import           Torch.Tensor

{- NB. We need to delay inlining a little in order to enable the rewrite rules to
 function. -}

infixl 6 .+, ..+, .+@, ..+@, @+., @+..

{-# INLINE[1] (.+) #-}
a .+ b = add a b

{-# INLINE[1] (.+@) #-}
a .+@ b = do
  b' <- toScalar b
  add a b'

{-# INLINE[1] (..+) #-}
a ..+ b = do
  a' <- a
  b' <- b
  add a' b'

{-# INLINE[1] (..+@) #-}
a ..+@ b = do
  a' <- a
  b' <- toScalar b
  add a' b'

a @+. b = b .+@ a
a @+.. b = b ..+@ a

infixl 6 .-, ..-, .-@, ..-@, @-., @-..

{-# INLINE[1] (.-) #-}
a .- b = sub a b

{-# INLINE[1] (.-@) #-}
a .-@ b = do
  b' <- toScalar b
  sub a b'

{-# INLINE[1] (..-) #-}
a ..- b = do
  a' <- a
  b' <- b
  sub a' b'

{-# INLINE[1] (..-@) #-}
a ..-@ b = do
  a' <- a
  b' <- toScalar b
  sub a' b'

a @-. b = b .-@ a
a @-.. b = b ..-@ a

infixl 7 .*, ..*, .*@, ..*@, @*., @*..

{-# INLINE[1] (.*) #-}
a .* b = mul a b

{-# INLINE[1] (..*) #-}
a ..* b = do
  a' <- a
  b' <- b
  mul a' b'

{-# INLINE[1] (.*@) #-}
a .*@ b = do
  b' <- toScalar b
  mul a b'

{-# INLINE[1] (..*@) #-}
a ..*@ b = do
  a' <- a
  b' <- b
  b'' <- toScalar b'
  mul a' b''

a @*. b = b .*@ a
a @*.. b = b ..*@ a

infixl 7 ./, ../, ./@, ../@, @/., @/..

{-# INLINE[1] (./) #-}
a ./ b = Torch.Tensor.div a b

{-# INLINE[1] (../) #-}
a ../ b = do
  a' <- a
  b' <- b
  Torch.Tensor.div a' b'

{-# INLINE[1] (./@) #-}
a ./@ b = do
  b' <- toScalar b
  Torch.Tensor.div a b'

{-# INLINE[1] (../@) #-}
a ../@ b = do
  a' <- a
  b' <- b
  b'' <- toScalar b'
  Torch.Tensor.div a' b''

a @/. b = b ./@ a
a @/.. b = b ../@ a

infix  4  .==, ./==, .<, .<=, .>=, .>
-- TODO
-- infixr 3  .&&
-- infixr 2  .||

a .== b = eq a b
a ./== b = neq a b
a .> b = gt a b
a .< b = lt a b
a .<= b = ltq a b
a .>= b = gtq a b

infixl 1 .+=, ..+=, .+=@, ..+=@, .-=, ..-=, .-=@, ..-=@, .*=, ..*=, .*=@, ..*=@, ./=, ../=, ./=@, ../=@

{-# INLINE[1] (.+=) #-}
a .+= b = add_ a b

{-# INLINE[1] (..+=) #-}
a ..+= b = do
  b' <- b
  add_ a b'

{-# INLINE[1] (.+=@) #-}
a .+=@ b = addScalar_ a b

{-# INLINE[1] (..+=@) #-}
a ..+=@ b = do
  b' <- b
  addScalar_ a b'

{-# INLINE[1] (.-=) #-}
a .-= b = sub_ a b

{-# INLINE[1] (..-=) #-}
a ..-= b = do
  b' <- b
  sub_ a b'

{-# INLINE[1] (.-=@) #-}
a .-=@ b = subScalar_ a b

{-# INLINE[1] (..-=@) #-}
a ..-=@ b = do
  b' <- b
  subScalar_ a b'

{-# INLINE[1] (.*=) #-}
a .*= b = mul_ a b

{-# INLINE[1] (..*=) #-}
a ..*= b = do
  b' <- b
  mul_ a b'

{-# INLINE[1] (.*=@) #-}
a .*=@ b = mulScalar_ a b

{-# INLINE[1] (..*=@) #-}
a ..*=@ b = do
  b' <- b
  mulScalar_ a b'

{-# INLINE[1] (./=) #-}
a ./= b = div_ a b

{-# INLINE[1] (../=) #-}
a ../= b = do
  b' <- b
  div_ a b'

{-# INLINE[1] (./=@) #-}
a ./=@ b = divScalar_ a b

{-# INLINE[1] (../=@) #-}
a ../=@ b = do
  b' <- b
  divScalar_ a b'

-- TODO Add more reasonable rules and variations on the ones below
-- https://downloads.haskell.org/~ghc/latest/docs/html/users_guide/glasgow_exts.html#rewrite-rules
-- TODO Pay more attention to numerical stability

{-# RULES
   "assignAddScale" forall a b s. a ..+= b .*@ s = add_' a b s
#-}

{-# RULES
   "addcmul_" forall a ten1IO ten2 s. a ..+= (ten1IO  ..* (ten2 .*@ s)) = ((\ten1 -> addcmul_ a s ten1 ten2) =<< ten1IO)
#-}

{-# RULES
   "addcdiv_" forall a ten1IO ten2 s. a ..+= (ten1IO  ../ (ten2 .*@ s)) = ((\ten1 -> addcdiv_ a s ten1 ten2) =<< ten1IO)
#-}
