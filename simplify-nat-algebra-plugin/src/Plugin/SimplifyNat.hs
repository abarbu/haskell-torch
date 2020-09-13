{-# LANGUAGE MultiWayIf, ViewPatterns, PostfixOperators, ScopedTypeVariables #-}
{-# OPTIONS_GHC -fwarn-incomplete-patterns -fwarn-overlapping-patterns -fwarn-unused-pattern-binds -Wno-type-defaults #-}

{-| Simplifies and normalizes type-level arithmetic that consists of equalities
  and inequalities in two ways. This is extremely useful when types combine
  together to form very complex expressions, and even more useful when paired
  with GHC.TypeLits.KnownNat.Solver which can discharge the corresponding
  KnownNat constraints once expressions are simplified. Types in haskell-torch
  which initially can span 200-300 lines, are compressed to 2-3 lines of
  straightforward human understandable and useful constraints.

*** The transformation performed in 1 is not always sound! ((n - 4) + n) is
    considered equal to n. This is not true! It should be (n, 4 <=? n) to avoid
    negative numbers. We don't generate these constraints at the moment since
    haskell-torch usually manually includes added inequalities for these cases,
    but you should be careful.

1. Looks for constraints like ((a <=? b) ~ True) and (a ~ b) where a and b have
kind Nat. When operations consist entirely of +, -, *, exp, div, mod, and log,
expressions are simplified and rebalanced. For example

  ((1 <=? ((((((Div ((((((inW + 4) - 4) - 1) + 1) - 1) - 1) 2 + 1) + 4) - 4) - 1) + 1) - 1)) ~ 'True)

is simplified to

  (4 <=? inW) ~ 'True

Note that no algorithm can gurantee such simplifications, your mileage may
vary. The simplification algorithm from "Computer Algebra and Symbolic
Computation: Elementary Algorithms" and "Computer Algebra and Symbolic
Computation: Mathematical Methods" by Joel S. Cohen is adapted by porting it
from Scheme, from https://github.com/dharmatech/mpl. Some improvements from
https://github.com/Metaxal/rascas are also included. Finally, a few hand-tuned
modifications are made to ensure that some common types in haskell-torch are
simplified correctly, rebalanced, and then rendered in a way that is expected of
Haskell types, e.g., no negative numbers.

2. Multiple inequalities that are redundant are resolved. (4 <=? n, 1 <=? n) is
resolved to just (4 <=? n). Only the most rudimentary of cases are handled,
under the assumption that other cases are simplified and normalized as shown
above.
-}

module Plugin.SimplifyNat(plugin) where

import qualified Data.Ratio as R
import Data.Maybe
import Data.List(intercalate,partition,groupBy,sortOn,nubBy)
import GHC.Stack
import TcType
import TcTypeNats
import GhcPlugins hiding (isExact)
import TcRnTypes
import Constraint
import TcPluginM
import Predicate
import TcEvidence
import TyCoRep (UnivCoProvenance( PluginProv ))

plugin :: Plugin
plugin = defaultPlugin {
    tcPlugin = install
  , pluginRecompile = purePlugin
  }

install :: x -> Maybe TcRnTypes.TcPlugin
install args = Just $ TcPlugin { tcPluginInit  = pure ()
                               , tcPluginSolve = run args
                               , tcPluginStop  = const $ pure ()
                               }

run :: p1 -> p2 -> [Ct] -> [Ct] -> [Ct] -> TcPluginM TcPluginResult
run args s givens deriveds wanteds = do
  case wanteds of
    [] -> simplifyExpr args s givens deriveds
    _  -> solveExpr args s givens deriveds wanteds
  where
    ppCts :: Outputable a => [a] -> SDoc
    ppCts [] = text "(empty)"
    ppCts cs = vcat (map ppr cs)

simplifyExpr args s givens deriveds = pure $ TcPluginOk [] []

data SimpleInequality = SIneq { sct :: Ct, st1 :: Type, st2 :: Type, sval :: Integer, svar :: String }

solveExpr args s givens derivdes wanteds = do
  simplifiedCts' <- mapM simplifyCt wanteds
  let simplifiedCts = catMaybes simplifiedCts'
  let redundantCts' =
          map (sortOn (\SIneq{sval=n} -> n))
        $ filter ((> 1) . length)
        $ map (nubBy (\SIneq{sval=n} SIneq{sval=n'} -> n==n'))
        $ groupBy (\SIneq{svar=v} SIneq{svar=v'} -> v==v')
        $ catMaybes
        $ map extractSimpleInequality wanteds
  let redundantCts = concatMap dischargeConstraints redundantCts'
  pure $ TcPluginOk (map fst simplifiedCts ++ redundantCts) (concat $ map snd simplifiedCts)

-- | The last constraint is not redundant, discharge all the others based on the
-- last one
dischargeConstraints :: [SimpleInequality] -> [(EvTerm, Ct)]
dischargeConstraints ins =
  map (\SIneq{sct=ct,st1=t1,st2=t2} ->
         -- We could easily construct real proof for this, but I don't know how
         -- to do that.
         (evByFiat "SimplifyNat" t1 t2, ct)) discharge
  where (_:discharge) = reverse ins

-- | The 'EvTerm' equivalent for 'Unsafe.unsafeCoerce'
evByFiat :: String -- ^ Name the coercion should have
         -> Type   -- ^ The LHS of the equivalence relation (~)
         -> Type   -- ^ The RHS of the equivalence relation (~)
         -> EvTerm
evByFiat name t1 t2 = EvExpr $ Coercion $ mkUnivCo (PluginProv name) Nominal t1 t2

-- | Only handles the simplest inequality constraints like
--   [WD] hole{co_ayI1T} {5}:: (2 <=? inW0) ~ 'True (CNonCanonical)
extractSimpleInequality :: Ct -> Maybe SimpleInequality
extractSimpleInequality ct =
  case classifyPredType (ctPred ct) of
    EqPred _ t1 t2 ->
      (case (splitTyConApp_maybe (typeKind t1)
            ,splitTyConApp_maybe t1
            ,splitTyConApp_maybe t2) of
         (Just ((== boolTyCon) -> True, y),
          Just (((== typeNatLeqTyCon) -> True), [tleft, tright]),
          Just (((== promotedTrueDataCon) -> True), _)) ->
           (case (tleft, tright) of
              (isNumLitTy -> Just n, getTyVar_maybe -> Just tv) ->
                Just $ SIneq ct t1 t2 n (occNameString $ nameOccName $ tyVarName tv)
              _ -> Nothing)
         _ -> Nothing)
    _ -> Nothing

simplifyCt :: Ct -> TcPluginM (Maybe ((EvTerm, Ct), [Ct]))
simplifyCt ct = do
  case classifyPredType (ctPred ct) of
    EqPred _ t1 t2 -> do
      o1 <- simplifyType ct t1 t2
      o2 <- simplifyType ct t2 t1
      case (o1,o2) of
        (Just (ev, t', w), Nothing) -> pure $ Just ((ev, ct), [w])
        (Nothing, Just (ev, t', w)) -> pure $ Just ((ev, ct), [w])
        (Just (_, t1', w1), Just (_, t2', w2)) ->
          pure $ Just $ ((evByFiat "SimplifyNat" (mkBoxedTupleTy [t1, t2]) (mkBoxedTupleTy [t1',t2'])
                         , ct)
                        ,[w1,w2])
        (Nothing, Nothing) -> pure Nothing
    _ -> do
      pure $ Nothing

simplifyType :: Ct -> Type -> Type -> TcPluginM (Maybe (EvTerm, Type, Ct))
simplifyType ct t1 t2 = do
  case splitTyConApp_maybe (typeKind t1) of
    Just (k, y) -> if | k == typeNatKindCon ->
                         do
                           simplifyNatTy ct t1 t2
                      | k == boolTyCon ->
                         do
                           simplifyBoolTy ct t1 t2
                      | otherwise -> do
                         pure $ Nothing
    _ -> do
      pure $ Nothing  

simplifyNatTy :: Ct -> Type -> Type -> TcPluginM (Maybe (EvTerm, Type, Ct))
simplifyNatTy ct t1 t2 = do
  let original = convertExpression t1
  let simplified = fmap (reorganizeE . automaticSimplify . automaticSimplify) original
  let t1' = fmap unconvertExpression simplified
  case t1' of
    (Just x) ->
      if not (eqType t1 x) then do
        tcPluginTrace "SimplifyNat simplified" (ppr t1 <+> text " to " <+> ppr x)
        w <- newWanted (ctLoc ct) (mkPrimEqPred x t2)
        pure $ Just (evByFiat "SimplifyNat" t1 x, x, mkNonCanonical w)
      else pure $ Nothing
    _ -> pure $ Nothing

simplifyBoolTy :: Ct -> Type -> Type -> TcPluginM (Maybe (EvTerm, Type, Ct))
simplifyBoolTy ct t1 t2 = do
  let original = convertFormula t1
  let simplified = fmap (reorganizeG . simplify . simplify) original
  let t1' = fmap (unconvertFormula ct) simplified
  case t1' of
    (Just x) ->
      if not (eqType t1 x) then do
        tcPluginTrace "SimplifyNat simplified" (ppr t1 <+> text " to " <+> ppr x)
        w <- newWanted (ctLoc ct) (mkPrimEqPred x t2)
        pure $ Just (evByFiat "SimplifyNat" t1 x, x, mkNonCanonical w)
      else
        pure $ Nothing
    _ -> pure $ Nothing

convertFormula ty
  | Just tv <- getTyVar_maybe ty = Just $ GBare $ ESymbol (Just (NoEq ty)) $ occNameString $ nameOccName $ tyVarName tv
  | Just n <- isNumLitTy ty  = Just $ GBare $ ENum $ fromIntegral n
  | Just s <- splitTyConApp_maybe ty =
      case s of
        ((== typeNatLeqTyCon) -> True, [t1,t2]) -> do
          e1 <- convertExpression t1
          e2 <- convertExpression t2
          Just $ GLeq e1 e2
        ((== promotedTrueDataCon) -> True, _) -> Nothing
        (tc, _) -> Nothing -- error $ "JJ" ++ (showSDocUnsafe $ ppr tc) ++ (show $ tc == promotedTrueDataCon) -- Nothing
  | otherwise = error $ "FF" ++ (showSDocUnsafe $ ppr ty) -- Nothing

convertExpression :: Type -> Maybe Expression
convertExpression ty
  | Just tv <- getTyVar_maybe ty = Just $ ESymbol (Just (NoEq ty)) $ occNameString $ nameOccName $ tyVarName tv
  | Just n <- isNumLitTy ty  = Just $ ENum $ fromIntegral n
  | Just s <- splitTyConApp_maybe ty =
      case s of
        (tc@((`elem` [typeNatAddTyCon,typeNatSubTyCon,typeNatMulTyCon,typeNatExpTyCon
               ,typeNatDivTyCon,typeNatLogTyCon,typeNatModTyCon])
            -> True), [t1,t2]) -> do
          e1 <- convertExpression t1
          e2 <- convertExpression t2
          if | tc == typeNatAddTyCon -> Just $ ESum [e1, e2]
             | tc == typeNatSubTyCon -> Just $ EDifference e1 e2
             | tc == typeNatMulTyCon -> Just $ EProduct [e1, e2]
             | tc == typeNatExpTyCon -> Just $ EPower e1 e2
             | tc == typeNatDivTyCon -> Just $ EDiv e1 e2
             | tc == typeNatModTyCon -> Just $ EModulo e1 e2
             | otherwise -> error $ "TT" ++ (showSDocUnsafe $ ppr ty) -- Nothing
        (tc@((`elem` [typeNatLogTyCon]) -> True), [t]) -> do
          e <- convertExpression t
          if | tc == typeNatLogTyCon -> Just $ ELog2 (ENum 2) e
             | otherwise -> error $ "LL" ++ (showSDocUnsafe $ ppr ty) -- Nothing
        _ -> Nothing -- error $ "II" ++ (showSDocUnsafe $ ppr ty) -- Nothing
  | otherwise = error $ "MM" ++ (showSDocUnsafe $ ppr ty) -- Nothing

unconvertFormula :: Ct -> Formula -> Type
unconvertFormula ct (GEq e1 e2) =
  mkTcEqPredLikeEv (ctEvidence ct) (unconvertExpression e1) (unconvertExpression e2)
unconvertFormula _ (GLeq e1 e2) =
  mkTyConApp typeNatLeqTyCon [unconvertExpression e1, unconvertExpression e2]
unconvertFormula _ (GBare e) = unconvertExpression e

unconvertExpression :: Expression -> Type
unconvertExpression (EProduct [x, EPower y (ENum ((== -1) -> True))]) =
  unconvertExpression (EDiv x y)
unconvertExpression (EProduct [x, EPower y (ENum n@((<0) -> True))]) =
  unconvertExpression (EDiv x (EPower y (ENum (- n))))
unconvertExpression (EProduct (ENum n@(isExact'' -> False):t))
  | R.numerator n == 1 =
    unconvertExpression $ case t of
                            [] -> error "Bad sum, not simplified"
                            [x] -> EDiv x (toENum $ R.denominator n)
                            l -> EDiv (EProduct l) (toENum $ R.denominator n)
  | otherwise =
    unconvertExpression $ case t of
                            [] -> error "Bad sum, not simplified"
                            [x] -> EDiv (EProduct [toENum (R.numerator n), x]) (toENum $ R.denominator n)
                            l -> EDiv (EProduct [toENum (R.numerator n), EProduct l]) (toENum $ R.denominator n)
unconvertExpression (ENum n)
  | R.denominator n == 1 = mkNumLitTy (R.numerator n)
  | otherwise = unconvertExpression (EDifference
                                      (toENum $ R.numerator n)
                                      (toENum $ R.denominator n))
unconvertExpression (ESum [ENum n@((<0) -> True), x]) = unconvertExpression (EDifference x (ENum (- n)))
unconvertExpression (ESum (ENum n@((<0) -> True):tail)) = unconvertExpression (EDifference (ESum tail) (ENum (- n)))
unconvertExpression (ESum l) =
  foldr1 (\x y -> mkTyConApp typeNatAddTyCon [x,y]) (map unconvertExpression l)
unconvertExpression (EProduct l) =
  foldr1 (\x y -> mkTyConApp typeNatMulTyCon [x,y]) (map unconvertExpression l)
unconvertExpression (EDifference x y) =
  mkTyConApp typeNatSubTyCon [unconvertExpression x, unconvertExpression y]
unconvertExpression (EPower x (ENum n@((<0) -> True))) = error "Negative power, can't convert type"
unconvertExpression (EPower x y) =
  mkTyConApp typeNatExpTyCon [unconvertExpression x, unconvertExpression y]
unconvertExpression (EDiv x y) =
  mkTyConApp typeNatDivTyCon [unconvertExpression x, unconvertExpression y]
unconvertExpression (EModulo x y) =
  mkTyConApp typeNatModTyCon [unconvertExpression x, unconvertExpression y]
unconvertExpression (ELog2 (ENum 2) y) =
  mkTyConApp typeNatLogTyCon [unconvertExpression y]
unconvertExpression (ESymbol (Just (NoEq ty)) _) = ty
unconvertExpression e = error $ "Failed to convert expression to type: " ++ show e

-- When simplifying formulas make sure that we don't end up with negative
-- numbers.

data Formula = GEq Expression Expression
             | GLeq Expression Expression
             | GBare Expression
             deriving (Eq)

instance Show Formula where
  show (GEq x y)  = show x ++ " = " ++ show y
  show (GLeq x y) = show x ++ " <= " ++ show y
  show (GBare x)  = show x

simplify :: Formula -> Formula
simplify (GEq x y) = case automaticSimplify (EDifference y x) of
                       EDifference x y -> GEq x y
                       e -> GEq e (ENum 0)
simplify (GLeq x y) = case automaticSimplify (EDifference y x) of
                       EDifference x y -> GLeq y x
                       e -> GLeq (ENum 0) e
simplify (GBare x) = GBare $ automaticSimplify x

reorganizeG :: Formula -> Formula
reorganizeG (GEq x y)  = rebalanceNegatives (rebalanceSides GEq) (monomap reorganizeE x) (monomap reorganizeE y)
reorganizeG (GLeq x y) = rebalanceNegatives (rebalanceSides GLeq) (monomap reorganizeE x) (monomap reorganizeE y)
reorganizeG (GBare x)  = GBare $ monomap reorganizeE x

rebalanceSides :: (Expression -> Expression -> p) -> Expression -> Expression -> p
rebalanceSides gfn (ENum n) (EDiv x (ENum y)) = gfn (ENum (n*y)) x
rebalanceSides gfn (EDiv x (ENum y)) (ENum n) = gfn (ENum (n*y)) x
rebalanceSides gfn (ENum n) (EProduct [ENum x, y])
  | R.numerator x == 1   = gfn (ENum (n * (fromIntegral $ R.denominator x))) y
  | R.denominator x == 1 = gfn (ENum n) (EProduct [ENum x, y])
  | otherwise            = gfn (ENum (n * (fromIntegral $ R.denominator x))) (EProduct [ENum x, y])
rebalanceSides gfn x y = gfn x y

-- 0 <= x - 1 ~> 1 <= x
rebalanceNegatives :: (Expression -> Expression -> p) -> Expression -> Expression -> p
rebalanceNegatives gfn (ENum 0) (EDifference x (ENum y@((> 0) -> True))) = gfn (ENum y) x
rebalanceNegatives gfn (EDifference x (ENum y@((> 0) -> True))) (ENum 0) = gfn x (ENum y)
rebalanceNegatives gfn x y = gfn x y

-- -1 + x ~> x - 1
-- This assumes that the formula is simplified!
reorganizeE :: Expression -> Expression
reorganizeE (ESum l) = case (neg, rest) of
                         ([], p) -> ESum p
                         ([ENum n], [p]) -> EDifference p (ENum (- n))
                         ([ENum n], ps) -> EDifference (ESum ps) (ENum (- n))
                         _ -> error "Probably running reorganize on an unsimplified expression"
  where (neg, rest) = partition (\x -> case x of
                                        (ENum ((< 0) -> True)) -> True
                                        _ -> False) l
reorganizeE x = x

data NoEq a = NoEq a

instance Eq (NoEq a) where
  a == b = True

data Expression = EPower Expression Expression
                | EProduct [Expression]
                | ESum [Expression]
                | EDiv Expression Expression
                | ENegation Expression
                | EDifference Expression Expression
                | EExp Expression
                | ELog1 Expression
                | ELog2 Expression Expression
                | ESqrt Expression
                | ENum Rational
                | ESymbol (Maybe (NoEq Type)) String
                | EFunction String [Expression]
                | EModulo Expression Expression
                deriving (Eq)

instance Show Expression where
  show (EPower x y) = show x ++ "^" ++ show y
  show (EProduct l) = "(" ++ intercalate " * " (map show l) ++ ")"
  show (ESum l) = "(" ++ intercalate " + " (map show l) ++ ")"
  show (EDiv x y) = "[" ++ "(" ++ show x ++ ")" ++ "/" ++ "(" ++ show y ++ ")" ++ "]"
  show (ENegation x) = "-" ++ show x
  show (EDifference x y) = show x ++ " - " ++ show y
  show (EExp x) = "exp(" ++ show x ++ ")"
  show (ELog1 x) = "log(" ++ show x ++ ")"
  show (ELog2 x y) = "log(" ++ show x ++ "," ++ show y ++ ")"
  show (ESqrt x) = "sqrt(" ++ show x ++ ")"
  show (EModulo x y) = "mod(" ++ show x ++ "," ++ show y ++ ")"
  show (ENum x) | R.denominator x == 1 = show (R.numerator x)
                | R.denominator x > 10000 = show (fromRational x :: Float)
                | otherwise = show x
  show (ESymbol _ x) = x
  show (EFunction fn args) = fn ++ "(" ++ intercalate ", " (map show args) ++ ")"

-- * Toplevel

automaticSimplify :: Expression -> Expression
automaticSimplify = automaticSimplify' . monomap automaticSimplify'
  where
    automaticSimplify' :: Expression -> Expression
    -- distributes multiplication over addtion
    automaticSimplify' (EProduct (n@ENum{}:ESum l:tail)) = sprod (ssum (map (n $*) l) : tail)
    automaticSimplify' x@EPower{}       = simplifyPower x
    automaticSimplify' x@EProduct{}     = simplifyProduct x
    automaticSimplify' x@ESum{}         = simplifySum x
    automaticSimplify' x@EDiv{}    = simplifyDiv x
    automaticSimplify' x@ENegation{}    = simplifyDifference x
    automaticSimplify' x@EDifference{}  = simplifyDifference x
    automaticSimplify' (EExp x)         = sexp x
    automaticSimplify' (ELog1 x)        = slog1 x
    automaticSimplify' (ELog2 x y)      = slog2 x y
    automaticSimplify' x@ESqrt{}        = ssqrt x
    automaticSimplify' x                = x

sprod :: [Expression] -> Expression
sprod = simplifyProduct . EProduct

ssum :: [Expression] -> Expression
ssum = simplifySum . ESum

ssqrt :: Expression -> Expression
ssqrt v@(ENum x) | isExact' (sqrt (fromRational x)) = ENum $ toRational (sqrt (fromRational x))
                 | otherwise = v $^ (ENum $ 1 R.% 2)
ssqrt v@x = v $^ (ENum $ 1 R.% 2)

sexp :: Expression -> Expression
sexp (ENum x) = ENum $ toRational $ exp $ fromRational x
sexp x = EExp x

sneg :: Expression -> Expression
sneg x = simplifyDifference (ENegation x)

sfn :: String -> [Expression] -> Expression
sfn fn args = EFunction fn args

slog1 :: Expression -> Expression
slog1 (ENum x) = ENum $ toRational $ log $ fromRational x
slog1 (EExp x) = x
slog1 x = ELog1 x

slog2 :: Expression -> Expression -> Expression
slog2 (ENum x) (ENum y) = ENum $ toRational $ log (fromRational x)
slog2 x y = ELog2 x y

-- $ for $ymbol
infixl 7 $+, $-, $*, $/
infixr 8 $^
  
($+) :: Expression -> Expression -> Expression
a $+ b = simplifySum (ESum [a,b])

($-) :: Expression -> Expression -> Expression
a $- b = simplifyDifference (EDifference a b)

($*) :: Expression -> Expression -> Expression
a $* b = simplifyProduct (EProduct [a,b])

($/) :: Expression -> Expression -> Expression
a $/ b = simplifyDiv (EDiv a b)

sbase :: Expression -> Expression
sbase (EPower e _) = e
sbase e = e

sexponent :: Expression -> Expression
sexponent (EPower _ e) = e
sexponent _ = ENum 1

-- * Difference, sum, product, power, div

simplifyDifference :: Expression -> Expression
simplifyDifference (ENegation x) = (ENum (-1)) $* x
simplifyDifference (EDifference x y) = x $+ (ENum (-1) $* y)
simplifyDifference _ = error "Not a difference"

simplifyDiv :: Expression -> Expression
simplifyDiv (EDiv x y) = x $* (y $^ (ENum (-1)))
simplifyDiv _ = error "Not a div"

($^) :: Expression -> Expression -> Expression
(ENum 0) $^ _ = ENum 0
(ENum 1) $^ _ = ENum 1
_ $^ (ENum 0) = ENum 0
v $^ (ENum 1) = v
(ENum x) $^ (ENum y) = ENum $ powRational x y
(EPower r s) $^ w@ENum{} = r $^ (s $* w)
(EProduct l) $^ w@ENum{} = sprod $ map ($^ w) l
v $^ w = EPower v w

simplifyPower :: Expression -> Expression
simplifyPower (EPower u v) = u $^ v
simplifyPower _ = error "Not a power"

mergeProducts :: [Expression] -> [Expression] -> [Expression]
mergeProducts [] x = x
mergeProducts x [] = x
mergeProducts (p:ps) (q:qs) =
  case simplifyProductRec [p, q] of
    []  -> mergeProducts ps qs
    [x] -> x : mergeProducts ps qs
    l -> (if | l == [p,q] -> (p : mergeProducts ps (q:qs))
             | l == [q,p] -> (q : mergeProducts (p:ps) qs)
             | otherwise -> error "Bad merge")

simplifyProductRec :: [Expression] -> [Expression]
simplifyProductRec [EProduct ps, EProduct qs] = mergeProducts ps qs
simplifyProductRec [EProduct ps, q] = mergeProducts ps [q]
simplifyProductRec [p, EProduct qs] = mergeProducts [p] qs
simplifyProductRec [ENum p, ENum q] = listOrNullIf1 (ENum $ p * q)
simplifyProductRec [ENum 1, x] = [x]
simplifyProductRec [x, ENum 1] = [x]
simplifyProductRec [p, q] = if | base' p == base' q ->
                                   listOrNullIf1 (fromJust (base' p) $^ (fromJust (exponent' p) $+ fromJust (exponent' q)))
                               | orderRelation q p -> [q,p]
                               | otherwise -> [p, q]
simplifyProductRec (EProduct ps:qs) = mergeProducts ps (simplifyProductRec qs)
simplifyProductRec (x:xs) = mergeProducts [x] (simplifyProductRec xs)
simplifyProductRec [] = error "Empty product"

simplifyProduct :: Expression -> Expression
simplifyProduct (EProduct [x]) = x
simplifyProduct (EProduct l) | anyZeros l = ENum 0
                             | otherwise = case simplifyProductRec l of
                                                [] -> ENum 1
                                                [x] -> x
                                                l -> EProduct l
simplifyProduct _ = error "Not a product"

mergeSums :: HasCallStack => [Expression] -> [Expression] -> [Expression]
mergeSums [] x = x
mergeSums x [] = x
mergeSums (p:ps) (q:qs) =
  case simplifySumRec [p, q] of
    []  -> mergeSums ps qs
    [x] -> x : mergeSums ps qs
    l -> (if | l == [p,q] -> (p : mergeSums ps (q:qs))
             | l == [q,p] -> (q : mergeSums (p:ps) qs)
             | otherwise -> error $ "Bad merge:\nl: " ++ show l ++ "\np: " ++ show p ++ "\nq: " ++ show q)

simplifySumRec :: HasCallStack => [Expression] -> [Expression]
simplifySumRec [ESum ps, ESum qs] = mergeSums ps qs
simplifySumRec [ESum ps, q] = mergeSums ps [q]
simplifySumRec [p, ESum qs] = mergeSums [p] qs
simplifySumRec [ENum p, ENum q] = listOrNullIf0 (ENum $ p + q)
simplifySumRec [ENum 0, x] = [x]
simplifySumRec [x, ENum 0] = [x]
simplifySumRec [p, q] = if | term' p == term' q ->
                               listOrNullIf0 (fromJust (term' p) $* (fromJust (const' p) $+ fromJust (const' q)))
                           | orderRelation q p -> [q, p]
                           | otherwise -> [p, q]
simplifySumRec (ESum ps:qs) = mergeSums ps (simplifySumRec qs)
simplifySumRec (x:xs) = mergeSums [x] (simplifySumRec xs)
simplifySumRec [] = error "Empty sum"

simplifySum :: Expression -> Expression
simplifySum (ESum [x]) = x
simplifySum (ESum xs) = case simplifySumRec xs of
                          [] -> ENum 0
                          [x] -> x
                          l -> ESum l
simplifySum _ = error "Not a sum"

-- * Order relations

base' :: Expression -> Maybe Expression
base' (EPower x _) = Just x
base' (ENum x) = Nothing
base' x = Just x

exponent' :: Expression -> Maybe Expression
exponent' (EPower _ y) = Just y
exponent' (ENum x) = Nothing
exponent' x = Just (ENum 1)

term' :: Expression -> Maybe Expression
term' (ENum u) = Nothing
term' (EProduct ((ENum _):urest)) = Just (EProduct urest)
term' u@(EProduct _) = Just u
term' u = Just $ EProduct [u]

const' :: Expression -> Maybe Expression
const' (ENum _) = Nothing
const' (EProduct ((ENum u1):_)) = Just $ ENum u1
const' u@(EProduct _) = Just $ ENum 1
const' _ = Just $ ENum 1

o3 :: [Expression] -> [Expression] -> Bool
o3 [] _ = True
o3 _ [] = False
o3 (u:us) (v:vs) | u /= v = orderRelation u v
                 | otherwise = o3 us vs

orderRelation :: Expression -> Expression -> Bool
orderRelation (ENum u) (ENum v) = u < v -- O-1
orderRelation (ESymbol _ u) (ESymbol _ v) = u < v -- O-2
orderRelation (EProduct u) (EProduct v) = o3 (reverse u) (reverse v)
orderRelation (ESum u) (ESum v) = o3 (reverse u) (reverse v)
orderRelation u@EPower{} v@EPower{} | base' u == base' v = orderRelation (fromJust $ exponent' u) (fromJust $ exponent' v)
                                    | otherwise = orderRelation (fromJust $ base' u) (fromJust $ base' v)
orderRelation u@(isFunction -> True) v@(isFunction -> True) = -- O-6
  case (u,v) of
    (ELog1 x, ELog1 y)       -> o3 [x] [y]
    (ELog2 x x', ELog2 y y') -> o3 [x] [y]
    (ESqrt x, ESqrt y) -> o3 [x] [y]
    _ -> orderRelation (fnArg u) (fnArg v)
orderRelation ENum{} (notENum -> True) = True -- O-7
orderRelation u@EProduct{} v@EPower{}            = orderRelation u (EProduct [v]) -- O-8 ...
orderRelation u@EProduct{} v@ESum{}              = orderRelation u (EProduct [v])
orderRelation u@EProduct{} v@(isFunction -> True) = orderRelation u (EProduct [v])
orderRelation u@EProduct{} v@ESymbol{}           = orderRelation u (EProduct [v])
orderRelation u@EPower{} v@ESum{}              = orderRelation u (EPower v (ENum 1)) -- O-9 ...
orderRelation u@EPower{} v@(isFunction -> True) = orderRelation u (EPower v (ENum 1))
orderRelation u@EPower{} v@ESymbol{}           = orderRelation u (EPower v (ENum 1))
orderRelation u@ESum{} v@(isFunction -> True) = orderRelation u (ESum [v])
orderRelation u@ESum{} v@ESymbol{}           = orderRelation u (ESum [v])
orderRelation u@(isFunction -> True) v@ESymbol{} | fnKind u == v = False -- O-12
                                                | otherwise = orderRelation (fnKind u) v
orderRelation u v = not $ orderRelation v u

-- * Misc

notENum :: Expression -> Bool
notENum ENum{} = False
notENum _ = True

listOrNullIf1 :: Expression -> [Expression]
listOrNullIf1 (ENum x) | x == 1 = []
                       | otherwise = [ENum x]
listOrNullIf1 e = [e]

listOrNullIf0 :: Expression -> [Expression]
listOrNullIf0 (ENum x) | x == 0 = []
                       | otherwise = [ENum x]
listOrNullIf0 e = [e]

anyZeros :: [Expression] -> Bool
anyZeros (ENum 0 : _) = True
anyZeros (_ : tail) = anyZeros tail
anyZeros [] = False

powRational :: Rational -> Rational -> Rational
powRational x y = toRational $ (fromRational x ** fromRational y :: Double)

isFunction :: Expression -> Bool
isFunction ELog1{} = True
isFunction ELog2{} = True
isFunction ESqrt{} = True
isFunction EFunction{} = True
isFunction _ = False

fnArg :: Expression -> Expression
fnArg (ELog1 x) = x
fnArg (ELog2 x _) = x
fnArg (ESqrt x) = x
fnArg _ = error "Not a symbolic function"

fnKind :: Expression -> Expression
fnKind EPower{}         = ESymbol Nothing "^"
fnKind EProduct{}       = ESymbol Nothing "*"
fnKind ESum{}           = ESymbol Nothing "+"
fnKind EDiv{}           = ESymbol Nothing "/"
fnKind ENegation{}      = ESymbol Nothing "-"
fnKind EDifference{}    = ESymbol Nothing "-"
fnKind EExp{}           = ESymbol Nothing "e"
fnKind ELog1{}          = ESymbol Nothing "log"
fnKind ELog2{}          = ESymbol Nothing "log"
fnKind ESqrt{}          = ESymbol Nothing "sqrt"
fnKind EModulo{}        = ESymbol Nothing "mod"
fnKind ENum{}           = ESymbol Nothing "num"
fnKind ESymbol{}        = ESymbol Nothing "sym"
fnKind (EFunction fn _) = ESymbol Nothing fn

isExact' :: Float -> Bool
isExact' x = x == fromInteger (round x)

isExact'' :: Rational -> Bool
isExact'' x = R.denominator x == 1

rationalMod :: Rational -> Rational -> Rational
rationalMod x y = ((R.numerator x * R.denominator y) `mod` (R.numerator y * R.denominator x)) R.% (R.denominator x * R.denominator y)

monomap :: (Expression -> Expression) -> Expression -> Expression
monomap f (EPower x y)        = f $ EPower (monomap f x) (monomap f y)
monomap f (EProduct l)        = f $ EProduct $ map (monomap f) l
monomap f (ESum l)            = f $ ESum $ map (monomap f) l
monomap f (EDiv x y)          = f $ EDiv (monomap f x) (monomap f y)
monomap f (ENegation x)       = f $ ENegation (monomap f x)
monomap f (EDifference x y)   = f $ EDifference (monomap f x) (monomap f y)
monomap f (EExp x)            = f $ EExp (monomap f x)
monomap f (ELog1 x)           = f $ ELog1 (monomap f x)
monomap f (ELog2 x y)         = f $ ELog2 (monomap f x) (monomap f y)
monomap f (ESqrt x)           = f $ ESqrt (monomap f x)
monomap f (EModulo x y)       = f $ EModulo (monomap f x) (monomap f y)
monomap f x@ENum{}            = f x
monomap f x@ESymbol{}         = f x
monomap f (EFunction fn args) = f $ EFunction fn (map (monomap f) args)

toENum :: Integral a => a -> Expression
toENum n = ENum $ fromIntegral n
