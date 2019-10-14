{-# LANGUAGE DeriveDataTypeable, PackageImports, TemplateHaskell, TypeApplications, TypeFamilies #-}

-- | Two templates (@i@ and @s@) which allow for convenient indexing. This adds
-- no functionality itself, the templates are all trivial, and merely string
-- together operations from Torch.Tensor.
--
-- @[i|1|]@ is a (Tensor -> IO Tensor), a get operation which indexes with one
-- along the first dimension.
--
-- @[s|1|]@ is a (Tensor -> Tensor -> IO Tensor), a set operation which indexes
-- with one along the first dimension and then takes a second tensor the content
-- of which overwrites the remaining portion of the first. The original
-- unindexed, but updated, tensor is returned.
--
-- A few forms of both @i@ and @s@ are allowed:
-- * @[i|1,2,3|]@ using multiple indices for multiple dimensions
-- * @[i|1,:,3|]@ skipping a dimension
-- * @[i|2:|]@    taking all elements after the 2nd one
-- * @[i|:2|]@    taking all elements up to the 2nd one
-- * @[i|2:10|]@  taking all elements between the 2nd and the 10th
-- * @[i|$x,2|]@  indexes with a local variable, a number x
--
-- Note that [i|$x:$y|] is not allowed because the size of the resulting vector
-- is not known at compile time. TODO, some forms of this should be ok, for
-- example @[i|$x:$x+4|]@ because the size is known to be 4. We should allow
-- this forma t some point.
--
-- TODO Implement negative indices

module Torch.Indexing (i,s) where
import           Control.Monad
import           Control.Monad.Combinators.Expr
import           Data.Generics                          hiding (Prefix)
import           Data.List
import           Data.Singletons.TH
import qualified "template-haskell" Language.Haskell.TH as TH
import           Language.Haskell.TH.Quote
import           Text.Megaparsec
import           Text.Megaparsec.Char
import qualified Text.Megaparsec.Char.Lexer             as L
import           Torch.Tensor
import           Torch.Types

data IndexLiteral = IndexLInt Int
                  | IndexLFloat Float
                  | IndexLAnti String
                  | IndexLNegate IndexLiteral
                  | IndexLBracket IndexLiteral
                  | IndexLAdd IndexLiteral IndexLiteral
                  | IndexLSub IndexLiteral IndexLiteral
                  | IndexLMul IndexLiteral IndexLiteral
                  deriving (Data, Show)

data IndexTH = IIndexTH IndexLiteral
             | ISliceTH (Maybe IndexLiteral) (Maybe IndexLiteral) (Maybe IndexLiteral)
             | IEllipsisTH
             | IIndexMaskTH IndexLiteral
             | INewAxisTH
             | IIndexListTH [IndexTH]
             | IIndexSequenceTH IndexLiteral
             deriving (Data, Show)

type Parser = Parsec Void String

indexLitExp :: IndexLiteral -> TH.ExpQ
indexLitExp (IndexLInt i) = [|i|]
indexLitExp (IndexLFloat f) = [|f|]
indexLitExp (IndexLAnti s) = TH.varE $ TH.mkName s
indexLitExp (IndexLNegate e) = [|- $r|] where r = indexLitExp e
indexLitExp (IndexLBracket e) = indexLitExp e
indexLitExp (IndexLAdd a b) = [|$a' + $b'|]
  where a' = indexLitExp a
        b' = indexLitExp b
indexLitExp (IndexLSub a b) = [|$a' - $b'|]
  where a' = indexLitExp a
        b' = indexLitExp b
indexLitExp (IndexLMul a b) = [|$a' * $b'|]
  where a' = indexLitExp a
        b' = indexLitExp b
indexLitExpMaybe :: Maybe IndexLiteral -> TH.ExpQ

indexLitExpMaybe Nothing = [|Nothing|]
indexLitExpMaybe (Just l) = [|Just $l'|] where l' = indexLitExp l

intToLit :: (Applicative f, Integral a) => a -> f TH.Type
intToLit = pure . TH.LitT . TH.NumTyLit . fromIntegral

indexTHExpTensor :: IndexTH -> Int -> TH.Q TH.Exp
indexTHExpTensor (IIndexTH l) dim = [|(\t_ -> (Torch.Tensor.narrow1 (Torch.Types.dimension_ @($d)) t_ $v))|]
  where v = indexLitExp l
        d = intToLit dim
indexTHExpTensor (ISliceTH (Just (IndexLInt from)) Nothing Nothing) dim =
  [|(\t_ -> (Torch.Tensor.narrowFrom (Torch.Types.dimension_ @($d)) (Torch.Types.size_ @($from')) t_))|]
  where from' = pure (TH.LitT (TH.NumTyLit (fromIntegral from)))
        d = intToLit dim
indexTHExpTensor (ISliceTH Nothing Nothing (Just (IndexLInt to))) dim =
  [|(\t_ -> (Torch.Tensor.narrowTo (Torch.Types.dimension_ @($d)) (Torch.Types.size_ @($to')) t_))|]
  where to' = intToLit to
        d = intToLit dim
indexTHExpTensor (ISliceTH (Just (IndexLInt from)) Nothing (Just (IndexLInt to))) dim =
  [|(\t_ -> (Torch.Tensor.narrowFromTo (Torch.Types.dimension_ @($d)) (Torch.Types.size_ @($from')) (Torch.Types.size_ @($to')) t_))|]
  where from' = intToLit from
        to' = intToLit to
        d = intToLit dim
indexTHExpTensor (ISliceTH Nothing Nothing Nothing) dim = [|pure|]
indexTHExpTensor (IIndexListTH l) dim =
  foldl1' (\a b -> [|$a >=> $b|])
  $ zipWith (\e d -> case e of
                ISliceTH Nothing Nothing Nothing -> [|pure|]
                _                                -> indexTHExpTensor e d) l [0..]
indexTHExpTensor x _ = error $ "Indexing type not yet implemented for tensors " ++ show x

symbol :: (MonadParsec e s m, Token s ~ Char) => Tokens s -> m (Tokens s)
symbol  = L.symbol space

lexeme :: (MonadParsec e s m, Token s ~ Char) => m a -> m a
lexeme  = L.lexeme space

integer :: (MonadParsec e s m, Integral a, Token s ~ Char) => m a
integer = L.signed space $ lexeme L.decimal

float' :: (MonadParsec e s m, RealFloat a, Token s ~ Char) => m a
float' = L.signed space $ lexeme L.float

identifier :: (MonadParsec e s m, Token s ~ Char) => m [Char]
-- TODO Unicode
identifier = some (alphaNumChar <|> char '\'')

number :: (MonadParsec e s m, Token s ~ Char) => m [Char]
-- This is very loose
number = some (numberChar <|> char '-' <|> char '+' <|> char '.')

bracket :: Parser a -> Parser a
bracket = between (symbol "[") (symbol "]")

indexOf' :: (MonadParsec e s f, Token s ~ Char, Tokens s ~ [Char]) => f IndexLiteral
indexOf' = (symbol "$" >> (IndexLAnti <$> identifier))
          <|> (symbol "-" >> IndexLNegate <$> indexOf)
          <|> (IndexLInt <$> integer)
          <|> (IndexLFloat <$> float')

indexOf :: (MonadParsec e s m, Token s ~ Char, Tokens s ~ [Char]) => m IndexLiteral
indexOf = makeExprParser term table
  where term = (IndexLBracket <$> between (symbol "(") (symbol ")") indexOf)
               <|> (symbol "$" >> (IndexLAnti <$> identifier))
               <|> (IndexLInt <$> integer)
               <|> (IndexLFloat <$> float')
        table = [[prefix "-" IndexLNegate
                 ,prefix "+" id]
                ,[binary "*" IndexLMul]
                ,[binary "+" IndexLAdd
                 ,binary "-" IndexLSub]]
        binary  name f = InfixL  (f <$ symbol name)
        prefix  name f = Prefix  (f <$ symbol name)
        postfix name f = Postfix (f <$ symbol name)

parser1 :: Parser IndexTH
parser1 = (IIndexListTH <$> bracket (sepBy parser1 (symbol ",")))
          <|> (symbol "..." >> pure IEllipsisTH)
          <|> (try (symbol ":") >> (ISliceTH Nothing Nothing <$> optional indexOf))
          <|> try (do
                  i <- Just <$> indexOf
                  try (symbol ":")
                  i' <- optional indexOf
                  i'' <- optional (try (symbol ":") >> indexOf)
                  case i'' of
                    Nothing -> pure (ISliceTH i Nothing i')
                    Just _  -> pure (ISliceTH i i' i''))
          <|> try (do
                  i <- optional indexOf
                  try (symbol ":")
                  i <- optional indexOf
                  try (symbol ":")
                  i <- optional indexOf
                  pure (ISliceTH Nothing Nothing i))
          <|> (IIndexTH <$> indexOf)
          -- float TODO Float based-indexing!

parser :: Parser IndexTH
parser = (try $ parser1  <* eof) <|>
         (IIndexListTH <$> sepBy parser1 (symbol ",")) <* eof

runner :: (TH.Q TH.Exp -> TH.Q TH.Exp) -> QuasiQuoter
runner post = QuasiQuoter {
    quoteExp = \s -> do
      loc <- TH.location
      case parse parser (show (TH.loc_filename loc) ++ ":" ++ show (fst (TH.loc_start loc))) s of
        Left e  -> error $ show e ++ " " ++ show s
        Right x -> post (indexTHExpTensor x 0)
  , quotePat = undefined
  , quoteType = undefined
  , quoteDec = undefined
  }

i :: QuasiQuoter
i = runner (\r -> [|$r|])

s :: QuasiQuoter
s = runner (\r -> [|\x_ y_ -> $r x_ >>= \x_' -> set_ x_' y_ >> pure x_ |])
