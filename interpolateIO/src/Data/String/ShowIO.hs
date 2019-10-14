{-# LANGUAGE ScopedTypeVariables, MultiParamTypeClasses   #-}
{-# LANGUAGE FlexibleContexts, DefaultSignatures #-}
{-# LANGUAGE FlexibleInstances, UndecidableInstances #-}

module Data.String.ShowIO (
-- * Just like Show but in IO. Takes the tedium out of printing impure values.
 ShowIOS, ShowIO, showIOs, showIO, showsPrecIO, showListIO, showStringIO, IsStringIO, fromStringIO
) where

import Data.String
import Generics.Eot as GE

-- | The @showsIO@ functions return a function that prepends the
-- output 'String' to an existing 'String'.  This allows constant-time
-- concatenation of results using lifted functione composition @>=>@.
type ShowIOS = String -> IO String

-- | equivalent to 'showsPrec' with a precedence of 0.
showIOs           :: (ShowIO a) => a -> ShowIOS
showIOs           =  showsPrecIO 0

-- | utility function converting a 'String' to a show function that
-- simply prepends the string unchanged.
showStringIO      :: String -> ShowIOS
showStringIO s s' =  pure $ s ++ s'

-- | Conversion of values to readable 'IO String's.
class  ShowIO a  where
    -- | Convert a value to a readable 'IO String'.
    --
    -- 'showsPrecIO' should satisfy the law
    --
    -- > do
    -- >  sr <- showsPrecIO d x r
    -- >  srs <- showsPrecIO d x (r ++ s)
    -- >  pure $ sr ++ s  ==  srs

    showsPrecIO :: Int    -- ^ the operator precedence of the enclosing
                         -- context (a number from @0@ to @11@).
                         -- Function application has precedence @10@.
                ->  a    -- ^ the value to be converted to a 'IO String'
                -> ShowIOS

    -- | A specialised variant of 'showsPrecIO', using precedence context
    -- zero, and returning an ordinary 'IO String'.
    showIO      :: a   -> IO String
    default showIO :: (HasEot a, EotShowIO GE.Datatype (Eot a)) => a -> IO String
    showIO a = eotShowIO (datatype (Proxy :: Proxy a)) (toEot a)

    -- | The method 'showListIO' is provided to allow the programmer to
    -- give a specialised way of showing lists of values.
    -- For example, this is used by the predefined 'Show' instance of
    -- the 'Char' type, where values of type 'String' should be shown
    -- in double quotes, rather than between square brackets.
    showListIO  :: [a] -> ShowIOS

    showsPrecIO _ x s = (++ s) <$> showIO x 
    showListIO ls   s = showListIO__ showIOs ls s

instance {-# OVERLAPS #-} Show a => ShowIO a where
  showIO x = pure $ show x

instance {-# OVERLAPS #-} ShowIO a => ShowIO (IO a) where
  showIO x = x >>= showIO

showListIO__ :: (a -> ShowIOS) ->  [a] -> ShowIOS
showListIO__ _     []     s = pure $ "[]" ++ s
showListIO__ showx (x:xs) s = ('[' :) <$> (showx x =<< showl xs)
  where
    showl []     = pure (']' : s)
    showl (y:ys) = (',' :) <$> (showx y =<< showl ys)

class IsStringIO a where
  fromStringIO :: String -> IO a

instance IsString a => IsStringIO a where
  fromStringIO x = pure $ fromString x

class EotShowIO meta eot where
  eotShowIO :: meta -> eot -> IO String

instance (EotShowIO [GE.Constructor] a) => EotShowIO GE.Datatype a where
  eotShowIO meta a = eotShowIO (constructors meta) a

instance (EotShowIO [String] this, EotShowIO Int this, EotShowIO [GE.Constructor] next)
       => EotShowIO [GE.Constructor] (Either this next) where
  eotShowIO (m:_) (Left this)  = case m of
    Constructor con (Selectors fieldNames) ->
      (\x -> con <> " { " <> x <> " } ") <$> eotShowIO fieldNames this
    Constructor con (NoSelectors nr) ->
      (\x -> con <> " { " <> x <> " } ") <$> eotShowIO nr this
    Constructor con NoFields ->
      pure con
  eotShowIO (_:ms) (Right next) = eotShowIO ms next
  eotShowIO [] _  = error "Impossible"

instance {-# OVERLAPS #-} ShowIO x => EotShowIO Int (x, ()) where
  eotShowIO _ (x, ()) = showIO x

instance {-# OVERLAPS #-} (ShowIO x, EotShowIO Int xs) => EotShowIO Int (x, xs) where
  eotShowIO n (x, xs) = do
    ps <- showIO x
    ps' <- eotShowIO (n + 1) xs
    pure $ ps <> " " <> ps'

instance {-# OVERLAPS #-} ShowIO x => EotShowIO [String] (x, ()) where
  eotShowIO [name] (x, ()) = (\v -> name <> " = " <> v) <$> showIO x
  eotShowIO _ _ = error "Impossible"

instance {-# OVERLAPS #-} (ShowIO x, EotShowIO [String] xs) => EotShowIO [String] (x, xs) where
  eotShowIO (name:names) (x, xs) = do
    ps <- showIO x
    ps' <- eotShowIO names xs
    pure $ name <> " = " <> ps <> ", " <> ps'
  eotShowIO [] _ = error "Impossible"

instance EotShowIO meta () where
  eotShowIO _ _ = pure ""

instance EotShowIO meta Void where
  eotShowIO _ _ = pure ""
