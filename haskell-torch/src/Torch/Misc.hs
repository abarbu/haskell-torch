{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, EmptyCase, FlexibleContexts, FlexibleInstances                  #-}
{-# LANGUAGE FunctionalDependencies, GADTs, KindSignatures, LambdaCase, MultiParamTypeClasses, OverloadedLabels, OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures, PolyKinds, QuasiQuotes, RankNTypes, ScopedTypeVariables, TemplateHaskell, TypeApplications     #-}
{-# LANGUAGE TypeFamilies, TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances                                 #-}

-- | Miscellaneous helper functions that don't depend on anything in this
-- package.
module Torch.Misc where
import           Data.Aeson
import           Data.Hashable
import           Data.IORef
import           Data.Sequence             (Seq)
import qualified Data.Sequence             as S
import           Data.Singletons.Prelude   as SP hiding (Seq)
import           Data.String
import           Data.Text                 (Text)
import qualified Data.Text                 as T
import qualified Data.Vector.Storable      as V
import           Foreign.C.Types
import           Foreign.ForeignPtr
import           Foreign.ForeignPtr.Unsafe
import           Foreign.Ptr
import           Foreign.Storable
import           GHC.Natural               (Natural)
import           GHC.TypeLits              as TL
import           System.CPUTime
import qualified System.Directory          as P
import qualified System.FilePath           as P
import           System.Random             (randomRIO)
import           Text.Printf               (printf)

-- * Demoting types to values

-- | Demote a type-level number to a value
demoteN :: forall a b. (SingKind (KindOf a), SingI a, Integral (Demote (KindOf a)), Num b) => b
demoteN = fromIntegral (demote @a)

-- | Demote a type-level list of numbers to a list of values
demoteNs :: forall a b. (SingKind (KindOf a), SingI a, [Natural] ~ Demote (KindOf a), Integral b) => [b]
demoteNs = map fromIntegral (demote @a)

-- | Demote a type-level list of numbers to a vector of values
demoteNv :: forall a b. (SingKind (KindOf a), SingI a, [Natural] ~ Demote (KindOf a), Integral b, Storable b) => V.Vector b
demoteNv = V.fromList (map fromIntegral (demote @a))

-- | Demote a type-level list of numbers to a vector of ints. This is
-- particularly useful when not specifying the type would lead to ambiguous type
-- errors and the defaulting mechanism doesn't pick up the slack.
demoteIntV :: forall a. (SingKind (KindOf a), SingI a, [Natural] ~ Demote (KindOf a)) => V.Vector Int
demoteIntV = V.fromList (map fromIntegral (demote @a))

-- * Convenience types for multiple singleton constraints

type Sing2 x y = (SingI x, SingI y)
type Sing3 x y z = (SingI x, SingI y, SingI z)

type NonzeroSing x = (SingI x, 1 TL.<= x)
type NonzeroSing2 x y = (SingI x, SingI y, 1 TL.<= x, 1 TL.<= y)
type NonzeroSing3 x y z = (SingI x, SingI y, SingI z, 1 TL.<= x, 0 TL.<= y, 0 TL.<= z)

-- * A potential type defaulting mechanism
-- TODO Should remove this, it never worked out

data Def

type family Defaulted f a b where
  Defaulted f (f x) _ = x
  Defaulted _ Def b = b

-- * RNG operations

randomElement :: Seq a -> IO (Maybe (a, Seq a))
randomElement s | S.null s  = pure Nothing
                | otherwise = do
                    i <- randomRIO (0, S.length s - 1)
                    case S.lookup i s of
                      Just e -> pure $ Just (e, S.deleteAt i s)
                      _      -> error "impossible"

shuffleList :: [a] -> IO [a]
shuffleList [] = pure []
shuffleList xs = do
  idx <- randomRIO (0, length xs - 1)
  let (left, (mid:right)) = splitAt idx xs
  fmap (mid:) $ shuffleList $ left ++ right

-- * Stopwatch for very basic timings

-- | A simple stopwatch that records the runtime and number of calls for an
-- opeartion. Useful for the most trivial of tests. Stopwatches are not safe for
-- recursive calls!
type Stopwatch = IORef (Maybe (Integer   -- elapsed picoseconds
                              ,Integer   -- number of calls
                              ,Integer   -- shortest call
                              ,Integer   -- longest call
                              ,Maybe Integer)) -- Start time

-- | Make a new stopwatch
stopwatch = newIORef Nothing

-- | Start the stopwatch
startStopwatch :: Stopwatch -> IO ()
startStopwatch s = do
  t <- getCPUTime
  modifyIORef' s (\case
                     Nothing -> Just (0, 0, 10^100, 0, Nothing)
                     Just (e, n, mi, ma, Nothing) -> Just (e, n, mi, ma, Just t)
                     Just (e, n, mi, ma, Just _) -> error "Starting already started stopwatch!")

-- | Stop the stopwatch
stopStopwatch :: Stopwatch -> IO ()
stopStopwatch s = do
  t <- getCPUTime
  modifyIORef' s (\case
                     Nothing -> error "Stopping an uninitialized stopwatch"
                     Just (e, n, mi, ma, Nothing) -> error "Stopping a stopped stopwatch"
                     Just (e, n, mi, ma, Just t') ->
                       let x = t - t'
                       in Just (e + x, n + 1, min mi x, max ma x, Nothing))

-- | Call an IO operation with the stopwatch
withStopwatch :: Stopwatch -> IO b -> IO b
withStopwatch s f = do
  startStopwatch s
  r <- f
  stopStopwatch s
  pure r

-- | Human-readable values from the stopwatch. Returns the elapsed time, number
-- of iterations, min time and max time in seconds.
readStopwatch :: Stopwatch -> IO (Double, Integer, Double, Double)
readStopwatch s = do
  Just (e, n, mi, ma, _) <- readIORef s
  pure (fromIntegral e * 1e-12
       ,n
       ,fromIntegral mi * 1e-12
       ,fromIntegral ma * 1e-12)

-- | Print a stopwatch.
printStopwatch s = do
  (e, n, mi, ma) <- readStopwatch s
  putStrLn $ "Elapsed" <#> showSeconds e
    <#> "with" <#> showSeconds (e / fromIntegral n)
    <#> "/ iteration for" <#> show n
    <#> "iterations with min" <#> showSeconds mi
    <#> "and max" <#> showSeconds ma

-- | Convert a number of seconds to a string.  The string will consist
-- of four decimal places, followed by a short description of the time
-- units. From criterion
showSeconds :: Double -> String
showSeconds k
    | k < 0      = '-' : showSeconds (-k)
    | k >= 1     = k        `with` "s"
    | k >= 1e-3  = (k*1e3)  `with` "ms"
    | k >= 1e-6  = (k*1e6)  `with` "μs"
    | k >= 1e-9  = (k*1e9)  `with` "ns"
    | k >= 1e-12 = (k*1e12) `with` "ps"
    | k >= 1e-15 = (k*1e15) `with` "fs"
    | k >= 1e-18 = (k*1e18) `with` "as"
    | otherwise  = printf "%g s" k
     where with (t :: Double) (u :: String)
               | t >= 1e9  = printf "%.4g %s" t u
               | t >= 1e3  = printf "%.0f %s" t u
               | t >= 1e2  = printf "%.1f %s" t u
               | t >= 1e1  = printf "%.2f %s" t u
               | otherwise = printf "%.3f %s" t u

-- * File path and directory manipulation
infixr 5 </>

(</>) :: Text -> Text -> Text
d </> p = T.pack (T.unpack d P.</> T.unpack p)

dropExtension = T.pack . P.dropExtension . T.unpack
takeBaseName = T.pack . P.takeBaseName . T.unpack
takeDirectory = T.pack . P.takeDirectory . T.unpack
takeFileName = T.pack . P.takeFileName . T.unpack
doesFileExist = P.doesFileExist . T.unpack
doesDirectoryExist = P.doesDirectoryExist . T.unpack
show' x = T.pack (show x)
createDirectoryIfMissing d = P.createDirectoryIfMissing False (T.unpack d)
createDirectoryIfMissing' d = P.createDirectoryIfMissing True (T.unpack d)
listDirectory d = map T.pack <$> P.listDirectory (T.unpack d)

-- * Orphan AESON instances

instance FromJSON CChar where
    parseJSON = fmap CChar . parseJSON
instance ToJSON CChar where
    toJSON (CChar x) = toJSON x

instance FromJSON CSChar where
    parseJSON = fmap CSChar . parseJSON
instance ToJSON CSChar where
    toJSON (CSChar x) = toJSON x

instance FromJSON CUChar where
    parseJSON = fmap CUChar . parseJSON
instance ToJSON CUChar where
    toJSON (CUChar x) = toJSON x

instance FromJSON CShort where
    parseJSON = fmap CShort . parseJSON
instance ToJSON CShort where
    toJSON (CShort x) = toJSON x

instance FromJSON CUShort where
    parseJSON = fmap CUShort . parseJSON
instance ToJSON CUShort where
    toJSON (CUShort x) = toJSON x

instance FromJSON CInt where
    parseJSON = fmap CInt . parseJSON
instance ToJSON CInt where
    toJSON (CInt x) = toJSON x

instance FromJSON CUInt where
    parseJSON = fmap CUInt . parseJSON
instance ToJSON CUInt where
    toJSON (CUInt x) = toJSON x

instance FromJSON CLong where
    parseJSON = fmap CLong . parseJSON
instance ToJSON CLong where
    toJSON (CLong x) = toJSON x

instance FromJSON CULong where
    parseJSON = fmap CULong . parseJSON
instance ToJSON CULong where
    toJSON (CULong x) = toJSON x

instance FromJSON CPtrdiff where
    parseJSON = fmap CPtrdiff . parseJSON
instance ToJSON CPtrdiff where
    toJSON (CPtrdiff x) = toJSON x

instance FromJSON CSize where
    parseJSON = fmap CSize . parseJSON
instance ToJSON CSize where
    toJSON (CSize x) = toJSON x

instance FromJSON CWchar where
    parseJSON = fmap CWchar . parseJSON
instance ToJSON CWchar where
    toJSON (CWchar x) = toJSON x

instance FromJSON CSigAtomic where
    parseJSON = fmap CSigAtomic . parseJSON
instance ToJSON CSigAtomic where
    toJSON (CSigAtomic x) = toJSON x

instance FromJSON CLLong where
    parseJSON = fmap CLLong . parseJSON
instance ToJSON CLLong where
    toJSON (CLLong x) = toJSON x

instance FromJSON CULLong where
    parseJSON = fmap CULLong . parseJSON
instance ToJSON CULLong where
    toJSON (CULLong x) = toJSON x

instance FromJSON CBool where
    parseJSON = fmap CBool . parseJSON
instance ToJSON CBool where
    toJSON (CBool x) = toJSON x

instance FromJSON CIntPtr where
    parseJSON = fmap CIntPtr . parseJSON
instance ToJSON CIntPtr where
    toJSON (CIntPtr x) = toJSON x

instance FromJSON CUIntPtr where
    parseJSON = fmap CUIntPtr . parseJSON
instance ToJSON CUIntPtr where
    toJSON (CUIntPtr x) = toJSON x

instance FromJSON CIntMax where
    parseJSON = fmap CIntMax . parseJSON
instance ToJSON CIntMax where
    toJSON (CIntMax x) = toJSON x

instance FromJSON CUIntMax where
    parseJSON = fmap CUIntMax . parseJSON
instance ToJSON CUIntMax where
    toJSON (CUIntMax x) = toJSON x

instance FromJSON CClock where
    parseJSON = fmap CClock . parseJSON
instance ToJSON CClock where
    toJSON (CClock x) = toJSON x

instance FromJSON CUSeconds where
    parseJSON = fmap CUSeconds . parseJSON
instance ToJSON CUSeconds where
    toJSON (CUSeconds x) = toJSON x

instance FromJSON CSUSeconds where
    parseJSON = fmap CSUSeconds . parseJSON
instance ToJSON CSUSeconds where
    toJSON (CSUSeconds x) = toJSON x

instance FromJSON CFloat where
    parseJSON = fmap CFloat . parseJSON
instance ToJSON CFloat where
    toJSON (CFloat x) = toJSON x

instance FromJSON CDouble where
    parseJSON = fmap CDouble . parseJSON
instance ToJSON CDouble where
    toJSON (CDouble x) = toJSON x

-- * Leftovers, the miscellaneous bits

infixr 5 <#>
-- | A handy operator for concatenating two strings with a space between them.
-- TODO This is rarely used, maybe remove it.
(<#>) :: (Semigroup a, Data.String.IsString a) => a -> a -> a
a <#> b = a <> " " <> b

splitEvery :: Int -> [a] -> [[a]]
splitEvery _ [] = []
splitEvery n xs = as : splitEvery n bs
     where (as,bs) = splitAt n xs

-- | Like 'withForeignPtr' but takes a list of pointers.
withForeignPtrs :: [ForeignPtr a] -> ([Ptr a] -> IO b) -> IO b
withForeignPtrs fs io
  = do r <- io (map unsafeForeignPtrToPtr fs)
       mapM touchForeignPtr fs
       return r

instance Hashable CLong where
    hash (CLong a) = fromIntegral a
    hashWithSalt i (CLong j) = hashWithSalt i j
