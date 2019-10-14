{-# LANGUAGE AllowAmbiguousTypes, ConstraintKinds, DataKinds, FlexibleContexts, FlexibleInstances, FunctionalDependencies, GADTs       #-}
{-# LANGUAGE KindSignatures, MultiParamTypeClasses, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, QuasiQuotes #-}
{-# LANGUAGE RankNTypes, ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies, TypeInType      #-}
{-# LANGUAGE TypeOperators, UndecidableInstances                                                                                       #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

-- | You'll notice that this module calls performMinorGC in various
-- places. There's no way to tell the Haskell GC about external memory so huge
-- amounts of memory can build up before any finalizers get run :(

module Torch.Datasets.Common where
import           Control.Monad.Except
import           Control.Monad.Loops
import           Data.Foldable
import           Data.IORef
import           Data.Sequence        (Seq)
import qualified Data.Sequence        as S
import           Data.Singletons
import           Data.Text            (Text)
import qualified Data.Text            as T
import qualified Data.Vector          as V'
import           Foreign.ForeignPtr
import qualified Foreign.Matio        as M
import           Foreign.Matio.Types  (CMat)
import           Foreign.Storable
import           Pipes
import qualified Pipes                as P
import qualified Pipes.Prelude        as P
import qualified Shelly               as S
import qualified System.Directory     as P
import           System.Mem
import           Torch.Misc
import           Torch.Tensor
import           Torch.Types

type CanFail = ExceptT Text IO

canFail :: ExceptT e m a -> m (Either e a)
canFail = runExceptT

type Path = Text

-- | A datasets is a wrapper around one stream of data. This means that
-- conceptually a training set and test set are disjoint datasets! Certain
-- operations behave differently at training and test time, like data
-- augmentation and batch normalization. The extra type safety here makes sure
-- that there's no confusion and no room for mistakes.
data Dataset metadata (dataPurpose :: DataPurpose) index object label =
  Dataset { checkDataset      :: IO (Either Text ())
          , fetchDataset      :: IO (Either Text (DataStream dataPurpose index object label))
          , forceFetchDataset :: IO (Either Text (DataStream dataPurpose index object label))
          , accessDataset     :: IO (Either Text (DataStream dataPurpose index object label))
          , metadataDataset   :: IO (Either Text metadata)
          }

-- | One of the most standard ways to view a dataset is as a pair of fixed
-- training and test sets.
type TrainTestDataset metadata index object label =
  (Dataset metadata Train index object label,
   Dataset metadata Test index object label)

-- | A data sample from a dataset. It is tagged with its purpose, either training or
-- test. This is so that some APIs can prevent you from misusing data, like say
-- training on the test data. It has properties which are accessible for any data
-- point and an associated object/label which requires IO to access.
-- Listing data and its properties is cheap but reading the data point or the label
-- might be expensive so we keep these behind IO ops that you run when you need the
-- data itself.
--
-- Functions take optional arguments which provide storage for the output. When
-- the argument is given the results must be written to it! TODO How can I enforce this?
data DataSample (dataPurpose :: DataPurpose) properties object label =
  DataSample { dataProperties :: properties
             , dataObject     :: IO object
             , dataLabel      :: IO label }

dataPurpose :: forall (datap :: DataPurpose) p o l. SingI datap => DataSample datap p o l -> DataPurpose
dataPurpose _ = demote @datap

-- | This is an inefficient way of satisfying the requirement that datasets must
-- write to their input if provided.
setTensorIfGiven_ :: IO (Tensor ty ki sz) -> Maybe (Tensor ty ki sz) -> IO (Tensor ty ki sz)
setTensorIfGiven_ f x = do
  t <- f
  case x of
    Nothing -> pure t
    Just x' -> set_ x' t >> pure x'

-- | This is a more efficient way of streaming data, write to the given tensor,
-- we'll take care of dealing with providing a fresh one or using the input.
useTensorIfGiven_ f x =
  case x of
    Nothing -> empty >>= f
    Just t  -> f t

-- | A stream of data. This is like an entire training or test set. Datasets
-- with unbounded size can be conveniently represented because lists are lazy.
-- A dataset contains a single stream of data.
type DataStream dp index object label = Producer (DataSample dp index object label) IO ()

-- | Create a stream of data samples from an action
mapObjects :: (o -> IO o') -> Pipe (DataSample dp p o l) (DataSample dp p o' l) IO ()
mapObjects f = forever $ do
  (DataSample p o l) <- await
  yield $ DataSample p (o >>= f) l

-- | Create a stream of data labels from an action
mapLabels :: (l -> IO l') -> Pipe (DataSample dp p o l) (DataSample dp p o l') IO ()
mapLabels f = forever $ do
  (DataSample p o l) <- await
  yield $ DataSample p o (l >>= f)

-- * Manipulate datasets

applyTrain :: (DataStream Train index object label -> DataStream Train index' object' label')
           -> Dataset metadata Train index  object  label
           -> Dataset metadata Train index' object' label'
applyTrain f d = Dataset { checkDataset = checkDataset d
                         , fetchDataset = (f <$>) <$> fetchDataset d
                         , forceFetchDataset = (f <$>) <$> forceFetchDataset d
                         , accessDataset = (f <$>) <$> accessDataset d
                         , metadataDataset = metadataDataset d }

applyTest :: (DataStream Test index object label -> DataStream Test index' object' label')
          -> Dataset metadata Test index  object  label
          -> Dataset metadata Test index' object' label'
applyTest f d = Dataset { checkDataset = checkDataset d
                        , fetchDataset = (f <$>) <$> fetchDataset d
                        , forceFetchDataset = (f <$>) <$> forceFetchDataset d
                        , accessDataset = (f <$>) <$> accessDataset d
                        , metadataDataset = metadataDataset d }

augmentTrainData :: (object -> IO object')
                 -> Dataset metadata Train index  object  label
                 -> Dataset metadata Train index  object' label
augmentTrainData f = applyTrain (\d ->
                                    P.for d (\s ->
                                                yield $ DataSample { dataProperties = dataProperties s
                                                                   , dataObject     = f =<< dataObject s
                                                                   , dataLabel      = dataLabel s }))

augmentTestData :: (object -> IO object')
                 -> Dataset metadata Test index  object  label
                 -> Dataset metadata Test index  object' label
augmentTestData f = applyTest (\d ->
                                  P.for d (\s ->
                                              yield $ DataSample { dataProperties = dataProperties s
                                                                 , dataObject     = f =<< dataObject s
                                                                 , dataLabel      = dataLabel s }))

augmentTrain :: (index -> index') -> (object -> IO object') -> (label -> IO label')
             -> Dataset metadata Train index  object  label
             -> Dataset metadata Train index' object' label'
augmentTrain fi fo fl =
  applyTrain (\d ->
                 P.for d (\s ->
                             yield $ DataSample { dataProperties = fi $ dataProperties s
                                                , dataObject     = fo =<< dataObject s
                                                , dataLabel      = fl =<< dataLabel s }))

augmentTest :: (index -> index') -> (object -> IO object') -> (label -> IO label')
             -> Dataset metadata Test index  object  label
             -> Dataset metadata Test index' object' label'
augmentTest fi fo fl =
  applyTest (\d ->
                 P.for d (\s ->
                             yield $ DataSample { dataProperties = fi $ dataProperties s
                                                , dataObject     = fo =<< dataObject s
                                                , dataLabel      = fl =<< dataLabel s }))

-- * Standard operations on datasets

forEachData :: MonadIO m => (a -> IO b) -> Producer a m r -> m r
forEachData f stream = runEffect $ for stream (\x -> liftIO (f x >> performMinorGC >> pure ()))

forEachDataN :: MonadIO m => (a -> Int -> IO b) -> Producer a m r -> m r
forEachDataN f stream = do
  n <- liftIO $ newIORef (0 :: Int)
  runEffect $ for stream (\x -> liftIO $ do
                             f x =<< readIORef n
                             modifyIORef' n (+1)
                             performMinorGC
                             pure ())

-- | Each of the arguments are called with the step number & epoch. TODO
forEachDataUntil :: MonadIO m
                 => (Int -> Int -> IO Bool) -- ^ are we done?
                 -> (Int -> Int -> Producer a m r -> IO (Producer a m r)) -- ^ what to do when starting a new epoch
                 -> (Int -> Int -> a -> IO b) -- ^ per sample op
                 -> Producer a m r -- ^ data stream
                 -> m (Int, Int)
forEachDataUntil isDone newEpoch perSample initialStream = do
  step  <- liftIO $ newIORef (0 :: Int)
  epoch <- liftIO $ newIORef (0 :: Int)
  stream' <- liftIO $ newEpoch 0 0 initialStream
  let go stream = do
        n <- next stream
        s <- liftIO $ readIORef step
        e <- liftIO $ readIORef epoch
        b <- liftIO $ isDone s e
        if not b then
          case n of
            Left r -> do
              stream' <- liftIO $ newEpoch s e initialStream
              liftIO $ modifyIORef epoch (+ 1)
              go stream'
            Right (value, stream') -> do
              liftIO $ perSample s e value
              liftIO $ modifyIORef step (+ 1)
              go stream'
          else
          pure ()
  go stream'
  s <- liftIO $ readIORef step
  e <- liftIO $ readIORef epoch
  pure (s,e)

foldData :: MonadIO m => (b -> a -> m b) -> b -> Producer a m () -> m b
foldData f i stream = P.foldM (\a b -> liftIO performMinorGC >> f a b)
                      (pure i)
                      pure
                      stream

lossForEachData :: (TensorConstraints ty ki sz)
                => (t -> IO (Tensor ty ki sz))
                -> Producer t IO ()
                -> IO (Tensor ty ki sz)
lossForEachData loss stream =
  P.foldM (\l d -> do
              performMinorGC
              l' <- liftIO $ loss d
              l `add` l')
          zeros
  pure stream

lossSum :: (SingI ty, SingI ki
          , Num (TensorTyToHs ty), Storable (TensorTyToHs ty)
          , Num (TensorTyToHsC ty), Storable (TensorTyToHsC ty))
        => V'.Vector (Scalar ty ki)
        -> IO (Scalar ty ki)
lossSum v = do
  z <- zeros
  foldM add z v

-- | This works like the dataset shuffler in tensorflow. You have a horizon out
-- to which you shuffle data. This is because data streams might be infinite in
-- length and a shuffle of the entire data isn't possible there.
--
-- If you're gathering data from an intermittent source and your horizon is long
-- this will block until enough data is available!
shuffle :: Int -> Producer a IO r -> Producer a IO r
shuffle horizon p = go p S.empty
  where go p s = do
          n <- liftIO $ next p
          case n of
            Left r -> iterateUntilM S.null randomSeqYield s >> pure r
            Right (a, p') -> do
              let s' = a S.<| s
              if S.length s' < horizon then
                   go p' s' else
                   randomSeqYield s' >>= go p'

randomSeqYield :: Seq a -> Pipes.Proxy x' x () a IO (Seq a)
randomSeqYield s = do
  r <- liftIO $ randomElement s
  case r of
    (Just (e, s')) -> do
      yield e
      pure s'
    Nothing -> lift mzero

batch :: Int -> Bool -> Producer a IO r -> Producer (V'.Vector a) IO r
batch batchSize returnPartial p = go p S.empty
  where go p s = do
          n <- liftIO $ next p
          case n of
            Left r -> do
              when returnPartial $ yield (V'.fromList $ toList s)
              pure r
            Right (a, p') -> do
              let s' = s S.|> a
              if S.length s' >= batchSize then
                   yield (V'.fromList $ toList s') >> go p' S.empty else
                   go p' s'

batchFn :: Int -> Bool -> Producer a IO r -> (V'.Vector a -> IO b) -> Producer b IO r
batchFn batchSize returnPartial p fn = go p S.empty
  where go p s = do
          n <- liftIO $ next p
          case n of
            Left r -> do
              when returnPartial $ do
                y <- liftIO $ fn $ V'.fromList $ toList s
                yield y
              pure r
            Right (a, p') -> do
              let s' = s S.|> a
              if S.length s' >= batchSize then
                do
                  y <- liftIO $ fn $ V'.fromList $ toList s'
                  yield y
                  go p' S.empty else
                go p' s'

batchTensors :: forall batchSize ty ki sz ty' ki' sz' dp p r.
               (SingI batchSize
               ,Num (TensorTyToHs ty), Storable (TensorTyToHs ty)
               ,Num (TensorTyToHs ty'), Storable (TensorTyToHs ty')
               ,Num (TensorTyToHsC ty), Storable (TensorTyToHsC ty)
               ,Num (TensorTyToHsC ty'), Storable (TensorTyToHsC ty')
               ,SingI (InsertIndex sz 0 batchSize)
               ,SingI ty, SingI ki, SingI sz, SingI sz'
               ,SingI (InsertIndex sz' 0 batchSize)
               ,SingI ty', SingI ki')
             => BatchSize batchSize
             -> Producer (DataSample dp p (Tensor ty ki sz)
                                          (Tensor ty' ki' sz'))
                         IO r
             -> Producer (DataSample dp (V'.Vector p)
                          (Tensor ty ki (InsertIndex sz 0 batchSize))
                          (Tensor ty' ki' (InsertIndex sz' 0 batchSize)))
                         IO r
batchTensors BatchSize p = batchFn (demoteN @batchSize) False p
  (\v -> pure $ DataSample (V'.map dataProperties v)
                           (do
                               v <- (V'.mapM dataObject v)
                               s <- stack (groups_ @batchSize) (dimension_ @0) v
                               case s of
                                 Nothing -> error "Oddly-sized batch -- this is a bug"
                                 Just x  -> pure x)
                           (do
                               v <- (V'.mapM dataLabel v)
                               s <- stack (groups_ @batchSize) (dimension_ @0) v
                               case s of
                                 Nothing -> error "Oddly-sized batch -- this is a bug"
                                 Just x  -> pure x))

-- TODO preload
-- preload :: Int -> Producer (DataSample dp p o l) IO r -> Producer (DataSample dp p o l) IO r
-- preload n = undefined

-- TODO preloadBatch
-- preloadBatches :: Int -> Producer (Vector (DataSample dp p o l)) IO r -> Producer (Vector (DataSample dp p o l)) IO r
-- preloadBatches n = undefined

{-
 TODO filter (pick subset that matches predicate)
 sample with replacement
 balanace with respect to some label

 most of this works with pipes aside from sampling!
 lets just do a list for now
-}

onError :: MonadError e m => m a -> m a -> m a
onError f g = f `catchError` const g

retryAfter :: MonadError e m => m a -> m a -> m a
retryAfter f g = f `catchError` const (g >> f)

checkMD5 :: Path -> Text -> CanFail ()
checkMD5 path md5 = do
  e <- liftIO $ doesFileExist path
  if e then
    do
      out <- S.shelly $ S.run "md5sum" [path]
      case T.splitOn " " out of
        (s:_) -> if s == md5 then
                  pure () else
                  throwError $ "MD5 check failed, got " <#> path <#> s <#> "but expected" <#> md5
        _ -> throwError $ "Bad result from md5" <#> out
    else throwError $ "Missing file" <#> path

downloadUrl :: Text -> Path -> CanFail ()
downloadUrl url dest = do
  c <- S.shelly $ S.verbosely $ do
    S.run_ "curl" ["--progress-bar", "-L", url, "--output", dest]
    S.lastExitCode
  unless (c == 0) $ throwError $ "Download with curl failed" <#> url <#> dest

extractTar :: Path -> Path -> CanFail ()
extractTar path dest = do
  e <- liftIO $ doesFileExist path
  unless e $ throwError $ "Can't extract, tar file doesn't exist" <#> path
  c <- S.shelly $ S.verbosely $ do
    S.run_ "tar" ["xvf", path, "-C", dest]
    S.lastExitCode
  unless (c == 0) $ throwError $ "Extracting tar file failed" <#> path

extractGzip :: Path -> CanFail ()
extractGzip path = do
  e <- liftIO $ doesFileExist path
  unless e $ throwError $ "Can't extract, gzip file doesn't exist" <#> path
  c <- S.shelly $ S.verbosely $ do
    S.run_ "gunzip" [path]
    S.lastExitCode
  unless (c == 0) $ throwError $ "gunzip failed" <#> path

-- | This is a high-level wrapper to get a file, or fail gently
downloadAndVerifyFile :: Text -> Text -> Text -> Text -> IO (Either Text ())
downloadAndVerifyFile url filename dir md5 = canFail $ do
  liftIO $ createDirectoryIfMissing' dir
  checkMD5 (dir </> filename) md5 `retryAfter` downloadUrl url (dir </> filename)

-- | This is the end point for downloading a model or other data that must be
-- cached. The data is stored in the user's XDG cache directory so it will
-- persist after reboots.
getCachedFileOrDownload :: Text -> Text -> Text -> Text -> IO Text
getCachedFileOrDownload url md5 filename subtype = do
  location <- T.pack <$> P.getXdgDirectory P.XdgCache ("haskell-torch/" <> T.unpack subtype)
  let path = location </> filename
  e <- doesFileExist path
  if e then
    pure $ location </> filename else do
    r <- downloadAndVerifyFile url filename location md5
    case r of
      Left err -> do
        f <- canFail (checkMD5 path md5)
        case f of
          -- Remove the file if the md5sum was bad
          Left e -> do
            P.removeFile $ T.unpack path
          _ -> pure ()
        error $ T.unpack err
      Right _ -> pure $ location </> filename

-- * Utility functions on pipes

pipeMatfilesReadOnly :: [Text] -> Producer (ForeignPtr CMat) IO ()
pipeMatfilesReadOnly fs = mapM_ (\f -> do
                                    liftIO $ print $ "Reading " ++ show f
                                    mf <- liftIO $ M.openReadOnly f
                                    yield mf) fs

pipeListDirectory :: Text -> Producer Text IO ()
pipeListDirectory path = do
  l <- liftIO $ listDirectory path
  mapM_ (yield . (path </>)) l
