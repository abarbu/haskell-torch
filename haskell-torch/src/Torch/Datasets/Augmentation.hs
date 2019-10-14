{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, FlexibleContexts, FlexibleInstances, FunctionalDependencies, GADTs #-}
{-# LANGUAGE KindSignatures, MultiParamTypeClasses, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies, TypeInType, TypeOperators  #-}
{-# LANGUAGE UndecidableInstances                                                                                                     #-}
{-# OPTIONS_GHC -pgmP cc -optP -E -optP -undef -optP -std=c89 #-}

-- | Turn data transformations into dataset augmentations. These operations modify
-- streams of data to generate new samples. They can randomly choose which
-- operations to apply and in what sequence.

module Torch.Datasets.Augmentation where
import           Pipes
import           System.Random         (randomRIO)
import           Torch.Datasets.Common
import           Torch.Misc

-- * Lifting per item operations into whole dataset operations

-- | The most flexible transform, change anything you want about a sample.
transformStream :: (DataSample dataPurpose properties object label -> IO (DataSample dataPurpose properties' object' label'))
                -> DataStream dataPurpose properties object label
                -> DataStream dataPurpose properties' object' label'
transformStream f x = for x (\s -> do
                                s' <- liftIO $ f s
                                yield s')

-- | Only transform the objects in the dataset leaving other properties
-- unchanged.
transformObjectStream :: (object -> IO object')
                      -> DataStream dataPurpose properties object label
                      -> DataStream dataPurpose properties object' label
transformObjectStream f x = for x (\(DataSample prop obj lab) -> yield (DataSample prop (f =<< obj) lab))

-- * Per-data-sample operations

-- | Transform just the object from a data sample.
transformSampleObject :: (object -> IO object')
                      -> DataSample dataPurpose properties object label
                      -> IO (DataSample dataPurpose properties object' label)
  -- NB: Technically due to how this is implemented IO is not needed here, it
  -- folds into the object which is an IO operation. Since this trick doesn't
  -- work for the other variants of this operation and relies on something
  -- rather low-level about how DataSample is defined, we introduce this
  -- needless IO op to keep the API uniform and flexible to changes in
  -- DataSample.
transformSampleObject f (DataSample prop obj lab) =
  pure $ DataSample prop (f =<< obj) lab

-- | Transform all of the properties of the sample individually.
transformSample :: (properties -> IO properties')
                -> (object -> IO object')
                -> (label -> IO label')
                -> DataSample dataPurpose properties object label
                -> IO (DataSample dataPurpose properties' object' label')
transformSample fp fo fl (DataSample prop obj lab) = do
  p <- liftIO $ fp prop
  pure $ DataSample p (fo =<< obj) (fl =<< lab)

-- | Transform all of the properties of the sample as a whole.
transformSample' :: (properties -> object -> label -> IO (properties', object', label'))
               -> DataSample dataPurpose properties object label
               -> IO (DataSample dataPurpose properties' object' label')
transformSample' f (DataSample prop obj lab) = do
  o' <- liftIO obj
  l' <- liftIO lab
  (p, o, l) <- liftIO $ f prop o' l'
  pure $ DataSample p (pure o) (pure l)

-- * Randomly applying transformations to get more variety in your data

oneRandomTransform :: [DataSample dataPurpose properties object label -> DataSample dataPurpose properties object label]
                   -> DataStream dataPurpose properties object label
                   -> DataStream dataPurpose properties object label
oneRandomTransform l x = transformStream (oneRandomTransformPerSample l) x

-- | Apply one of the transformations randomly to each item of the data stream.
oneRandomTransformPerSample :: [DataSample dataPurpose properties object label -> DataSample dataPurpose properties object label]
                            -> DataSample dataPurpose properties object label
                            -> IO (DataSample dataPurpose properties object label)
oneRandomTransformPerSample l d = do
  e <- randomRIO (0, length l - 1)
  pure $ (l !! e) d

transformInRandomOrder :: [DataSample dataPurpose properties object label -> DataSample dataPurpose properties object label]
                       -> DataStream dataPurpose properties object label
                       -> DataStream dataPurpose properties object label
transformInRandomOrder l x = transformStream (transformInRandomOrderPerSample l) x

-- | Apply all of the transformations in a random order to each item of the data stream.
transformInRandomOrderPerSample :: [DataSample dataPurpose properties object label -> DataSample dataPurpose properties object label]
                                -> DataSample dataPurpose properties object label
                                -> IO (DataSample dataPurpose properties object label)
transformInRandomOrderPerSample l d = do
  l' <- shuffleList l
  pure $ foldl1 (.) l' d

