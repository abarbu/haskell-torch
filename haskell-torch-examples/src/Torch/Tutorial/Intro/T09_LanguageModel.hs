{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, DeriveAnyClass, DeriveGeneric, FlexibleContexts, FlexibleInstances  #-}
{-# LANGUAGE FunctionalDependencies, GADTs, OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, QuasiQuotes         #-}
{-# LANGUAGE RankNTypes, RecordWildCards, ScopedTypeVariables, TemplateHaskell, TypeApplications, TypeFamilies, TypeFamilyDependencies #-}
{-# LANGUAGE TypeInType, TypeOperators, UndecidableInstances                                                                           #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=10 -fdefer-typed-holes #-}
{-# OPTIONS_GHC -fplugin-opt GHC.TypeLits.Normalise -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Torch.Tutorial.Intro.T09_LanguageModel where
import           Control.Monad
import           Data.Coerce
import           Data.Default
import qualified Data.HashTable.IO         as H
import           Data.IORef
import           Data.Maybe
import           Data.Singletons
import           Data.Singletons.TypeLits
import           Data.String.InterpolateIO
import           Data.Text                 (Text)
import qualified Data.Text                 as T
import qualified Data.Text.IO              as T
import qualified Data.Vector.Storable      as V
import           Foreign.C.Types
import qualified GHC.TypeLits              as TL
import           Pipes
import qualified Pipes                     as P
import           Torch
import           Torch.Tensor              as T

readCorpus :: Text -> Text -> Text -> Text
           -> IO (CLong
                ,H.CuckooHashTable Text CLong
                ,H.CuckooHashTable CLong Text
                ,Producer (DataSample Train () (Tensor TLong KCpu '[20,30]) (Tensor TLong KCpu '[20,30])) IO ())
readCorpus filename directory md5 dataUrl = do
  (Right _) <- downloadAndVerifyFile dataUrl filename directory md5
  txt <- T.readFile $ T.unpack $ directory </> filename
  word2index <- H.new
  index2word <- H.new
  vocabularySize <- newIORef 0
  mapM_ (\t ->
            mapM_ (\w -> do
                      b <- H.lookup word2index w
                      case b of
                        Nothing -> do
                          i <- readIORef vocabularySize
                          H.insert word2index w i
                          H.insert index2word i w
                          modifyIORef' vocabularySize (+1)
                        Just _ -> pure ())
            $ T.words t ++ ["<eos>"])
    $ T.lines txt
  -- I would really like to write this code, but the constraints don't quite
  -- work out, likely because GHC cannot simplify the mess that comes out. This
  -- should hopefully be alleviated by a new typechecker plugin.
  --
  -- This is -1 because we slice up the vector in a bit, dropping the last
  -- element from the training portion and the first from the label portion. So
  -- we end up with a dataset where you see a token and must predict the next
  -- one.
  -- stream <- withSomeSing ((fromIntegral $ length $ T.words txt) - 1)
  --   (\ x@(SNat :: SNat 929580) -> do
  --       full <- V.fromList <$> mapM (\w -> fromJust <$> H.lookup word2index w) (T.words txt)
  --       dataTensor   <- fromVector @TLong @'[929580] (V.slice 0 (V.length full - 1) full)
  --       labelTensor  <- fromVector @TLong @'[929580] (V.slice 1 (V.length full - 1) full)
  --       (dataParts, _)  <- chunk (dimension_ @0) (chunks_ @30) =<< toDevice dataTensor
  --       (labelParts, _) <- chunk (dimension_ @0) (chunks_ @30) =<< toDevice labelTensor
  --       pure $ zipWithM_ (\d l -> P.yield (DataSample () (pure d) (pure l))) dataParts labelParts)
  stream <- do
    full            <- V.fromList <$> mapM (fmap fromJust . H.lookup word2index) (concatMap (\w -> T.words w <> ["<eos>"]) $ T.lines txt)
    dataTensor      <- fromVector @TLong @'[929588] (V.slice 0 (V.length full - 1) full)
    labelTensor     <- fromVector @TLong @'[929588] (V.slice 1 (V.length full - 1) full)
    dataParts  :: Tensor TLong KCpu '[20, 46479] <- reshape @'[20, 46479] =<< narrow (dimension_ @0) (size_ @0) (size_ @929580) =<< toDevice dataTensor
    labelParts :: Tensor TLong KCpu '[20, 46479] <- reshape @'[20, 46479] =<< narrow (dimension_ @0) (size_ @0) (size_ @929580) =<< toDevice labelTensor
    (dataParts', _)  <- split (dimension_ @1) (size_ @30) dataParts
    (labelParts', _) <- split (dimension_ @1) (size_ @30) labelParts
    pure $ zipWithM_ (\d l -> P.yield (DataSample () (pure d) (pure l))) dataParts' labelParts'
  vocabularySize' <- readIORef vocabularySize
  pure (vocabularySize'
       ,word2index
       ,index2word
       ,stream)

forward :: forall batch seqLen. (KnownNat batch, KnownNat seqLen)
        => ((EmbeddingParam 'TFloat KCpu 10000 128
           ,LSTMParams 'TFloat KCpu 128 1024 1 'False 'True
           ,LinearParam 'TFloat KCpu 1024 10000)
          ,LSTMStateBatchFirst 'TFloat KCpu batch 'False 1 1024)
        -> DataPurpose
        -> Tensor TLong KCpu '[batch, seqLen]
        -> IO (Tensor TFloat KCpu '[batch TL.* seqLen, 10000]
             ,LSTMStateBatchFirst 'TFloat KCpu batch 'False 1 1024)
forward ((ep,w1,w2),state1) isTraining =
     embedding (nrEmbeddings_ @10000) (embeddingDimensions_ @128) Nothing Nothing False ep
  >=> lstmBatchFirst (inF_ @128) (hiddenF_ @1024) (nrLayers_ @1) (isBidirectional_ @False) 0 isTraining w1 state1
  >=> (\(out,hc) -> do
           r <- (  reshape @'[batch TL.* seqLen,1024]
               >=> linear (inF_ @1024) (outFeatures_ @10000) w2) out
           pure (r, hc))

ex = do
  let epochs = 2 :: Int
  let learningRate = 0.002
  --
  (vocabularySize
   ,word2index
   ,index2word
   ,trainStream) <- readCorpus "tutorial-09-penn-treebank-training-sentences.txt"
                       "/tmp" "f26c4b92c5fdc7b3f8c7cdcb991d8420"
                       "https://raw.githubusercontent.com/yunjey/pytorch-tutorial/master/tutorials/02-intermediate/language_model/data/train.txt"
  --
  net <- gradP
  -- You can use this to load the model exported from PyTorch, just change the
  -- file name at the end.
  --
  -- net <- readModelFromFileWithNames ("embed.weight" :: Text
  --                                  ,(["lstm.weight_ih_l0"]
  --                                   ,["lstm.bias_ih_l0"]
  --                                   ,["lstm.weight_hh_l0"]
  --                                   ,["lstm.bias_hh_l0"])
  --                                  ,("linear.weight", "linear.bias")) "/tmp/t09-0.pt"
  params <- toParameters net
  optimizer <- newIORef (adam (def { adamLearningRate = learningRate }) params)
  --
  let criterion y ypred = crossEntropyLoss y def def def ypred
  --
  withGrad
    $ mapM_ (\epoch -> do
            currentState <- newIORef =<< gradP
            forEachDataN
              (\d n -> do
                  state <- readIORef currentState
                  -- This is the truncated backprop through time
                  mapM detachAny =<< toTensors state
                  --
                  loss <- do
                    o <- dataObject d
                    l <- reshape =<< dataLabel d
                    (pred, nextState) <- forward (net,state) (dataPurpose d) o
                    writeIORef currentState nextState
                    criterion l pred
                  zeroGradients_ params
                  backward1 loss False False
                  clipGradNorm_ params 0.5 2
                  step_ optimizer
                  perplexity <- T.exp loss
                  when (n `rem` 100  == 0) $ putStrLn =<< [c|Epoch #{epoch+1}/#{epochs} perplexity #{perplexity} loss #{loss} |]
                  putStrLn =<< [c|Step1 #{epoch+1}/#{epochs} #{n+1} perplexity #{perplexity} loss #{loss} |]
                  pure ())
              trainStream)
    [0..epochs-1]
  --
  withoutGrad
    $ do
    currentState <- newIORef =<< gradP
    input <- withSomeSing (fromIntegral vocabularySize)
            (\ x@(SNat :: SNat sz) -> do
                prob <- ones @TFloat @KCpu @'[sz]
                unsqueeze (dimension_ @0) =<< multinomialVector (size_ @1) (replacement_ @True) prob)
    putStrLn "------------------------------ predictions ------------------------------\n"
    mapM_ (\n -> do
              state <- readIORef currentState
              (output :: Tensor TFloat KCpu '[1,10000], nextState) <- forward (net,state) Test input
              writeIORef currentState nextState
              prob <- T.exp output
              wordIndex <- fromScalar =<< squeeze =<< multinomialMatrix (size_ @1) (replacement_ @True) prob
              constant_ input wordIndex
              mword <- H.lookup index2word (coerce wordIndex)
              case mword of
                Nothing -> error "Impossible"
                Just word ->
                  if "<eos>" == word then
                    putStrLn "" else
                    putStr (T.unpack word ++ " "))
      [0..1000]
    putStrLn ""
