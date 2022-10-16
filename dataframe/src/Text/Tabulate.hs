{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE DeriveGeneric #-} -- Remove this
{-# LANGUAGE DeriveDataTypeable #-} -- Remove this
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ConstrainedClassMethods #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-#LANGUAGE ScopedTypeVariables #-}


-- | Module implements the default methods for Tabulate
-- All examples listed in the document need the following language pragmas
-- and following modules imported
--
-- @
-- {#- LANGUAGE MultiParamTypeClasses}
-- {#- LANGUAGE DeriveGeneric}
-- {#- LANGUAGE DeriveDataTypeable}
--
-- import qualified GHC.Generics as G
-- import Data.Data
-- @
--

module Text.Tabulate where

import Data.Maybe
import Data.Data
import Data.Tree
import Data.Typeable
import Data.Generics.Aliases
import GHC.Generics as G
import GHC.Show
import qualified Data.Map as Map
import qualified Text.PrettyPrint.Boxes as B
import qualified Data.List as List
import qualified Data.List as L
import Text.Printf
import qualified Data.Vector as V

-- | Future change to support providing custom formatting functions
data TablizeValueFormat = T {floatValueFormat::Maybe (Float -> String),
                             stringValueFormat::Maybe (String -> String),
                             integerValueFormat::Maybe (Integer -> String),
                             intValueFormat::Maybe (Int -> String),
                             doubleValueFormat::Maybe (Double -> String)}

-- | Default TabulateValueFormat
getDefaultTabulateValueFormat = T {floatValueFormat=Nothing,
                                   stringValueFormat=Nothing,
                                   integerValueFormat=Nothing,
                                   intValueFormat=Nothing,
                                   doubleValueFormat=Nothing}

data Tag = Constr | Fields | Values deriving (Show)

class GRecordMeta f where
  toTree:: f a -> [Tree String]

instance GRecordMeta U1 where
    toTree U1 = []

instance (GRecordMeta (a), GRecordMeta (b)) => GRecordMeta (a :*: b) where
  toTree (x :*: y) = (toTree x)  ++ (toTree y)

instance (GRecordMeta (a), GRecordMeta (b)) => GRecordMeta (a :+: b) where
  toTree x = toTree x

instance (GRecordMeta a, Selector s) => GRecordMeta (M1 S s a) where
  toTree a = [Node (selName a) $ toTree (unM1 a)] where

instance (GRecordMeta a, Constructor c) => GRecordMeta (M1 C c a) where
  -- we don't want to build node for constructor
  --toTree a = [Node (conName a) $ toTree (unM1 a)]
  toTree a = toTree (unM1 a)

instance (GRecordMeta a) => GRecordMeta (M1 D c a) where
  toTree (M1 x) = toTree x

instance (CellValueFormatter a, Data a, RecordMeta a) => GRecordMeta (K1 i a) where
  --toTree x = [Node (show (unK1 x)) (toTree' $ unK1 x)]
  toTree x = toTree' $ unK1 x

-- | Use this flag to expand a Record Type as a table when
-- nested inside another record.
data ExpandWhenNested

-- | Use this flag to not expand a Record type as a table when
-- nested inside another record. The 'Show' instance of the nested record
-- is used by default without expanding. This means that the fields of the
-- nested record are not displayed as separate headers.
data DoNotExpandWhenNested

-- | Class instance that needs to be instantiated for each
-- record that needs to be printed using printTable
--
-- @
--
-- data Stock = Stock {price:: Double, name:: String} derive (Show, G.Generic, Data)
-- instance Tabulate S 'ExpandWhenNested'
-- @
--
-- If 'S' is embedded inside another `Record` type and should be
-- displayed in regular Record Syntax, then
--
-- @
--
-- instance Tabulate S 'DoNotExpandWhenNested'
-- @
--
class Tabulate a flag | a->flag where {}

--instance TypeCast flag HFalse => Tabulate a flag
instance {-# OVERLAPPABLE #-} (flag ~ DoNotExpandWhenNested) => Tabulate a flag

class RecordMeta a where
  toTree':: a -> [Tree String]

instance (Tabulate a flag, RecordMeta' flag a) => RecordMeta a where
  toTree' = toTree'' (undefined::proxy flag)

class RecordMeta' flag a where
  toTree'':: proxy flag -> a -> [Tree String]

instance (G.Generic a, GRecordMeta (Rep a)) => RecordMeta' ExpandWhenNested a where
  toTree'' _ a = toTree (G.from a)

instance (CellValueFormatter a) => RecordMeta' DoNotExpandWhenNested a where
  toTree'' _ a = [Node (ppFormatter a) []]


-- |  Class that implements formatting using printf.
--    Default instances for String, Char, Int, Integer, Double and Float
--    are provided. For types that are not an instance of this class
--    `show` is used.
class CellValueFormatter a where

  -- Function that can be implemented by each instance
  ppFormatter :: a -> String

  -- Future support for this signature will be added
  ppFormatterWithStyle :: TablizeValueFormat -> a -> String

  -- Default instance of function for types that do
  -- do not have their own instance
  default ppFormatter :: (Show a) => a -> String
  ppFormatter x =  show x

  -- Future support.
  default ppFormatterWithStyle :: (Show a) => TablizeValueFormat ->  a -> String
  ppFormatterWithStyle _ x =  "default_" ++ show x


instance CellValueFormatter Integer where
  ppFormatter x = printf "%d" x

  ppFormatterWithStyle style x = case integerValueFormat style of
    Just f -> f x
    Nothing -> ppFormatter x

instance CellValueFormatter Int where
  ppFormatter x = printf "%d" x

  ppFormatterWithStyle style x = case intValueFormat style of
    Just f -> f x
    Nothing -> ppFormatter x


instance CellValueFormatter Float where
  ppFormatter x = printf "%14.7g" x

  ppFormatterWithStyle style x = case floatValueFormat style of
    Just f -> f x
    Nothing -> ppFormatter x

instance CellValueFormatter String where
  ppFormatter x = printf "%s" x

  ppFormatterWithStyle style x = case stringValueFormat style of
    Just f -> f x
    Nothing -> ppFormatter x


instance CellValueFormatter Double where
  ppFormatter x = printf "%14.7g" x

  ppFormatterWithStyle style x = case doubleValueFormat style of
    Just f -> f x
    Nothing -> ppFormatter x

instance CellValueFormatter Bool

instance (Show a, CellValueFormatter a) => CellValueFormatter (Maybe a)


gen_renderTableWithFlds :: [DisplayFld t] -> [t] -> B.Box
gen_renderTableWithFlds flds recs = results where
  col_wise_values = fmap (\(DFld f) -> fmap (ppFormatter .f) recs) flds
  vertical_boxes = fmap (B.vsep 0 B.top) $ fmap (fmap B.text) col_wise_values
  results = B.hsep 5 B.top vertical_boxes


class Boxable b where
  -- toBox :: (Data a, G.Generic a, GRecordMeta(Rep a)) => b a ->  [[B.Box]]
  -- toBoxWithStyle :: (Data a, G.Generic a, GTabulate(Rep a)) => TablizeValueFormat -> b a ->  [[B.Box]]

  -- | Used to print a container of Records in a tabular format.
  --
  -- @
  --
  -- data Stock = Stock {price:: Double, ticker:: String} deriving (Show, Data, G.Generic)
  -- instance Tabulate Stock DoNotExpandWhenNested
  -- -- this can be a Vector or Map
  -- let s =  [Stock 10.0 "yahoo", Stock 12.0 "goog", Stock 10.0 "amz"]
  -- T.printTable s
  -- @
  --
  -- Nested records can also be printed in tabular format
  --
  -- @
  --
  -- data FxCode = USD | EUR deriving (Show, Data, G.Generic)
  -- instance 'CellValueFormatter' FxCode
  --
  -- data Price = Price {px:: Double, fxCode:: FxCode} deriving (Show, Data, G.Generic)
  -- instance 'Tabulate' Price 'ExpandWhenNested'
  -- -- since Price will be nested, it also needs an instance of
  -- -- CellValueFormatter
  -- instance CellValueFormatter Price
  --
  -- data Stock = Stock {ticker:: String, price:: Price} deriving (Show, Data, G.Generic)
  -- instance Tabulate Stock DoNotExpandWhenNested
  --
  -- -- this can be a Vector or Map
  -- let s =  [Stock "yahoo" (Price 10.0 USD), Stock "ikea" (Price 11.0 EUR)]
  -- printTable s
  -- @
  --
  printTable :: (G.Generic a, GRecordMeta (Rep a)) => b a -> IO ()
  --printTableWithStyle :: (Data a, G.Generic a, GTabulate(Rep a)) => TablizeValueFormat -> b a -> IO ()

  -- | Similar to 'printTable' but rather than return IO (), returns a
  -- 'Box' object that can be printed later on, using 'printBox'
  renderTable :: (G.Generic a, GRecordMeta (Rep a)) => b a -> B.Box

  -- | Used for printing selected fields from Record types
  -- This is useful when Records have a large number of fields
  -- and only few fields need to be introspected at any time.
  --
  -- Using the example provided under 'printTables',
  --
  -- @
  -- 'printTableWithFlds' [DFld (px . price), DFld ticker] s
  --
  -- @
  printTableWithFlds :: [DisplayFld t] -> b t -> IO ()

  -- | Same as printTableWithFlds but returns a `Box` object, rather than
  -- returning an `IO ()`.
  renderTableWithFlds :: [DisplayFld t] -> b t -> B.Box

-- | Instance methods to render or print a list of records in a tabular format.
instance Boxable [] where
  -- | Used to print a list of Records in a tabular format.
  -- @
  --
  -- data Stock = Stock {price:: Double, ticker:: String}
  -- instance Tabulate S DoNotExpandWhenNested
  -- let s =  [Stock 10.0 "yahoo", Stock 12.0 "goog", Stock 10.0 "amz"]
  -- T.printTable s
  --
  -- @
  printTable m = B.printBox $ ppRecords m

  renderTable m = ppRecords m

  -- | Print a "List" of records as a table with just the given fields.
  -- Called by "printTableWithFlds".
  printTableWithFlds flds recs = B.printBox $ renderTableWithFlds flds recs
  renderTableWithFlds = gen_renderTableWithFlds


instance Boxable V.Vector where
  -- | Prints a "Vector" as a table. Called by "printTable".
  -- | Need not be called directly
  printTable m = B.printBox $ renderTable m  --TODO: switch this to Vector
  renderTable m = ppRecords $ V.toList m

  -- | Print a "Vector" of records as a table with the selected fields.
  -- Called by "printTableWithFlds".
  printTableWithFlds flds recs = B.printBox $ renderTableWithFlds flds $ V.toList recs
  renderTableWithFlds flds recs = gen_renderTableWithFlds flds $ V.toList recs


instance (CellValueFormatter k) => Boxable (Map.Map k) where

  -- | Prints a "Map" as a table. Called by "ppTable"
  -- | Need not be called directly
  printTable m = B.printBox $ renderTable m
  renderTable m = ppRecordsWithIndex m

  -- | Prints a "Map" as a table with the selected fields. Called by "printTable"
  -- | Need not be called directly
  printTableWithFlds flds recs = B.printBox $ renderTableWithFlds flds recs

  renderTableWithFlds flds recs = results where
    data_cols = renderTableWithFlds flds $ Map.elems recs
    index_cols = B.vsep 0 B.top $ fmap (B.text . ppFormatter) $ Map.keys recs
    vertical_cols = B.hsep 5 B.top [index_cols, data_cols]
    results = vertical_cols

-- Pretty Print the reords as a table. Handles both records inside
-- Lists and Vectors
ppRecords :: (GRecordMeta (Rep a), G.Generic a) => [a] -> B.Box
ppRecords recs = result where
  result = B.hsep 5 B.top $ createHeaderDataBoxes recs

-- Pretty Print the records as a table. Handles records contained in a Map.
-- Functions also prints the keys as the index of the table.
ppRecordsWithIndex :: (CellValueFormatter k, GRecordMeta (Rep a), G.Generic a) => (Map.Map k a) -> B.Box
ppRecordsWithIndex recs = result where
  data_boxes = createHeaderDataBoxes $ Map.elems recs
  index_box = createIndexBoxes recs
  result = B.hsep 5 B.top $ index_box:data_boxes


-- What follows are helper functions to build the B.Box structure to print as table.

-- Internal helper functions for building the Tree.

-- Build the list of paths from the root to every leaf.
constructPath :: Tree a -> [[a]]
constructPath (Node r []) = [[r]]
constructPath (Node r f) = [r:x | x <- (L.concatMap constructPath f)]

-- Fill paths with a "-" so that all paths have the
-- same length.
fillPath paths = stripped_paths where
  depth = L.maximum $ L.map L.length paths
  diff = L.map (\p -> depth - (L.length p)) paths
  new_paths = L.map (\(p,d) ->  p ++ L.replicate d "-") $ L.zip paths diff
  stripped_paths = [xs | x:xs <- new_paths]

-- Count the number of fields in the passed structure.
-- The no of leaves is the sum of all fields across all nested
-- records in the passed structure.
countLeaves :: Tree a -> Tree (Int, a)
countLeaves (Node r f) = case f of
  [] -> Node (1, r) []
  x -> countLeaves' x where
    countLeaves' x  = let
      count_leaves = fmap countLeaves x
      level_count = Prelude.foldr (\(Node (c, a) _) b -> c + b) 0 count_leaves
      in
      Node (level_count, r) count_leaves

-- Trims a the tree of records and return just the
-- leaves of the record
trimTree (Node r f) = trimLeaves r f

-- Helper function called by trimTree.
trimLeaves r f = Node r (trimLeaves' f) where
  trimLeaves' f =
    let result = fmap trimLeaves'' f where
          trimLeaves'' (Node r' f') = let
            result' = case f' of
              [] -> Nothing
              _ -> Just $ trimLeaves r' f'
            in
            result'
    in
      catMaybes result

-- Get  all the leaves from the record. Returns all leaves
-- across the record structure.
getLeaves :: (CellValueFormatter a) => Tree a -> [String]
getLeaves (Node r f) = case f of
  [] -> [(ppFormatter r)]
  _ -> foldMap getLeaves f

recsToTrees recs = fmap (\a -> Node "root" $ (toTree . G.from $ a)) $ recs

getHeaderDepth rec_trees = header_depth where
  header_depth = L.length . L.head . fillPath . constructPath . trimTree . L.head $ rec_trees

createBoxedHeaders :: [[String]] -> [B.Box]
createBoxedHeaders paths = boxes where
  boxes = L.map wrapWithBox paths
  wrapWithBox p = B.vsep 0 B.top $ L.map B.text p

--createHeaderCols :: [Tree String] -> [B.Box]
createHeaderCols rec_trees = header_boxes where
  header_boxes =  createBoxedHeaders . fillPath . constructPath . trimTree . L.head $ rec_trees

--createDataBoxes :: [Tree a] -> [B.Box]
createDataBoxes rec_trees = vertical_boxes where
  horizontal_boxes =  fmap (fmap  B.text) $ fmap getLeaves rec_trees
  vertical_boxes = fmap (B.vsep 0 B.top) $ L.transpose horizontal_boxes

--createIndexBoxes :: Map.Map a a -> B.Box
createIndexBoxes recs = index_box where
  rec_trees = recsToTrees $ Map.elems recs
  header_depth = getHeaderDepth rec_trees
  index_col = (L.replicate header_depth "-" ) ++  (L.map ppFormatter $ Map.keys recs)
  index_box = B.vsep 0 B.top $ L.map B.text index_col

createHeaderDataBoxes recs = vertical_boxes where
  rec_trees = recsToTrees recs
  header_boxes = createHeaderCols rec_trees
  data_boxes = createDataBoxes rec_trees
  vertical_boxes = fmap (\(a, b) -> B.vsep 0 B.top $ [a, b]) $ L.zip header_boxes data_boxes


-- testing

data T = C1 { aInt::Double, aString::String} deriving (Data, Typeable, Show,G.Generic)
data T1 = C2 { t1:: T, bInt::Double, bString::String} deriving (Data, Typeable, Show, G.Generic)

c1 = C1 1000 "record_c1fdsafaf"
c2 = C2 c1 100.12121 "record_c2"
c3 = C2 c1 1001.12111 "record_c2fdsafdsafsafdsafasfa"
c4 = C2 c1 22222.12121 "r"

instance Tabulate T ExpandWhenNested
instance Tabulate T1 ExpandWhenNested
instance CellValueFormatter T

data R2 = R2 {a::Maybe Integer} deriving (G.Generic, Show)
data R3 = R3 {r31::Maybe Integer, r32::String} deriving (G.Generic, Show)
tr =  Node "root" (toTree . G.from $ c2)
r2 = Node "root" (toTree . G.from $ (R2 (Just 10)))
r3 = Node "root" (toTree . G.from $ (R3 (Just 10) "r3_string"))

-- | Used with 'printTableWithFlds'
data DisplayFld a = forall s. CellValueFormatter s => DFld (a->s)

-- printTableWithFlds2 :: [DisplayFld t] -> V.Vector t -> IO ()
-- printTableWithFlds2 flds recs = B.printBox $ printTableWithFlds flds $ V.toList recs

-- printTableWithFlds3 :: (CellValueFormatter k) => [DisplayFld t] -> Map.Map k t -> IO ()
-- printTableWithFlds3 flds recs = results where
--   data_cols = printTableWithFlds flds $ Map.elems recs
--   index_cols = B.vsep 0 B.top $ fmap (B.text . ppFormatter) $ Map.keys recs
--   vertical_cols = B.hsep 5 B.top [index_cols, data_cols]
--   results = B.printBox vertical_cols
