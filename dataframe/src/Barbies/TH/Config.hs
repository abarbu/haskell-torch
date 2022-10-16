module Barbies.TH.Config
  ( DeclareBareBConfig(..)
  , classic
  , passthrough
  ) where
import Language.Haskell.TH

-- | Keep it in a separate module until NoFieldSelectors gets widespread
data DeclareBareBConfig = DeclareBareBConfig
  { friends :: [Name] -- ^ Members with these types won't be wrapped with 'Wear'
  , bareName :: String -> Maybe String
  -- ^ generate a type synonym for the 'Barbies.Bare.Bare' type?
  , coveredName :: String -> Maybe String
  -- ^ generate a type synonym for the 'Barbies.Bare.Covered' type?
  , barbieName :: String -> String
  -- ^ modify the name of the datatype
  , switchName :: Q Name
  -- ^ the name of the type parameter to toggle between Bare and covered
  , wrapperName :: Q Name
  -- ^ the name of the type parameter of the wrapper for each field
  }

-- | Does not define any type synonyms
classic :: DeclareBareBConfig
classic = DeclareBareBConfig
  { friends = []
  , bareName = const Nothing
  , coveredName = const Nothing
  , barbieName = id
  , switchName = newName "sw"
  , wrapperName = newName "h"
  }

-- | Defines a synonym for the bare type with the same name.
-- The strippable definition is suffixed by B, and the covered type is suffixed by H.
passthrough :: DeclareBareBConfig
passthrough = DeclareBareBConfig
  { friends = []
  , bareName = Just
  , coveredName = Just . (++"H")
  , barbieName = (++"B")
  , switchName = newName "sw"
  , wrapperName = newName "h"
  }
