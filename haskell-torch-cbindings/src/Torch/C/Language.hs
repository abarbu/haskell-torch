{-# LANGUAGE PatternSynonyms, TemplateHaskell #-}

-- | Some helpful @inline-c@ QuasiQuoters

module Torch.C.Language(cstorable) where
import           Foreign
import qualified Language.C.Inline         as C
import           Language.Haskell.TH
import           Language.Haskell.TH.Quote

cstorable :: Name -> String -> DecsQ
cstorable ty cname =
  [d|instance Storable $(conT ty) where
       sizeOf _ = fromIntegral $(quoteExp C.pure str)
       alignment _ = alignment (undefined :: Ptr ())
       peek = error "not implemented"
       poke = error "not implemented"|]
 where str = "size_t { sizeof(" ++ cname ++ ") }"
