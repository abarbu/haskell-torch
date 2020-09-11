{-# LANGUAGE MultiParamTypeClasses, KindSignatures, FlexibleInstances, DataKinds, PatternSynonyms, StandaloneDeriving, GeneralizedNewtypeDeriving, PolyKinds #-}
module Plugin.DefaultType(DefaultType,plugin) where
import GhcPlugins
import TcRnTypes
import Constraint
import TcPluginM
import qualified Inst
import InstEnv
import TcSimplify (approximateWC)
import qualified  Finder
import Panic      (panicDoc)
import Data.List
import TcType
import qualified Data.Map as M
import TyCoRep (Type(..))
import TyCon (TyCon(..))
import Control.Monad (liftM2)
import GHC.TypeLits

class DefaultType x (y :: x)

instance Eq Type where
  (==) = eqType
instance Ord Type where
  compare = nonDetCmpType
instance Semigroup (TcPluginM [a]) where
  (<>) = liftM2 (++)
instance Monoid (TcPluginM [a]) where
  mempty = pure mempty

plugin :: Plugin
plugin = defaultPlugin {
  defaultingPlugin = install,
  pluginRecompile = purePlugin
  }

install args = Just $ DefaultingPlugin { dePluginInit = initialize
                                       , dePluginRun  = run
                                       , dePluginStop = stop
                                       }

pattern FoundModule :: Module -> FindResult
pattern FoundModule a <- Found _ a
fr_mod :: a -> a
fr_mod = id

lookupModule :: ModuleName -- ^ Name of the module
             -> TcPluginM Module
lookupModule mod_nm = do
  hsc_env <- TcPluginM.getTopEnv
  found_module <- TcPluginM.tcPluginIO $ Finder.findPluginModule hsc_env mod_nm
  case found_module of
    FoundModule h -> return (fr_mod h)
    _          -> do
      found_module' <- TcPluginM.findImportedModule mod_nm $ Just $ fsLit "this"
      case found_module' of
        FoundModule h -> return (fr_mod h)
        _          -> panicDoc "Unable to resolve module looked up by plugin: "
                               (ppr mod_nm)

data PluginState = PluginState { defaultClassName :: Name }

-- | Find a 'Name' in a 'Module' given an 'OccName'
lookupName :: Module -> OccName -> TcPluginM Name
lookupName md occ = lookupOrig md occ

solveDefaultType :: PluginState -> [Ct] -> TcPluginM DefaultingPluginResult
solveDefaultType _     []      = return []
solveDefaultType state wanteds = do
  envs <- getInstEnvs
  insts <- classInstances envs <$> tcLookupClass (defaultClassName state)
  let defaults = foldl' (\m inst ->
                           case is_tys inst of
                             [matchty, replacety] ->
                               M.insertWith (++) matchty [replacety] m) M.empty insts
  let groups =
        foldl' (\m wanted ->
                  foldl' (\m var -> M.insertWith (++) var [wanted] m)
                         m
                         (filter (isVariableDefaultable defaults) $ tyCoVarsOfCtList wanted))
               M.empty wanteds
  M.foldMapWithKey (\var cts ->
                    case M.lookup (tyVarKind var) defaults of
                      Nothing -> error "Bug, we already checked that this variable has a default"
                      Just deftys -> do
                        pure [(deftys, (var, cts))])
    groups
  where isVariableDefaultable defaults v = isAmbiguousTyVar v && M.member (tyVarKind v) defaults

lookupDefaultTypes :: TcPluginM PluginState
lookupDefaultTypes = do
    md   <- lookupModule (mkModuleName "Plugin.DefaultType")
    name <- lookupName md (mkTcOcc "DefaultType")
    pure $ PluginState { defaultClassName = name }

initialize = do
  lookupDefaultTypes

run s ws = do
  solveDefaultType s (ctsElts $ approximateWC False ws)

stop _ = do
  return ()
