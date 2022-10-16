module IHaskell.Display.Matplotlib where
import Graphics.Matplotlib
import Graphics.Matplotlib.Internal
import IHaskell.Display

{-# LANGUAGE ExtendedDefaultRules #-}

-- Only bleeding edge matplotlib has these helpers as of 2021. Delete these
-- after a few years.

-- | Get the SVG for a figure
toSvg' :: Matplotlib -> IO (Either String String)
toSvg' m = withMplot m (\s -> python $ pyIncludes "" ++ s ++ pySVG)

pySVG' :: [[Char]]
pySVG' =
  ["import io"
  ,"i = io.StringIO()"
  ,"plot.savefig(i, format='svg')"
  ,"print(i.getvalue())"]

instance IHaskellDisplay Matplotlib where
  display m = do
    r <- toSvg' m
    case r of
      Left v -> error v
      Right v -> return $ Display [svg v]
