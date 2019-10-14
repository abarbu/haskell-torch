module Data.String.InterpolateIO.Util (unindent) where

import           Control.Arrow ((>>>))
import           Data.Char

-- | Remove indentation as much as possible while preserving relative
-- indentation levels.
--
-- `unindent` is useful in combination with `Data.String.Interpolate.c` to remove leading spaces that
-- resulted from code indentation.  That way you can freely indent your string
-- literals without the indentation ending up in the resulting strings.
--
-- Here is an example:
--
-- >>> :set -XQuasiQuotes
-- >>> import Data.String.Interpolate
-- >>> import Data.String.Interpolate.Util
-- >>> :{
--  putStr $ unindent [i|
--      def foo
--        23
--      end
--    |]
-- :}
-- def foo
--   23
-- end
--
-- To allow this, two additional things are being done, apart from removing
-- indentation:
--
-- - One empty line at the beginning will be removed and
-- - if the last newline character (@"\\n"@) is followed by spaces, the spaces are removed.
unindent :: String -> String
unindent =
      lines_
  >>> removeLeadingEmptyLine
  >>> trimLastLine
  >>> removeIndentation
  >>> concat
  where
    isEmptyLine :: String -> Bool
    isEmptyLine = all isSpace

    lines_ :: String -> [String]
    lines_ [] = []
    lines_ s = case span (/= '\n') s of
      (first, '\n' : rest) -> (first ++ "\n") : lines_ rest
      (first, rest) -> first : lines_ rest

    removeLeadingEmptyLine :: [String] -> [String]
    removeLeadingEmptyLine xs = case xs of
      y:ys | isEmptyLine y -> ys
      _ -> xs

    trimLastLine :: [String] -> [String]
    trimLastLine (a : b : r) = a : trimLastLine (b : r)
    trimLastLine [a] = if all (== ' ') a
      then []
      else [a]
    trimLastLine [] = []

    removeIndentation :: [String] -> [String]
    removeIndentation ys = map (dropSpaces indentation) ys
      where
        dropSpaces 0 s = s
        dropSpaces n (' ' : r) = dropSpaces (n - 1) r
        dropSpaces _ s = s
        indentation = minimalIndentation ys
        minimalIndentation =
            safeMinimum 0
          . map (length . takeWhile (== ' '))
          . removeEmptyLines
        removeEmptyLines = filter (not . isEmptyLine)

        safeMinimum :: Ord a => a -> [a] -> a
        safeMinimum x xs = case xs of
          [] -> x
          _ -> minimum xs
