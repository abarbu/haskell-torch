# String interpolation in IO!

This package is part of the Haskell-Torch ecosystem. Check out the documentation
[there](https://github.com/abarbu/haskell-torch).

It is a fork of [interpolate](http://hackage.haskell.org/package/interpolate) by
Simon Hengel. For when you have values that can only be read by IO operations
like IORefs or matrices backed by C data.

## Examples

    >>> :set -XQuasiQuotes
    >>> import Data.String.InterpolateIO

Interpolates strings

    >>> let name = "Marvin"
    >>> putStrLn =<< [c|name: #{name}|]
    name: Marvin

or integers

    >>> let age = 23
    >>> putStrLn =<< [c|age: #{age}|]
    age: 23

or arbitrary Haskell expressions

    >>> let profession = "\955-scientist"
    >>> putStrLn =<< [c|profession: #{unwords [name, "the", profession]}|]
    profession: Marvin the λ-scientist

or values in IO

    >>> import System.Environment
    >>> let profession = "\955-scientist"
    >>> putStrLn =<< [c|home directory: #{getEnv "HOME"}|]
    profession: <your home directory>
