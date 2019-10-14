main :: IO ()
main = putStrLn "Test suite not yet implemented"

-- TODO Move these to a real test suite, for now they're just snippets to try things out.

-- runWithCuda (pure ())
-- runWithCpu (pure ())

-- Torch.C.State.runWithCuda (Torch.C.CUDA.mkTensorOnDevice (3,4) 0)

-- z :: IO (Ptr HTensor)
-- z = do
--   c <- runWithCuda (mkTensorOnDevice (3::CLong,4::CLong) (Just 0))
