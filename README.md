![Haskell-Torch Logo](https://github.com/abarbu/haskell-torch/blob/master/logo-with-text.png)

[![Build Status](https://img.shields.io/circleci/project/github/abarbu/haskell-torch.svg)](https://circleci.com/gh/abarbu/haskell-torch)

## Haskell-Torch

Still a work in progress. Stay tuned for updates.

Practical deep learning in Haskell built on the same C++ foundations as PyTorch
with an extensive ecosystem. We've got Tensorboard support, efficient image
manipulation support with imagemagick, HDF5 and Mat file support, and
reinforcement learning with the OpenAI Gym (with Box2D and MuJoCo). Same speed,
more safety, comprehensive, lots of pretrained models, and it works
today. [Check out examples of how to build typesafe CNNs like AlexNet and ResNet
and language models based on stacked LSTMs, along with GANs and VAEs. You'll see
how to efficiently stream data to them from datasets like MNIST, load pretrained
models from PyTorch, transfer weights back to PyTorch, use Tensorboard, and
everything else you need to write deep learning code in
Haskell.](https://github.com/abarbu/haskell-torch/tree/master/haskell-torch/src/Torch/Tutorial)

*Unfortunately, Haskell suffers from major issues and limitations that basically make it useless as a language for machine learning or game development. We need your help GHC devs! See below*

### Why?

Writing deep learning code is too error-prone. We spend a lot of our time
debugging subtle issues and fearing big refactors. That's exactly what Haskell
is good at.

### Contributors

This library is built on top of all of the hard work of the PyTorch
community. Haskell-Torch was developed by [Andrei Barbu](https://0xab.com). Andrei is a research
scientist at MIT working on understanding flexible human intelligence: how
children learn language, how robots can understand language, how language
enables us to perform new tasks, and how language is represented in the human
brain.

### Installation

You will first need Conda, a Python package manager. It's a simple install
`https://www.anaconda.com/distribution/#download-section`

Then, run `setup.sh`

We strongly suggest adding this to your bashrc file:

```export OMP_NUM_THREADS=1```

OpenMP seems to have terrible heuristics around how many threads it should
create causing constant issues. You may see strange problems on larger
machines without setting this. Of course, you may want to set it to some
number larger than 1. These problems can also occur in Haskell and the slowdowns
on larger machines can be terrible (on a 50 core machine we see a nearly 40x
slowdown compared to single core performance without this flag!).

After running `setup.sh` you can now build, just remember to also activate the 
environment. We've already built for you the first time.

```
conda activate haskell-torch
stack build haskell-torch --fast
```

If you are using Jupyter, build it and start it:

```
conda activate haskell-torch
stack install ihaskell --fast
stack exec ihaskell -- install --stack
stack exec jupyter -- notebook
```

Only the last command is needed in the future.

Since building takes some time and the Haskell code is not the dominant
part of the runtime, feel free to use `--fast`.

If you have a CUDA-capable setup it will be automatically recognized and CUDA
support will be enabled. If you change anything about CUDA support you will want
to rerun `setup.sh`. 

Note that code that requires CUDA will not typecheck unless you install a
cuda-capable version!

If you are starting an IDE that has to talk to GHC, make sure to start that IDE
from a shell where you have activated the haskell-torch environment in conda.

To run ghci from the commandline

```
stack ghci --no-load haskell-torch
```

Then, `:load Torch` and test things out with `ones @TFloat @KCpu @'[3,1] >>= out`
You should see a vector printed to your terminal. Have fun!

### FAQ

* How do I get started?

Read and run the example/tutorial code! The library is large, the types are
complex, and the type errors can be really scary. That's why we have a
comprehensive set of examples. You will find these in
`haskell-torch/src/Torch/Tutorial/Intro`

Read through the tutorial and follow along with it. Then reproduce some of the
networks in the tutorial yourself. That will give you a feel for how building up
networks incrementally works in terms of how the types align, when things are
too ambiguous, etc. Don't forget to use tensorboard. There's an example for that
too.

* Running `stack ghci` results in a slew of errors, what do I do?

That's normal and in any case, running ghci the standard way will load many
modules on startup making it load times very painful. Instead, skip all of this
and tell ghci to load only the haskell-torch package.

```
stack ghci --no-load haskell-torch
```

Then, `:load Torch` Now you're good to go!

* I get ambiguous type errors, what should I do?

These are actually extremely common all over Haskell, in particular for numeric
code. They're hidden by a defaulting mechanism, but this isn't exposed to
users. So we can't pick defaults the way that the builtin libraries can. It
would be nice if type checker plugins had access to defaults, but that's not the
situation today.

We have several pure functions that let you deal with this. In particular the
Tensor module has `typed`, `sized`, `stored`, `like`, `onCpu`, `onCuda`. None of
these have any effect at runtime. They constrain the types of their inputs in
particular ways. For example, `typed @TFloat x` ensures that x is a float,
`sized @'[2,3] x` ensures that the tensor is 2 by 3, `like x y` returns y
ensuring that its type is like x.

You may also find that errors come from GHC not running the constraint solver
enough because the default limit is very low. GHC will tell you this is the
case, you don't need to guess, but you may want to increase the maximum number
of iterations with `:set -fconstraint-solver-iterations=50` in GHC or with
`{-# OPTIONS_GHC -fconstraint-solver-iterations=50 #-}` in your code.


* How do I get started writing models?

Once you work through the tutorials you'll want to write your own models. Don't
just jump into this! Reproduce one of the larger models in the examples from
scratch by yourself step by step. This will give you a feel for how building
networks works without having to worry about the parameters of the network. GHC
is rather brittle today when it comes to the complex code in this library, so
you'll want to follow a certain pattern when making new models. Otherwise,
you'll end up with pages of crazy type errors.

Start by defining the type of your model, like:

```forward :: _ -> DataPurpose -> (Tensor 'TFloat 'KCpu '[100, 3, 32, 32]) -> IO (Tensor 'TFloat 'KCpu '[100, 10])```

You almost always want to take the DataPurpose as input (this tells you if the
data is training data or test data; some layers behave differently at test and
training time). The first parameter will contain the parameters and we clearly
defined our input and output. I like to make even the batch size, 100 above, and
the device, `KCpu`, concrete when developing the network and then relax them
later (`KCuda` being another popular option). Write the code piping layers with
`>=>` and end it with a hole like `>=> _` to see what the current output is vs
the desired output. Make the first argument of forward an n-tuple with the
different weights/parameters. Haskell will infer the type of that n-tuple and
eventually you can put it into a data declaration and call it whatever you
want.

In the final code you will usually see at least one record that holds the
parameters of the network. It's type is just derived by copying and pasting the
type of the tuple that was being fed into the model, wrapping it in a record,
and then using the RecordWildCards extension to replace the tuple. That's
post-hoc, I don't find it a good strategy for development. It also fossilizes
your network because the types of the inputs fix the types in the intermediate
layers. Changing your network becomes very painful because you need to change
both the record and the parameters of the layers in sync. I undo this record and
switch back to an n-tuple or I parameterize the record by an additional n-tuple
that holds other parameters until they can be moved into the record. Once we
remove that there is little redundancy making network changes very quick and
easy.

Note that we aren't jumping in the middle of a network and building out. And we
aren't making our network polymorphic from the get-go. That's a recipe for
disaster today, you want to start with concrete types. Inference is too brittle,
errors are too messy, defaulting is a mess leading to ambiguity errors, and the
layers are too polymorphic for this to be practical right now. It will get
better with time.

If you want to port a model from PyTorch you should print it. This is far better
than looking at the tensorboard graph, but note that some layers don't appear in
the printout! It's good to both print it with `print(module)` and to look at
`print(torch.jit.trace(model, input).graph)`. You can significantly clean up the
latter to remove constant nodes and list operations which are rarely
relevant. The former tells you the parameters to every layer (or at least the
ones that appear) and the latter is the truth about which operations actually
occur and the sizes of all of the intermediate tensors. That's particularly
useful in conjunction with this trick:

```>=> (\x -> pure $ (x :: Tensor 'TFloat 'KCpu '[100, 64, 8, 8]))```

You can insert that anywhere in your pipeline to ensure that certain tensors are
of a particular size. GHC will then guide you to where the error is.

If you factor out some intermediate layers, it's often best to leave the
constraints on those layers to GHC instead of trying to figure them out or
inserting the insane types that are currently inferred. Check out T06_ResNet for
the residual block. The inferred constraints on `residualBlock` are 38 lines
long, 1148 characters. Some of these constraints are legitimate, but not all.
This one is ok `(1 <=? stride) ~ 'True` but `(2 <=? (inH TN.+ 2))` is pointless
and obviously true. 

One day we'll be able to put these constraints in, but the real issue is are the
insanely long and incomprehensible ones that are legitimate:

```outH ~ (((((Div (((inH TN.+ 2) TN.- 2) TN.- 1) stride TN.+ 1) TN.+ 2) TN.- 2) TN.- 1) TN.+ 1)```

We will soon have a type checker plugin to simplify these. The actual constraint
simplified is very readable and useful! `outH ~ ((Div (inH TN.- 1) stride) TN.- 1)`
That's just a bit more awkward than `out = ((inH - 1) / stride) - 1` but not too
terrible.

* How do I join in the fun and contribute?

We'd love to have more models, more datasets, more layers, more tutorials,
etc. Head over to one of the many big libraries that PyTorch has an port over
features. 

Three guidelines for contributing: we want fast code, we want code that works
exactly like the pytorch code in every way, and we want code that supports all
of the options the original does. Horrible code that is fast is infinitely
better than elegant but slow code! Refactoring in Haskell is awesome so we can
clean code up, but every slow piece of code we have is technical debt and a
showstopper for anyone that happens to depend on it. Better to be upfront about
not having a feature than to have a slow non-replacement for that feature.

If you add a model, that model must be exactly the same as the original. No
shortcuts. This includes every parameter, dropout layers, inplace operations,
etc. It should produce floating-point equivalent results to the pytorch model.

If you add a feature, add every option that exists for that feature. You will
notice that our functions take a lot options! Don't be shy. That's life in the
ML world and the only alternative is to dumb things down, but then they stop
being useful for serious research and development. Every feature we add that is
missing options is technical debt we have to pay. We want people to have
confidence that when the library says it does X, it does it well and it does it
comprehensively. No surprises.

* How does this compare feature-wise to PyTorch?

This library includes most common features and roughly two thirds of all of the
features you would find in PyTorch and torchvision. We will soon reach feature
parity. You will find that some functions are less polymorphic and split up into
several alternatives, so the mapping between pytorch and haskell-torch the two
is not one to one. Data loading and how networks are written is rather
different. The major features unavailable at the moment are sparse tensors and a
few scalar types like complex numbers. They're rarely seen in the real world at
the moment and the overhead of adding them in vs. focusing on other features
hasn't been worth it so far.

* I have a PyTorch module, how do I translate it to Haskell-Torch?

There are two phases to this, first writing the Haskell code and second
transferring the weights. To write the code, you can print your PyTorch model to
get an idea of what it contains and what important parameters exist. You can
also get a complete understanding of inputs, outputs, and operations by using
the JIT. You can call your module like this `script_module = torch.jit.trace(model, args)`
You can then print `script_module.graph`. It will show you the size of all 
inputs, outputs, and intermediate results as well as every single operation
and its parameters along with most constants. The sizes of inputs and outputs
are invaluable to helping GHC produce good error messages as you write code.
See the answer on how to implement models in Haskell-Torch.

Loading the pretrained weights from PyTorch is easy, but you need to be mindful
of the file formats involved. PyTorch provides several formats to save models,
we support one of them, the TorchScript format. It bundles code & tensors, see
`Torch.StoredModel` for more details, but we only read back the tensors. Once
you trace your model as described above, you can save it
`script_module.save('/tmp/model.pt')` You can load this model in PyTorch. Look
at `Torch.Models.Vision.AlexNet` for an example of this and at
`Tutorial.Intro.T01_Basics` for more.

One useful PyTorch trick is that you can add extra tensors to the stored
model. If your model isn't producing the answer you expect, it can be useful to
have access to the inputs it gets at each timestep or intermediate results. To
do so, modify your PyTorch model to add new empty parameters, like 
`self.hidden = nn.Parameter()` then, before you save your model assign to 
this parameter `model.hidden = nn.Parameter(mytensor)` Then when you trace
that tensor will be included in the trace. Note that you need to update
this parameter before you call trace, you cannot update it in `forward`! These
models and tensors can then be loaded into Haskell to compare against your results.
Check out `Tensor.allclose`

* I get an ambiguous type error in the middle of a computation because of
  broadcasting, what now?

GHC really needs to give us a handle to such ambiguities so that we can resolve
them with plugins. Until then, you have two kinds of tools. `Tensor.like` takes
two tensors and returns the second unchanged while making sure that its type is
like that of the former. `Tensor.like a b` returns b and makes sure its type is
the same as a. Second, you can annotate tensors with properties like size, see
`Tensor.sized`, `Tensor.typed`, `Tensor.stored`, and similar functions.

