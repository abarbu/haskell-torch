![Haskell-Torch Logo](https://github.com/abarbu/haskell-torch/blob/master/logo-with-text.png)

## Haskell-Torch

*If you've manged to find this package, you should ignore it for now. We'll have an official release in a week or two!*

Practical deep learning in Haskell built on the same C++ foundations as
PyTorch. Same speed, more safety, comprehensive, lots of pretrained models, and
it works today.

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

Run the setup script and then build with stack

```setup.sh```

Any time you want to build just run:

```
conda activate haskell-torch
stack build haskell-torch
```

Since building takes some time and the Haskell code does is not the dominant
runtime you might instead consider 

```stack build haskell-torch --fast```

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

Two guidelines for contributing: we want fast code and we want code that
supports all of the options the original does. Horrible code that is fast is
infinitely better than elegant but slow code. Refactoring in Haskell is awesome
and simple so we can clean code up, but every slow piece of code we have is
technical debt and a showstopper for anyone that happens to depend on it. Better
to be upfront about not having a feature than to have a slow non-replacement for
that feature. If you add a feature, add every option that exists for that
feature. You will notice that our functions take a lot options! Don't be
shy. That's life in the ML world and the only alternative is to dumb things
down, but then they stop being useful for serious research and
development. Every feature we add that is missing options is technical debt we
have to pay. We want people to have confidence that when the library says it
does X, it does it well and it does it comprehensively. No surprises.

* How does this compare feature-wise to PyTorch?

This library includes most common features and roughly two thirds of all of the
features you would find in PyTorch and torchvision. We will soon reach feature
parity. You will find that some functions are less polymorphic and split up into
several alternatives, so the mapping between the two is not one to one. Data
loading and how networks are written is also rather different. The major
features unavailable at the moment are sparse tensors and a few scalar types
like complex numbers. They're rarely seen in the real world at the moment and
the overhead of adding them in vs focusing on other features hasn't been worth
it so far.

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
them with plugins. Until then, you have two types of tools. `Tensor.like` takes
two tensors and returns the second unchanged while making sure that its type is
like that of the former. `Tensor.like a b` returns b and makes sure its type is
the same as a. Second, you can annotate tensors with many properties like as
size, see `Tensor.sized`, `Tensor.typed`, `Tensor.stored`, and similar
functions.

* Why install with Conda instead of doing everything manually or using nix?

Conda is the standard way to install PyTorch and related libraries, if we don't
go down that route, we lose the ability to interoperate with the rest of the
ecosystem. The goal of Haskell-Torch isn't to replace everything in the distant
future, but to try to be practical today. It's not the Haskell way of managing
packages, but that's a small price to pay.

### What we critically need from Haskell and GHC

Haskell has five major issues that stop ML code in its tracks. Haskell-Torch is
carefully designed to avoid some of these but part of the reason we're releasing
this library is as a request/plea for developers to look into these issues. The
initial version of Haskell-Torch had a lot more type safety, but we had to strip
pretty much all of that out to make the library usable. We could do so much more
in terms of type safety with these changes, they would help many projects, and
Haskell could be a truly great language for ML. 

GHC devs, please help!

1. *gc* The only reason this library is being released after being private for
   several years is because of the new low-latency GC that is coming soon. The
   current GC is totally inadequate for ML. Haskell finalizers have to run
   regularly to clean up huge amounts of data sitting on the C++ heap. In
   addition to the new GC we need to have support for collecting garbage in the
   Haskell heap more often and informing the GC of off-heap memory. The GC has
   no hope of being useful if it doesn't know that a trivial haskell pointer is
   actually backed by a 10gb tensor so it really needs to pay attention to it.
2. *ambiguity* Haskell relies on a crutch for even simple code that involves
   numbers like `print 1`. The type defaulting mechanism provides `1` with an
   Integer type. This problem of intermediate results with totally safe and sane
   defaults is pervasive throughout numerical computing and ML. We need access
   to the type defaulting mechanism in type plugins, perhaps by giving all
   plugins final notice that a wanted will be ambiguous.
3. *errors* The errors in haskell are terrible unreadable monsters particularly
   with the complex types in libraries like this. I have gotten 500 line errors
   from Haskell-Torch in the past. We need to have something like error plugins
   that can post-process errors to make them readable for humans. Individual
   libraries understand the errors that they will produce far better than any
   general-purpose code ever could. We really want to say "the 2nd argument of
   this tensor should be divisible by 2". the Tame thing happens for other
   libraries with a lot of polymorphism, for example lens, and we could all do
   much better. Also, newcomers to haskell simply can't understand ghcs errors,
   we could make a plugin that provides friendly errors for those who want
   them. Libraries can also make much saner suggestions about how to fix certain
   errors.
4. *non-recursive bindings*. Haskell `let` is equivalent to Scheme's
   `letrec`. This means that you can't write code like: `out <- fn1 out; out <- fn2 out`. 
   You need to uniquely name each of these values. Long chains like
   this are common in ML code and the intermediate values have no useful
   names. You would be surprised by how much pain this adds to the language and
   how error prone writing code is because. This is the #1 source of runtime
   bugs in Haskell-Torch and the resulting bug is extremely nasty to find: you
   just sit in an infinite loop. We need a flag to make let bindings act like
   Scheme's `let` not `letrec`. I know that to many this will seem
   like a minor concern, but it is not. A lot of ML code is ported from
   different sources that are originally written in this style. This issue takes
   something that would be a trivial 5 minute port and turns out into a
   nightmare of debugging and infinite loops.
5. *plugin and extension export*. We need to export what plugins and extension
   are required to work with our code. This library requires a lot of extensions
   and a lot of plugins! The problem is not with end users having a template
   that they copy and paste into their code, or the fact that this is verbose,
   it's more fundamental. We will have to change the extensions and plugins that
   we use over time. That will require every user to change their code in a way
   that is totally unpredictable and impossible to diagnose from the error
   messages. Without the ability to export plugins or extensions we are doomed
   to either constantly break user code in a really nasty way or we are locked
   in to our early choices forever.

Some minor usability requests, although one can live without these:

 6. We need qualified exports to keep the library in some decent shape. Not
    having them is a pain and a usability nightmare for large heterogenous
    libraries like this. ML libraries have to do all sorts of random things from
    load image, to writing tensors, to dealing with memory concerns. Having all
    of them in the same flat namespace is a mess and asking users to copy and
    paste an import block is not practical. Users can't be asked to constantly
    update that import block as the shape of the library changes.
 7. Linear types. We really need them to ensure type safety for many parts of
    the library, particularly inplace operations. We can't wait to have them!
 8. Type argument support. We need to be able to annotate which type parameters
    are actually mean to be arguments in Haddock. Documentation is far too hard
    to read without this.
