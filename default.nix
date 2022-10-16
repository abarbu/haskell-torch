{ withCuda ? false }:
let
  sources = import nix/sources.nix {};
  haskell-nix = import sources."haskell.nix" {};
  nixpkgs = haskell-nix.pkgs;
  gitignore = (import sources."gitignore.nix" {
    inherit (nixpkgs) lib;
  }).gitignoreSource;

  src = nixpkgs.lib.cleanSourceWith {
    name = "haskell-torch";
    src = gitignore ./.;
  };
  gym-metadata = import ./gym-haskell/nix/metadata.nix { sources = (import gym-haskell/nix/sources.nix {}) // sources; inherit withCuda; };
  pkgs = import haskell-nix.sources.nixpkgs
    (haskell-nix.nixpkgsArgs //
     { config.allowUnfree = true;
       overlays =
         haskell-nix.overlays ++
         gym-metadata.overlays ++
         [(self: super: { jupyterWith = import sources."jupyterWith.nix" {}; })
          (self: super:
            let torch_pkg = if withCuda then super.python38Packages.pytorchWithCuda else super.python38Packages.pytorchWithoutCuda;
            in {
              hdf5_hl = super.hdf5;
              c10 = torch_pkg;
              torch = torch_pkg.lib;
              torch_cpu = torch_pkg.dev;
              torch_cuda = torch_pkg.dev;
              "MagickWand-7.Q16HDRI" = super.imagemagick;
              "MagickCore-7.Q16HDRI" = super.imagemagick.dev;
              tensorboard = super.python38Packages.tensorflow-tensorboard;
              # gym
              # "python3.8" = pkgs.python38Packages.python;
              # mesa = super.mesa.override {
              #   enableOSMesa = true;
              # };
              mesa = super.mesa.override {
                enableOSMesa = true;
              };
            })
          # (self: super: {
          #   python3Packages = super.python3Packages // {
          #     # TODO Doesn't seem to work
          #     nbconvert = super.python3Packages.nbconvert.overrideAttrs (
          #       _: {
          #         patches = [ ./patches/jupyter-nbconvert-fix-theme-6.1.patch ];
          #       }
          #     );
          #   };
          # })
          (self: super: {
            python38Packages = super.python38Packages // {
              jupytext = super.python38Packages.jupytext.overrideAttrs (
                _: {
                  patches = [ ./patches/jupytext-add-haskell.patch ];
                }
              );
              # TODO Doesn't seem to work
              # nbconvert = super.python38Packages.nbconvert.overrideAttrs (
              #   _: {
              #     patches = [ ./patches/jupyter-nbconvert-fix-theme-6.1.patch ];
              #   }
              # );
            };
          })
          (self: super: {
            haskell-nix = super.haskell-nix // {
              # hsPkgs = super.haskell-nix.hsPkgs // {
              #   src = pkgs.fetchFromGitHub {
              #     owner = "lesscpy";
              #     repo = "lesscpy";
              #     rev = version;
              #     sha256 = "1jf5bp4ncvw2gahhkvjy5b0366y9x3ki9r9c5n6hkvifjk3jhmb2";
              #   };
              # hsPkgs = super.haskell-nix.hsPkgs // {
              #   ihaskell-magic = super.hackage-package {
              #     name = "ihaskell-magic";
              #     version = "0.3.0.1";
              #     compiler-nix-name = "ghc8107";
              #   };
              # };
              compiler = super.haskell-nix.compiler // {
                ghc8107 = super.haskell-nix.compiler.ghc8107.overrideAttrs  (
                  prev: {
                    src = prev.src.overrideAttrs (
                      prevSrc: {
                        patches = prevSrc.patches ++ [
                          ./patches/ambiguity-plugins-no-tests-ghc-8.10.3.patch
                        ];
                      }
                    );
                  }
                );
              };
            };
          })
         ];
     });
  gymDependencies = [ pkgs.python38Packages.python
                      pkgs.python38Packages.gym
                      # pkgs.atari-py
                      # pkgs.nes-py
                      # pkgs.gym-super-mario-bros
                      # pkgs.box2d
                      # pkgs.box2d-py
                      # pkgs.mujoco-py
                      # pkgs.mujoco
                    ];
in
pkgs.haskell-nix.stackProject' {
    inherit src;
    pkg-def-extras = [
      # (hackage: pkgs.lib.mapAttrs (n: v: hackage."${n}"."${v}".revisions.default) compiler-pkgs)
      (hackage:
        {
          # ihaskell-magic = hackage.ihaskell-magic;
          ihaskell-magic = (((hackage.ihaskell-magic)."0.3.0.1").revisions).default;
          ihaskell-graphviz = (((hackage.ihaskell-graphviz)."0.1.0.0").revisions).default;
          ihaskell-juicypixels = (((hackage.ihaskell-juicypixels)."1.1.0.1").revisions).default;
        }
          # {
          # lsp-test = import ./lsp-test.nix sources.lsp-test;
          # haskell-lsp = import ./haskell-lsp.nix sources.haskell-lsp;
          # haskell-lsp-types = import ./haskell-lsp-types.nix "${sources.haskell-lsp}/haskell-lsp-types";
          # ghcide = import ./ghcide.nix sources.ghcide;
          # ihaskell-magic = hackage.woof 
          # }
      )
    ];
    modules = [({pkgs, ...}: {
      doHaddock = false;
      packages.ihaskell.patches = [ ./patches/ihaskell-fixup-set-0.10.2.1.patch ];
      packages.haskell-torch.flags.cuda = withCuda;
      packages.haskell-torch-cbindings.flags.cuda = withCuda;
      packages.haskell-torch-tensorboard-proto.components.library.build-tools = [
        (pkgs.haskell-nix.hackage-package {name = "proto-lens-protoc";
                                           version = "0.7.1.0";
                                           compiler-nix-name = "ghc8107";}).components.exes.proto-lens-protoc
        pkgs.protobuf
      ];
      packages.haskell-torch-imagemagick.components.library.build-tools = [
        pkgs."MagickWand-7.Q16HDRI"
        pkgs."MagickCore-7.Q16HDRI"
        pkgs.imagemagick.dev
      ];
      packages.haskell-torch.components.library.build-tools = [
        pkgs.feh
      ];
      packages.haskell-torch-imagemagick.configureFlags = ["--extra-include-dirs=${pkgs.imagemagick.dev}/include/ImageMagick"];
      packages.haskell-torch.components.tests.doctest.build-tools =
        [ pkgs.haskell-nix.haskellPackages.hspec-discover ];
      packages.interpolateIO.components.tests.spec.build-tools =
        [ pkgs.haskell-nix.haskellPackages.hspec-discover ];
      # pkgs.haskell-nix.pkgs.hspec-discover
      # packages.cpython.patches = [ ./gym-haskell/patches/haskell-cpython-3.5.1-nixify-internals-getsize.patch ];
      # packages.gym.components.library.build-tools =
      #   [ pkgs.python38Packages.python
      #     pkgs.python38Packages.gym
      #   ];
      # packages.gym.components.tests.gym-test.build-tools =
      #   [ pkgs.python38Packages.python
      #     pkgs.python38Packages.gym
      #   ];
      # packages.cypthon.components.library.build-tools =
      #   [ pkgs.python38Packages.python
      #     pkgs.python38Packages.gym
      #   ];
      packages.gym.components.library.build-tools = gym-metadata.dependencies pkgs;
      packages.gym.components.tests.gym-test.build-tools = gym-metadata.dependencies pkgs;
      packages.cypthon.components.library.build-tools = gym-metadata.dependencies pkgs;
    })];
}
