{ withJupyter ? false,
  withCuda ? false }:
let
  package = import ./default.nix {inherit withCuda;};
  # pkgs = builtins.trace package.hsPkgs.ihaskell.components.library.outputs package.pkgs;
  # pkgs = builtins.trace (builtins.attrNames package.hsPkgs.ihaskell) package.pkgs;
  pkgs = package.pkgs;
  gym-shell = import ./gym-haskell/shell.nix { inherit withCuda;
                                               inherit package;
                                               sources = (import gym-haskell/nix/sources.nix {}) // (import nix/sources.nix {}); };
  buildInputs = [ pkgs.mesa.osmesa
                  pkgs.libGL
                  pkgs.xvfb-run
                  pkgs.xorg.libXinerama
                  pkgs.xorg.libXcursor
                  pkgs.libGLU
                  pkgs.mesa.drivers
                  pkgs.mesa
                  pkgs.mesa.dev
                  pkgs.python38Packages.matplotlib
                  pkgs.python38Packages.scipy
                  pkgs.texlive.combined.scheme-small
                  pkgs.curl
                  pkgs.cacert
                  pkgs.graphviz
                  pkgs."MagickWand-7.Q16HDRI"
                  pkgs."MagickCore-7.Q16HDRI"
                  pkgs.imagemagick.dev
                  pkgs.imagemagick
                  pkgs.protobuf
                  pkgs.hdf5
                  pkgs.matio
                  pkgs.c10
                  pkgs.feh
                ] ++ (if withJupyter
                      then [pkgs.python38Packages.jupytext]
                      else [])
                  ++ gym-shell.buildInputs;
  lib = pkgs.lib;
  mapPkgs = v: "${lib.concatImapStringsSep ":" (pos: x: x + "/${pkgs.python3.sitePackages}") v}";
  jupyterPkgs = package.pkgs.jupyterWith.nixpkgs.python3.pkgs;
  jupyterPackages = package.pkgs.jupyterWith.nixpkgs.python38Packages;
  # ihaskell = (pkgs.haskell-nix.hackage-package {
  #   name = "ihaskell";
  #   version = "0.10.2.1";
  #   compiler-nix-name = "ghc8107";
  #   # patches = [ ./patches/ihaskell-fixup-set-0.10.2.1.patch ];
  # });
  # ihaskell-magic = (pkgs.haskell-nix.hackage-package
  #   {name = "ihaskell-magic";
  #    version = "0.3.0.1";
  #    compiler-nix-name = "ghc8107";
  #   });
  extra-python-packages = pkgs.python38.withPackages (pp: [
    pp.matplotlib
    pp.gym
    package.pkgs.atari-py
    package.pkgs.nes-py
    package.pkgs.gym-super-mario-bros
    package.pkgs.box2d
    package.pkgs.box2d-py
    package.pkgs.mujoco-py
    package.pkgs.mujoco
  ]);
  jupyterEnvironment = (package.pkgs.jupyterWith.jupyterlabWith {
    extraPackages = p: [buildInputs] ++ [
      (jupyterPkgs.buildPythonPackage rec {
        pname = "jupyterthemes";
        version = "0.20.0";
        propagatedBuildInputs = [ jupyterPackages.notebook
                                  jupyterPackages.matplotlib
                                  (jupyterPkgs.buildPythonPackage rec {
                                    pname = "lesscpy";
                                    version = "0.13.0";
                                    propagatedBuildInputs = [ jupyterPackages.six jupyterPackages.ply ];
                                    doCheck = false;
                                    src = jupyterPkgs.fetchPypi {
                                      inherit pname version;
                                      sha256 = "1bbjag13kawnjdn7q4flfrkd0a21rgn9ycfqsgfdmg658jsx1ipk";
                                    };
                                    # src = pkgs.fetchFromGitHub {
                                    #   owner = "lesscpy";
                                    #   repo = "lesscpy";
                                    #   rev = version;
                                    #   sha256 = "1jf5bp4ncvw2gahhkvjy5b0366y9x3ki9r9c5n6hkvifjk3jhmb2";
                                    # };
                                    LC_ALL = "en_US.utf8";
                                  })
                                ];
        src = jupyterPkgs.fetchPypi {
          inherit pname version;
          sha256 = "07mldarwi9wi5m4v4x9s1n9m6grab307yxgip6csn4mjhh6br3ia";
        };
      })];
    extraJupyterPath =
      pkgs: "${extra-python-packages}/${extra-python-packages.sitePackages}"
      # pkgs:
      # mapPkgs (lib.attrVals [
      #   # matplotlib-related packages
      #   "matplotlib" "scipy" "numpy" "pillow" "cycler"
      #   "kiwisolver" "pytz" "cffi" "defusedxml" "certifi" "pyparsing" "six" "mock" "pbr"
      #   "sniffio" "tkinter" "olefile" "pytz" "anyio" "pyjson5" 
      #   #
      #   "jupytext"
      #   #
      #   "gym"
      # ]
      #   pkgs.python38Packages)
    ;
    kernels = [
      (package.pkgs.jupyterWith.kernels.iPythonWith {
        name = "python";
        packages = p:
          ((builtins.attrValues (pkgs.haskell-nix.haskellLib.selectLocalPackages package.hsPkgs))
           ++
           buildInputs
          );
      })
      (package.pkgs.jupyterWith.kernels.iHaskellWith {
        name = "haskell";
        extraIHaskellFlags = "--codemirror Haskell";
        customIHaskell = pkgs.symlinkJoin {
          name="ihaskell-haskell.nix";
          paths=[
            # ihaskell.components.exes.ihaskell
            # ihaskell.components.library
            package.hsPkgs.ihaskell.components.exes.ihaskell
            package.hsPkgs.ihaskell.components.library
          ];
        };
        packages = p:
          ((builtins.attrValues (pkgs.haskell-nix.haskellLib.selectLocalPackages package.hsPkgs))
           ++
           [
             # ihaskell
             # ihaskell-magic
             package.hsPkgs.ihaskell
             package.hsPkgs.ihaskell-magic
             package.hsPkgs.ihaskell-graphviz
             package.hsPkgs.ihaskell-juicypixels
             # (pkgs.haskell-nix.hackage-package {name = "ihaskell-magic"; version = "0.3.0.1"; compiler-nix-name = "ghc8107";
             #                                    pkg-set = package.pkgs;
             #                                   })
            # package.hsPkgs.ihaskell-magic
           ]
           ++
           buildInputs
          );
        haskellPackages = package;
      })];
  });
  mergeEnvs = envs: pkgs.mkShell (builtins.foldl' (a: v: {
    buildInputs = a.buildInputs ++ v.buildInputs;
    nativeBuildInputs = a.nativeBuildInputs ++ v.nativeBuildInputs;
    propagatedBuildInputs = a.propagatedBuildInputs ++ v.propagatedBuildInputs;
    propagatedNativeBuildInputs = a.propagatedNativeBuildInputs ++ v.propagatedNativeBuildInputs;
    shellHook = a.shellHook + "\n" + v.shellHook;
    LD_LIBRARY_PATH =
      # builtins.trace v.LD_LIBRARY_PATH v.LD_LIBRARY_PATH
      (if builtins.hasAttr "LD_LIBRARY_PATH" a then a.LD_LIBRARY_PATH else "")
      + ":" +
      (if builtins.hasAttr "LD_LIBRARY_PATH" v then v.LD_LIBRARY_PATH else "")
    ;
  }) (pkgs.mkShell {}) envs);
in

# jupyterEnvironment.env
# package.hsPkgs.ihaskell.components.exes.ihaskell

# (package.shellFor {
#     tools = {
#       cabal = { version = "latest"; };
#       hpack = { version = "latest"; };
#       hlint = { version = "latest"; };
#       ormolu = { version = "latest"; };
#       haskell-language-server = { version = "latest"; };
#     };
#     buildInputs = (if withJupyter then [jupyterEnvironment] else []) ++ buildInputs;
#     exactDeps = true;
#     LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
# })

builtins.trace (builtins.attrNames extra-python-packages)
  # (map (x: builtins.attrNames x) pkgs.python38Packages.matplotlib.requiredPythonModules)
  # (builtins.attrNames pkgs.python38Packages.matplotlib)
  (
  mergeEnvs (
  [(package.shellFor {
    tools = {
      cabal = { version = "latest"; };
      hpack = { version = "latest"; };
      hlint = { version = "latest"; };
      ormolu = { version = "latest"; };
      haskell-language-server = { version = "latest"; };
    };
    buildInputs = (if withJupyter then [jupyterEnvironment] else []) ++ buildInputs;
    exactDeps = true;
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
  })]
  ++
  (if withJupyter then [jupyterEnvironment.env] else  [])
  ++
  [gym-shell]
  ))

  # # builtins.trace
  # #   (
  # #     builtins.attrNames jupyterEnvironment
  # #   )
  # #   (
  #     package.shellFor {
  #   tools = {
  #     cabal = { version = "latest"; };
  #     hpack = { version = "latest"; };
  #     hlint = { version = "latest"; };
  #     ormolu = { version = "latest"; };
  #     haskell-language-server = { version = "latest"; };
  #   };
  #   buildInputs = (if withJupyter then [jupyterEnvironment] else []) ++ buildInputs;
  #   exactDeps = true;
  #   LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
  # }
  #   # )
