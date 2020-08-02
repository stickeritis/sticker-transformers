{ pkgs ? import (import ./nix/sources.nix).nixpkgs {} }:

let
  sources = import ./nix/sources.nix;
  models = import ./nix/models.nix;
  sticker = pkgs.callPackage sources.sticker {};
  libtorch = sticker.libtorch.v1_6_0;
  crateOverrides = with pkgs; defaultCrateOverrides // {
    hdf5-sys = attr: {
      HDF5_DIR = symlinkJoin { name = "hdf5-join"; paths = [ hdf5.dev hdf5.out ]; };
    };

    sticker-transformers = attr: {
      buildInputs = [ libtorch ] ++ lib.optional stdenv.isDarwin darwin.Security;
    } // models;

    torch-sys = attr: {
      nativeBuildInputs = lib.optional stdenv.isDarwin curl;

      LIBTORCH = libtorch.dev;
    };
  };
  buildRustCrate = pkgs.buildRustCrate.override {
    defaultCrateOverrides = crateOverrides;
  };
  crateTools = import "${sources.crate2nix}/tools.nix" {};
  cargoNix = pkgs.callPackage (crateTools.generatedCargoNix {
    name = "sticker-transformers";
    src = pkgs.nix-gitignore.gitignoreSource [ ".git/" "nix/" "*.nix" ] ./.;
  }) {
    inherit buildRustCrate;
  };
in [
  # Test with HDF5 support disabled.
  (cargoNix.rootCrate.build.override {
    features = [ "model-tests" ];
    runTests = true;
  })

  # Test with HDF5 support.
  (cargoNix.rootCrate.build.override {
    features = [ "load-hdf5" "model-tests" ];
    runTests = true;
  })
]
