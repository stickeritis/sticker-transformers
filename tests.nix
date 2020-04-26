{ pkgs ? import (import ./nix/sources.nix).nixpkgs {} }:

let
  sources = import ./nix/sources.nix;
  models = import ./nix/models.nix;
  danieldk = pkgs.callPackage sources.danieldk {};
  libtorch = danieldk.libtorch.v1_5_0;
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
  cargo_nix = pkgs.callPackage ./nix/Cargo.nix { inherit buildRustCrate; };
in [
  # Test with HDF5 support disabled.
  (cargo_nix.rootCrate.build.override {
    features = [ "model-tests" ];
    runTests = true;
  })

  # Test with HDF5 support.
  (cargo_nix.rootCrate.build.override {
    features = [ "load-hdf5" "model-tests" ];
    runTests = true;
  })
]
