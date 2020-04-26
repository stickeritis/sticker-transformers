# We pin nixpkgs to improve reproducability. We don't pin Rust to a
# specific version, but use the latest stable release.

let
  sources = import ./nix/sources.nix;
  nixpkgs = import sources.nixpkgs {};
  danieldk = nixpkgs.callPackage sources.danieldk {};
  mozilla = nixpkgs.callPackage "${sources.mozilla}/package-set.nix" {};
  libtorch = danieldk.libtorch.v1_5_0;
in with nixpkgs; mkShell {
  nativeBuildInputs = [
    mozilla.latest.rustChannels.stable.rust
    pkgconfig
  ];

  buildInputs = [
    curl
    libtorch
    openssl
  ] ++ lib.optional stdenv.isDarwin darwin.apple_sdk.frameworks.Security;

  propagatedBuildInputs = [
    (python3.withPackages (ps: with ps; [
      danieldk.python3Packages.pytorch.v1_4_0
      h5py
      tensorflow-bin
    ]))
  ];

  # Unless we use pkg-config, the hdf5-sys build script does not like
  # it if libraries and includes are in different directories.
  HDF5_DIR = symlinkJoin { name = "hdf5-join"; paths = [ hdf5.dev hdf5.out ]; };

  LIBTORCH = libtorch.dev;
}
