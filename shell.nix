# We pin nixpkgs to improve reproducability. We don't pin Rust to a
# specific version, but use the latest stable release.

let
  sources = import ./nix/sources.nix;
  nixpkgs = import sources.nixpkgs {};
  danieldk = nixpkgs.callPackage sources.danieldk {};
  mozilla = nixpkgs.callPackage "${sources.mozilla}/package-set.nix" {};
in with nixpkgs; mkShell {
  nativeBuildInputs = [
    mozilla.latest.rustChannels.stable.rust
    pkgconfig
  ];

  buildInputs = [
    curl
    openssl
  ] ++ lib.optional stdenv.isDarwin darwin.apple_sdk.frameworks.Security;

  LIBTORCH = "${danieldk.python3Packages.pytorch.v1_3_1.dev}";
}
