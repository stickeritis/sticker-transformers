#!/bin/bash

set -euo pipefail

SCRIPTDIR="$(dirname "${BASH_SOURCE[0]}")"
PARAMS="${SCRIPTDIR}/bert-base-german-cased.hdf5"

if [ -e "${PARAMS}" ]; then
  echo "The model was already downloaded."
  exit 0
fi

curl -o "${PARAMS}" \
  http://www.sfs.uni-tuebingen.de/a3-public-data/sticker2-models/bert-base-german-cased.hdf5
