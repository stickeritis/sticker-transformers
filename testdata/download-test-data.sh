#!/usr/bin/env bash

set -euo pipefail

models=(
  http://www.sfs.uni-tuebingen.de/a3-public-data/sticker2-models/bert-base-german-cased.hdf5
  http://www.sfs.uni-tuebingen.de/a3-public-data/sticker2-models/xlm-roberta-base.hdf5
)

SCRIPTDIR="$(dirname "${BASH_SOURCE[0]}")"

for model in ${models[@]}; do
  model_file="${SCRIPTDIR}/$(basename $model)"

  if [ -e "${model_file}" ]; then
    echo "${model_file} is already available"
    continue
  fi

  curl -o "${model_file}" "${model}"
done

