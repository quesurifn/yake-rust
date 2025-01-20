#!/bin/bash

set -e

cd "${BASH_SOURCE%/*}/../yake_rust/datasets"


URLS=(
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/110-PT-BN-KP.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/500N-KPCrowd-v1.1.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/cacic.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/citeulike180.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/fao30.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/fao780.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/Inspec.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/kdd.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/Krapivin2009.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/Nguyen2007.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/pak2018.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/PubMed.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/Schutz2008.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/SemEval2010.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/SemEval2017.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/theses100.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/wicc.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/wiki20.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/WikiNews.zip"
    "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/refs/heads/master/datasets/www.zip"
)

for URL in "${URLS[@]}"; do
    NAME=$(basename "$URL")

    if ! [ -f "$NAME" ]; then
      wget -N -q -O "$NAME" "$URL"
    fi
done
