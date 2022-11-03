#!/bin/bash

declare -a datasets=("nl-conll" "es-conll" "wnut17")
declare -a splits=("train" "dev" "test")

for dataset in ${datasets[@]}; do
    for split in ${splits[@]}; do
        python -m scripts.preprocess assets/$dataset-$split.iob assets/$dataset-$split.iob
        python -m scripts.convert_to_spans assets/$dataset-$split.iob corpus/ner/ --use-ents
        python -m scripts.convert_to-spans assets/$dataset-$split.iob corpus/spancat/ --spans-key sc
    done
done

# Run archaeo separately (needs splitting)
python -m scripts.convert_to_spans assets/archaeo.iob corpus/ner/ --use-ents --converter iob
python -m scripts.split_docs corpus/ner/archaeo.spacy corpus/ner/ --seed 42 --shuffle
python -m scripts.convert_to_spans assets/archaeo.bio corpus/spancat/ --spans-key sc --converter iob
python -m scripts.split_docs corpus/spancat/archaeo.spacy corpus/spancat --seed 42 --shuffle