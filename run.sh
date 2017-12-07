#!/bin/bash

# run.sh

# --
# Download data

mkdir -p data

wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzvf aclImdb_v1.tar.gz && rm aclImdb_v1.tar.gz
mv aclImdb_v1 ./data/aclImdb_v1

# --
# Train model

python bow.py