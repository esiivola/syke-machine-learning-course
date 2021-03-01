#!/bin/bash
export GIT_COMMITTER_NAME=anonymous
export GIT_COMMITTER_EMAIL=anon@localhost

cd ~

git clone https://github.com/esiivola/syke-machine-learning-course

cd ./syke-machine-learning-course

conda env create -f environment.yml

conda activate SYKE-ML-ENV
