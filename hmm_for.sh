#!/bin/bash
for i in $(seq 1 31)
do
    python hmm_preprocess.py $i && python hmm_ar.py
done
