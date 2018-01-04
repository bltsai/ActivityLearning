#!/bin/bash
for i in 10 30 50 75
do
    for j in 5 10 20 30
    do
        python3 ar_topic.py $i $j > ar_t_ratio_${i}_white_${j}.txt
    done
done