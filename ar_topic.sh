#!/bin/bash
for j in 5 10 20 30
do
    for i in 10 30 50 75
    do
        python3 ar_topic.py $i $j > ar_t_ratio_${i}_white_${j}.txt
        mv featurestream.txt featurestream_ratio_${i}_white_${j}.txt
        mv trainstream.txt trainstream_ratio_${i}_white_${j}.txt
        mv teststream.txt teststream_ratio_${i}_white_${j}.txt
    done
done