#!/bin/sh 

lambda=0.5
batch=1000
mu=0.5
for((i=50;i<100;i=$(($i+5))))
do
    lambda=`echo "scale=2; $i/100" | bc`
    for((batch=1000;batch<10000;batch=$(($batch+500))))
    do
        for((j=50;j<100;j=$(($j+5))))
        do
            mu=`echo "scale=2; $j/100" | bc`
            # echo $lambda $mu $batch
            nohup /usr/bin/python run_summarization.py --dataset_name=duc_2004 --lambda_val=$lambda --mu=$mu --batch=$batch > tmp/rw_$(lambda)_$(mu)_$(batch).out 2>&1 &
            sleep 10m
        done
    done
done