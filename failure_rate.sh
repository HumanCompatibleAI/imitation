#!/bin/bash
N=100
for i in $(seq 1 $N); do
    if ! pytest tests/algorithms/test_mce_irl.py -k test_mce_irl_reasonable_mdp; then failed=$(expr $failed + 1); fi
done
echo $failed / $N
