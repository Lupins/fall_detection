#!/bin/bash

for folder in /mnt/Data/URFD/temp-leite/Falls/*
do
  test="$folder/frame"
  for f in $test*
  do
    echo "$f"
    python frame_int_grad_saliency.py $f
  done
done
