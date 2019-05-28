#!/bin/bash

for folder in fdd/NotFalls/*
do
  test="$folder/frame"
  for f in $test*
  do
    echo "$f"
    python frame_int_grad_saliency.py $f
  done
done
