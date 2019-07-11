#!/bin/bash

for folder in /mnt/Data/leite/URFD/Falls/*
do
  test="$folder/frame"
  for f in $test*
  do
    echo "$f"
    python saliency_extractor.py $f
  done
done

for folder in /mnt/Data/leite/URFD/NotFalls/*
do
  test="$folder/frame"
  for f in $test*
  do
    echo "$f"
    python saliency_extractor.py $f
  done
done
