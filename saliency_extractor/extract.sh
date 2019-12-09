#!/bin/bash

for folder in /mnt/Data/leite/FDD/train/saliency/*
do
  test="$folder/frame"
  for f in $test*
  do
    echo "$f"
    python saliency_extractor.py $f>> saliency_extraction.txt
  done
done

# for folder in /mnt/Data/leite/URFD/NotFalls/*
# do
  # test="$folder/frame"
  # for f in $test*
  # do
    # echo "$f"
    # python saliency_extractor.py $f
  # done
# done
