#!/bin/bash

for folder in /mnt/Data/leite/FDD/test/saliency/*
do

  python organize_saliency.py $folder '/mnt/Data/leite/FDD/test/NotFalls/'
  test="$folder/frame"

  for f in $test*
  do

    echo "$f"
    python saliency_extractor.py $f

  done
  echo 'Last file '$f | mail -s 'Test Folder '$folder guilherme.vieira.leite@gmail.com
done

for folder in /mnt/Data/leite/FDD/train/saliency/*
do

  python organize_saliency.py $folder '/mnt/Data/leite/FDD/train/NotFalls/'
  test="$folder/frame"

  for f in $test*
  do

    echo "$f"
    python saliency_extractor.py $f

  done
  echo 'Last file '$f | mail -s 'Train Folder '$folder guilherme.vieira.leite@gmail.com
done
