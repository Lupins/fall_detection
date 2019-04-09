#!/bin/bash
#for f in input/*
#do
  #python int_grad_saliency.py $f | cut -f2 -d"/"
#done

#python int_grad_saliency.py fall-07-cam0.mp4

for f in input/*
do
  python int_grad_saliency.py $f
done
