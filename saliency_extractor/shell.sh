#!/bin/bash
<<<<<<< HEAD

for f in input/*
#for f in input/*
#do
  #python int_grad_saliency.py $f | cut -f2 -d"/"
#done

#python int_grad_saliency.py fall-07-cam0.mp4

for f in input/*
do
  python int_grad_saliency.py $f
>>>>>>> 66c7e669d4f5d6235c5f6b6a32ee5f1cfdabfebf
done
