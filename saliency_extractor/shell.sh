#!/bin/bash
for f in input/*
do
  python int_grad_saliency.py $f | cut -f2 -d"/"
done
