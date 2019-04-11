#!/bin/bash

for f in input/*
do
  python frame_int_grad_saliency.py $f
done
