#!/bin/bash

for f in input/Coffee_room_01/*
do
  python frame_int_grad_saliency.py $f
done
