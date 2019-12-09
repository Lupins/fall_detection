#python streams_fextractor.py -data /workspace/FDD-zip/ -class Falls NotFalls -streams saliency -id FDD -ext .avi
#python streams_fextractor.py -data /workspace/URFD-zip/ -class Falls NotFalls -streams saliency -id URFD -ext .avi
python streams_fextractor.py -data /mnt/Data/FDD/ -class Falls NotFalls -streams temporal saliency spatial -id FDD -ext .avi
