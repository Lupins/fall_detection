python streams_fextractor.py -data /mnt/Data/leite/URFD/train/ -class Falls NotFalls -streams pose -id URFD-train -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/URFD/train/ -class Falls NotFalls -streams ritmo -id URFD-train -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/URFD/train/ -class Falls NotFalls -streams saliency -id URFD-train -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/URFD/train/ -class Falls NotFalls -streams spatial -id URFD-train -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/URFD/train/ -class Falls NotFalls -streams temporal -id URFD-train -ext .avi

python streams_fextractor.py -data /mnt/Data/leite/URFD/test/ -class Falls NotFalls -streams pose -id URFD-test -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/URFD/test/ -class Falls NotFalls -streams ritmo -id URFD-test -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/URFD/test/ -class Falls NotFalls -streams saliency -id URFD-test -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/URFD/test/ -class Falls NotFalls -streams spatial -id URFD-test -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/URFD/test/ -class Falls NotFalls -streams temporal -id URFD-test -ext .avi

# echo 'Test pose is done' | mail -s '' guilherme.vieira.leite@gmail.com
