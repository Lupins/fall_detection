python streams_fextractor.py -data /mnt/Data/leite/FDD/train/ -class Falls NotFalls -streams pose -id FDD-train -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/FDD/train/ -class Falls NotFalls -streams ritmo -id FDD-train -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/FDD/train/ -class Falls NotFalls -streams saliency -id FDD-train -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/FDD/train/ -class Falls NotFalls -streams spatial -id FDD-train -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/FDD/train/ -class Falls NotFalls -streams temporal -id FDD-train -ext .avi

python streams_fextractor.py -data /mnt/Data/leite/FDD/test/ -class Falls NotFalls -streams pose -id FDD-test -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/FDD/test/ -class Falls NotFalls -streams ritmo -id FDD-test -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/FDD/test/ -class Falls NotFalls -streams saliency -id FDD-test -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/FDD/test/ -class Falls NotFalls -streams spatial -id FDD-test -ext .avi
python streams_fextractor.py -data /mnt/Data/leite/FDD/test/ -class Falls NotFalls -streams temporal -id FDD-test -ext .avi

# echo 'Feature over FDD dataset is done' | mail -s 'FDD feature is over' guilherme.vieira.leite@gmail.com
