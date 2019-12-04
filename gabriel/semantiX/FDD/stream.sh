python streams_fextractor.py -data /mnt/Data/leite/FDD/train/ -class Falls NotFalls -streams pose -id FDD-train -ext .avi | tee pose_train.txt
echo 'Train pose is done' | mail -s '' guilherme.vieira.leite@gmail.com
# python streams_fextractor.py -data /mnt/Data/leite/FDD/train/ -class Falls NotFalls -streams ritmo -id FDD-train
# echo 'Train Rythmn is done' | mail -s '' guilherme.vieira.leite@gmail.com
# python streams_fextractor.py -data /mnt/Data/leite/FDD/train/ -class Falls NotFalls -streams saliency -id FDD-train -ext .avi | tee saliency_train.txt
# echo 'Train Saliency is done' | mail -s '' guilherme.vieira.leite@gmail.com
# python streams_fextractor.py -data /mnt/Data/leite/FDD/train/ -class Falls NotFalls -streams spatial -id FDD-train
# echo 'Train RGB is done' | mail -s '' guilherme.vieira.leite@gmail.com
# python streams_fextractor.py -data /mnt/Data/leite/FDD/train/ -class Falls NotFalls -streams temporal -id FDD-train -ext .avi | tee temporal_train.txt
# echo 'Train OF is done' | mail -s '' guilherme.vieira.leite@gmail.com

# python streams_fextractor.py -data /mnt/Data/leite/FDD/test/ -class Falls NotFalls -streams pose -id FDD-test -ext .avi
# echo 'Test pose is done' | mail -s '' guilherme.vieira.leite@gmail.com
# python streams_fextractor.py -data /mnt/Data/leite/FDD/test/ -class Falls NotFalls -streams ritmo -id FDD-test -ext .avi
# echo 'Test rythmn is done' | mail -s '' guilherme.vieira.leite@gmail.com
# python streams_fextractor.py -data /mnt/Data/leite/FDD/test/ -class Falls NotFalls -streams saliency -id FDD-test -ext .avi
# echo 'Test saliency is done' | mail -s '' guilherme.vieira.leite@gmail.com
# python streams_fextractor.py -data /mnt/Data/leite/FDD/test/ -class Falls NotFalls -streams spatial -id FDD-test -ext .avi
# echo 'Test RGB is done' | mail -s '' guilherme.vieira.leite@gmail.com
# python streams_fextractor.py -data /mnt/Data/leite/FDD/test/ -class Falls NotFalls -streams temporal -id FDD-test -ext .avi
# echo 'Test OF is done' | mail -s '' guilherme.vieira.leite@gmail.com

# echo 'Feature over FDD dataset is done' | mail -s 'FDD feature is over' guilherme.vieira.leite@gmail.com
