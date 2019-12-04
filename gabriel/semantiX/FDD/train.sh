dat=$(date +"%F_%H:%M:%S")

ep='500'
lr='0.0001'
batch='1024'
data='FDD'

# python3 train.py -actions cross-train -streams temporal pose spatial ritmo saliency -class Falls NotFalls -thresh 0.5 -w0 1 -ep 500 -batch_norm True -mini_batch 0 -id URFD_test -nsplits 5 -lr 0.0001 | tee train_500ep_all.txt

python train_new.py -actions cross-train -streams temporal pose spatial ritmo -class Falls NotFalls -ep $ep -lr $lr -w0 1 -mini_batch $batch -batch_norm True -fold_norm 2 -kfold video -nsplits 5 -id ${data}-train | tee ${dat}_train_${data}_all-streams_ep_${ep}_lr_${lr}_batch_${batch}.txt

# echo 'Training over FDD dataset is done' | mail -s 'FDD Training is over' guilherme.vieira.leite@gmail.com
