folder='/mnt/h5_files/urfd/train/*.h5'

for file in $folder;
do
  echo $file
  ln -s $file ./
done
