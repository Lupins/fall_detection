folder='/mnt/h5_files/fdd/test/*.h5'

for file in $folder;
do
  echo $file
  ln -s $file ./
done
