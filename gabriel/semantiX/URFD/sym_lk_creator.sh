folder='/mnt/h5_files/urfd/train/*.h5'

for file in $folder;
do
  echo $file
  ln -s $file ./
done

folder='/mnt/h5_files/urfd/test/*.h5'

for file in $folder;
do
  echo $file
  ln -s $file ./
done

folder='/mnt/h5_files/fdd/train/*.h5'

for file in $folder;
do
  echo $file
  ln -s $file ./
done

folder='/mnt/h5_files/fdd/test/*.h5'

for file in $folder;
do
  echo $file
  ln -s $file ./
done
