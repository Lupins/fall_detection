'''
Splits the files/folders located in argv[1] given the proportion in argv[2]
'''
import os
import sys
import random

# argv[1]: folder with files/folders
# argv[2]: proportion of the validation (e.g. argv[2] = 80 will split 80% to
# validation and 20% to test)
def main(class_folder, proportion):
    
    os.chdir(class_folder)
    files = os.listdir('./')
    files.sort()
    universe = len(files)
    
    try:
        os.mkdir('../test')
    except:
        print('Folder: test already exists, not creating it again')

    n_test = int(universe * ((100 - proportion) / 100))
    idx_list = random.sample(range(universe), n_test)

    print('Moving these files into test folder:')
    for i in idx_list:
        print(files[i])
        os.rename(files[i], '../test/' + files[i])

main(sys.argv[1], int(sys.argv[2]))
