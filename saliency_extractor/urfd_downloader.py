import subprocess
import os

os.chdir('input')

# Fall sequences
for i in range(1, 31):

    file_name = 'fall-' + str(i).zfill(2) + '-cam0.mp4'

    return_code = subprocess.call('wget http://fenix.univ.rzeszow.pl/~mkepski/ds/data/' + file_name + ' -O ' + file_name, shell = True)

    file_name = 'fall-' + str(i).zfill(2) + '-cam1.mp4'

    return_code = subprocess.call('wget http://fenix.univ.rzeszow.pl/~mkepski/ds/data/' + file_name + ' -O ' + file_name, shell = True)

# ADL sequences
for i in range(1, 41):


    file_name = 'adl-' + str(i).zfill(2) + '-cam0.mp4'

    return_code = subprocess.call('wget http://fenix.univ.rzeszow.pl/~mkepski/ds/data/' + file_name + ' -O ' + file_name, shell = True)
