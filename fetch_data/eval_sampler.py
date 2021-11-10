import os
import random


val_files_path = os.path.abspath('.') + '/places_standard_dataset/original/val/'
val_files      = [val_files_path + image for image in os.listdir(val_files_path)]

print(f'found {len(val_files)} images in {val_files_path}')

random.shuffle(val_files)
val_files_random = val_files[0:2000]

list_of_random_val_files = os.path.abspath('.') \
+ '/places_standard_dataset/original/eval_random_files.txt'

print(f'copying 2000 random images to {list_of_random_val_files}')
with open(list_of_random_val_files, 'w') as fw:
    for filename in val_files_random:
        fw.write(filename+'\n')
print('...done')      

