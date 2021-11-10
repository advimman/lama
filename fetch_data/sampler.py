import os
import random

test_files_path = os.path.abspath('.') + '/places_standard_dataset/original/test/'
test_files = [test_files_path + image for image in os.listdir(test_files_path)]
print(f'found {len(test_files)} images in {test_files_path}')

random.shuffle(test_files)
test_files_random = test_files[0:2000]
#print(test_files_random[0:10])

list_of_random_test_files = os.path.abspath('.') \
+ '/places_standard_dataset/original/test_random_files.txt'

print(f'copying 100 random images to {list_of_random_test_files}')
with open(list_of_random_test_files, 'w') as fw:
    for filename in test_files_random:
        fw.write(filename+'\n')
print('...done')

# ----------------------------------------------------------------------------------


val_files_path = os.path.abspath('.') + '/places_standard_dataset/original/val/'
val_files = [val_files_path + image for image in os.listdir(val_files_path)]
print(f'found {len(val_files)} images in {val_files_path}')

random.shuffle(val_files)
val_files_random = val_files[0:100]

list_of_random_val_files = os.path.abspath('.') \
+ '/places_standard_dataset/original/val_random_files.txt'

print(f'copying 100 random images to {list_of_random_val_files}')
with open(list_of_random_val_files, 'w') as fw:
    for filename in val_files_random:
        fw.write(filename+'\n')
print('...done')      

