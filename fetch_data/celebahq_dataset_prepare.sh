mkdir celeba-hq-dataset

unzip data256x256.zip -d celeba-hq-dataset/

# Reindex
for i in `echo {00001..30000}`
do
    mv 'celeba-hq-dataset/data256x256/'$i'.jpg' 'celeba-hq-dataset/data256x256/'$[10#$i - 1]'.jpg'
done


# train/test/vis split
cat fetch_data/train_shuffled.flist | shuf > celeba-hq-dataset/temp_train_shuffled.flist
cat celeba-hq-dataset/temp_train_shuffled.flist | head -n 2000 > celeba-hq-dataset/val_shuffled.flist
cat celeba-hq-dataset/temp_train_shuffled.flist | tail -n +2001 > celeba-hq-dataset/train_shuffled.flist

cat fetch_data/val_shuffled.flist > celeba-hq-dataset/visual_test_shuffled.flist


# Split
for mode in train \
    val \
    visual_test
do
    mkdir celeba-hq-dataset/$mode"_256"
    cat celeba-hq-dataset/$mode"_shuffled.flist" | xargs -I {} mv celeba-hq-dataset/data256x256/{} celeba-hq-dataset/$mode"_256/"
done


# create location config celeba.yaml
PWD=$(pwd)
DATASET=${PWD}/celeba-hq-dataset
CELEBA=${PWD}/configs/training/location/celeba.yaml

touch $CELEBA
echo "# @package _group_" >> $CELEBA
echo "data_root_dir: ${DATASET}/" >> $CELEBA
echo "out_root_dir: ${PWD}/experiments/" >> $CELEBA
echo "tb_dir: ${PWD}/tb_logs/" >> $CELEBA
echo "pretrained_models: ${PWD}/" >> $CELEBA
