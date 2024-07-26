mkdir -p places_standard_dataset/val/
mkdir -p places_standard_dataset/visual_test/


python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_occ_512.yaml \
/home/isaacfs/places_standard_dataset/val_hires/ \
/home/isaacfs/places_standard_dataset/val/ \
--occ_indir /home/isaacfs/occlusions_mask/original/test/test_large/ 

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_occ_512.yaml \
/home/isaacfsplaces_standard_dataset/visual_test_hires/ \
/home/isaacfs/places_standard_dataset/visual_test/ \
--occ_indir /home/isaacfs/occlusions_mask/original/val/val_large/