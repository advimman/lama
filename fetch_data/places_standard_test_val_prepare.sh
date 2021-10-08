mkdir -p places_standard_dataset/original/test/
tar -xvf test_large.tar --transform='s/.*\///' -C places_standard_dataset/original/test/

mkdir -p places_standard_dataset/original/val/
tar -xvf val_large.tar --transform='s/.*\///' -C places_standard_dataset/original/val/
