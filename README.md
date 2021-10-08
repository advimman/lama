# Resolution-robust Large Mask Inpainting with Fourier Convolutions

Official implementation

by Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, 
Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, Victor Lempitsky.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saic-mdal/lama/blob/master//colab/LaMa_inpainting.ipynb)

[[Project Page](https://saic-mdal.github.io/lama/)] [[ArXiv](https://arxiv.org/abs/2109.07161)] [[Supplementary](https://ashukha.com/projects/lama_21/lama_supmat_2021.pdf)]

**TODO** Try in Colab

<p align="center">
  <img src="https://raw.githubusercontent.com/senya-ashukha/senya-ashukha.github.io/master/projects/lama_21/ezgif-4-0db51df695a8_hr.gif" />
</p>

# Environment setup

Clone the repo:
`git clone git@github.sec.samsung.net:UREP/inpainting-lama.git`

There are three enviroment options:

1. Python virtualenv:

    ```
    virtualenv inpenv --python=/usr/bin/python3
    source inpenv/bin/activate
    pip install torch==1.8.0 torchvision==0.9.0
    
    cd inpainting-lama
    pip install -r requirements.txt 
    ```

2. Conda
    
    ```
    % Install conda for Linux, for other OS download miniconda at https://docs.conda.io/en/latest/miniconda.html
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    $HOME/miniconda/bin/conda init bash

    cd inpainting-lama
    conda env create -f conda_env.yml
    conda activate lama
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
    pip install pytorch-lightning==1.2.9
    ```
 
3. Docker: No actions are needed 🎉.

# Inference <a name="prediction"></a>

Run
```
cd lama-inpainting
export TORCH_HOME=$(pwd) && export PYTHONPATH=.
```

**1. Download pretrained models**

Install tool for yandex disk link extraction:

```
pip3 install wldhx.yadisk-direct
```

Best model (Places):
    
```    
curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
unzip big-lama.zip
```

All models (Places):

```
curl -L $(yadisk-direct https://disk.yandex.ru/d/AmdeG-bIjmvSug) -o lama-models.zip
unzip lama-models.zip
```

**2. Prepare images and masks**

Download test images:

```
curl -L $(yadisk-direct https://disk.yandex.ru/d/xKQJZeVRk5vLlQ) -o LaMa_test_images.zip
unzip LaMa_test_images.zip
```
<details>
 <summary>OR prepare your data:</summary>
1) Create masks named as `[images_name]_maskXXX[image_suffix]`, put images and masks in the same folder. 

- You can use the [script](#test_datasets) for random masks generation. 
- Check the format of the files:
    ```    
    image1_mask001.png
    image1.png
    image2_mask001.png
    image2.png
    ```

2) Specify `image_suffix`, e.g. `.png` or `.jpg` or `_input.jpg` in `configs/prediction/default.yaml`.

</details>


**3. Predict**

On the host machine:

    python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output

**OR** in the docker
  
The following command will pull the docker image from Docker Hub and execute the prediction script
```
bash docker/2_predict.sh $(pwd)/big-lama $(pwd)/LaMa_test_images $(pwd)/output device=cpu
```
Docker cuda: TODO

# Train and Eval

⚠️ Warning: is not fully tested yet, e.g. did not re-training after refactoring ⚠️


Make sure you run:

```
cd lama-inpainting
export TORCH_HOME=$(pwd) && export PYTHONPATH=.
```

Then download models for _perceptual loss_:

    mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
    wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth


## Places
On the host machine:

    # Download data from http://places2.csail.mit.edu/download.html
    # Places365-Standard: Train(105GB)/Test(19GB)/Val(2.1GB) from High-resolution images section
    wget http://data.csail.mit.edu/places/places365/train_large_places365standard.tar
    wget http://data.csail.mit.edu/places/places365/val_large.tar
    wget http://data.csail.mit.edu/places/places365/test_large.tar

    # Unpack and etc.
    bash fetch_data/places_standard_train_prepare.sh
    bash fetch_data/places_standard_test_val_prepare.sh
    bash fetch_data/places_standard_evaluation_prepare_data.sh
    
    # Sample images for test and viz at the end of epoch
    bash fetch_data/places_standard_test_val_sample.sh
    bash fetch_data/places_standard_test_val_gen_masks.sh

    # Run training
    # You can change bs with data.batch_size=10
    python bin/train.py -cn lama-fourier location=places_standard
    
    # Infer model on thick/thin/medium masks in 256 and 512 and run evaluation 
    # like this:
    python3 bin/predict.py \
    model.path=$(pwd)/experiments/<user>_<date:time>_lama-fourier_/ \
    indir=$(pwd)/places_standard_dataset/evaluation/random_thick_512/ \
    outdir=$(pwd)/inference/random_thick_512 model.checkpoint=last.ckpt

    python3 bin/evaluate_predicts.py \
    $(pwd)/configs/eval_2gpu.yaml \
    $(pwd)/places_standard_dataset/evaluation/random_thick_512/ \
    $(pwd)/inference/random_thick_512 $(pwd)/inference/random_thick_512_metrics.csv

    
    
Docker: TODO
    
## CelebA
On the host machine:

    TODO: download & prepare 
    TODO: trian
    TODO: eval
    
    
Docker: TODO

## Places Challenge 

On the host machine:

    # This script downloads multiple .tar files in parallel and unpacks them
    # Places365-Challenge: Train(476GB) from High-resolution images (to train Big-Lama) 
    bash places_challenge_train_download.sh
    
    TODO: prepare
    TODO: train 
    TODO: eval
      
Docker: TODO

## Create your own data
On the host machine:

Explain explain explain

    TODO: format
    TODO: configs 
    TODO: run training
    TODO: run eval
    
**OR** in the docker:

    TODO: trian
    TODO: eval
    
# Hints

### Generate different kinds of masks
The following command will execute script that generates random masks.

    bash docker/1_generate_masks_from_raw_images.sh \
        configs/data_gen/random_medium_512.yaml \
        /directory_with_input_images \
        /directory_where_to_store_images_and_masks \
        --ext png

The test data generation command stores images in the format,
which is suitable for [prediction](#prediction).

The table below describes which configs we used to generated different test sets from the paper.
Note that we *do not fix random seed*, so the results will be a bit different each time.

|        | Places 512x512         | CelebA 256x256         |
|--------|------------------------|------------------------|
| Narrow | random_thin_512.yaml   | random_thin_256.yaml   |
| Medium | random_medium_512.yaml | random_medium_256.yaml |
| Wide   | random_thick_512.yaml  | random_thick_256.yaml  |

Feel free to change config path (argument #1) to any other config in `configs/data_gen` 
or adjust config files themselves.

### Override parameters in configs
Also you can override parameters in config like this:

    python3 bin/train.py -cn <config> data.batch_size=10 run_title=my-title

Where .yaml file extention is omitted

### Models options 
Config names for models from paper (substitude into the training command): 

    * big-lama
    * big-lama-regular
    * lama-fourier
    * lama-regular
    * lama_small_train_masks

Which are seated in configs/training/folder

### Training time

TODO

### Our training logs

The training logs are avalible at [https://disk.yandex.ru/d/9Bt1wNSDS4jDkQ](https://disk.yandex.ru/d/9Bt1wNSDS4jDkQ).
TODO: IPython with tables?

-----
## Acknowledgments

* Segmentation code and models if form [CSAILVision](https://github.com/CSAILVision/semantic-segmentation-pytorch).
* LPIPS metric is from [richzhang](https://github.com/richzhang/PerceptualSimilarity)
* SSIM is from [Po-Hsun-Su](https://github.com/Po-Hsun-Su/pytorch-ssim)
* FID is from [mseitzer](https://github.com/mseitzer/pytorch-fid)
