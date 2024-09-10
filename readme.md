# Multi-Source Music Generation with Latent Diffusion
This is the inference models of our paper MSLDM: Multi-Source Latent Diffusion for Music Generation. Our demo site is shown here: https://xzwy.github.io/MSLDMDemo/.

## To start this project, first
```bash
git clone https://github.com/XZWY/MSLDM
```

## enviroment
The environment for running our code can be installed using conda:
```bash
# Install environment
conda env create -f env.yaml

# Activate the environment
conda activate msdm
```

## slakh2100 dataset
### 1. download the complete dataset from https://github.com/gladia-research-group/multi-source-diffusion-models/blob/main/data/README.md
### 2, Move all the downloaded tar files inside msldm/data
### 3. Follow the data preparation instruction 3,4 in https://github.com/gladia-research-group/multi-source-diffusion-models/blob/main/data/README.md, but in your msldm/data directory
make sure that msldm/data looks like:
```
data/
 └───bass_22050
        └─── train
            └───Track00001.wav
            ...
        └─── validation
        └─── test
 └───drums_22050
 └───guitar_22050
 └───piano_22050
 └─── slakh2100/
       └─── train/
             └─── Track00001/
                   └─── bass.wav
                   └─── drums.wav
                   └─── guitar.wav
                   └─── piano.wav
            ...
      ...
```
# SourceVAE

## Inference to get the latent dataset (needed for msldm training)
```bash
cd SourceVAE/data
export PYTHONPATH=../../SourceVAE
python generate_dataset_slakh_latents.py --ckpt_path $ckpt_path --save_dir $save_dir --mode 'train' --device 'cuda:0' --batch_size 4 --n_workers 2
python generate_dataset_slakh_latents.py --ckpt_path $ckpt_path --save_dir $save_dir --mode 'validation' --device 'cuda:0' --batch_size 4 --n_workers 2
```
specify the checkpoint and download path in ckpt_path and save_dir, you can download the sourcevae checkpoint from here: https://uofi.box.com/s/as0yxoua68f5dcathvs8yi34far7k705.


##  Train SourceVAE
If you want to train your own SourceVAE follow the instructions below.
### generate dataset metadata
```python
python generate_slakh_dataset_metadata.py --mode train
python generate_slakh_dataset_metadata.py --mode validation
```
### train SourceVAE
```bash
bash ../start.sh
```
The logs and ckpts will be saved in SourceVAE/logfiles

## MSLDM
### training
> ⚠️ **NOTE:**  
> Before executing the training script, you have to change the **WANDB_\*** environment variables to match your 
personal or institutional account.

```bash
cd msldm
```
To train MSLDM (this should take about 8 GB of GPU memory):
```bash
bash start_msldm.sh # train msldm
```
To train MSLDM-Large (this should take about 33 GB of GPU memory):
```bash
bash start_msldm_large.sh # train msldm
```

### inference
the inference example is shown in ./inference.ipynb, the ckpt should be downloaded from https://uofi.box.com/s/z2qxbdsxravhdg1n95khz8um3olgeya3 and then saved in ./ckpts, the ./ckpt should look like:
```
ckpt/
 └───sourcevae_ckpt
 └───msldm.ckpt
 └───msldm_large.ckpt
```