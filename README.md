# Image-to-Image Translation: From Line to Sketch

## Abstract
We investigate a specific task of image-to-image translation: line generation sketch task. We delve into two methodologies:  pix2pix and pixel2style2pixel. The pix2pix framework employs a conditional adversarial network, wherein a generator built upon "U-Net" architecture collaborates with a convolutional "PatchGAN" classifier serving as the discriminator. On the other hand, the pixel2style2pixel is based on a novel encoder network, that directly generate series of style vectors which are fed into a pretrained StyleGAN generator, forming the extended $W+$ latent space. We demonstrate that these two approaches are both effective at synthesizing portrait sketches from lines.

## Pix2Pix
### Requirements
See `pix2pix/requirements.txt`.
### Prepare dataset
Pix2pix's training requires paired data. Create folder `/path/to/data` with subdirectories `A` and `B`. `A` and `B` should each have their own subdirectories `train`, `test`. In `/path/to/data/A/train`, put training images in style A. In `/path/to/data/B/train`, put the corresponding images in style B. Repeat same for other data splits (`val`, `test`, etc).

Once the data is formatted this way, call:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```

This will combine each pair of images (A,B) into a single image file, ready for training.

### Training/test options
Please see `options/train_options.py` and `options/base_options.py` for the training flags; see `options/test_options.py` and `options/base_options.py` for the test flags.

### Training and test
- Train a model:
```bash
python train.py --dataroot /path/to/data --name pix2pix --model pix2pix --direction AtoB
```

- Test the model:
```bash
python test.py --dataroot /path/to/data --name pix2pix --model pix2pix --direction AtoB
```
- The test results will be saved to a file: `./results/pix2pix/test_latest/index.html`. 
- You can find more scripts at `scripts` directory.

## Pixel2Stype2Pixel
### Requirements
See `pixel2style2pixel/cog.yaml`.
### Training
#### Preparing your Data
- Currently, we provide support for numerous datasets and experiments (encoding, frontalization, etc.).
    - Refer to `configs/paths_config.py` to define the necessary data paths and model paths for training and evaluation. 
    - Refer to `configs/transforms_config.py` for the transforms defined for each dataset/experiment. 
    - Finally, refer to `configs/data_configs.py` for the source/target data paths for the train and test sets
      as well as the transforms.
- If you wish to experiment with your own dataset, you can simply make the necessary adjustments in 
    1. `data_configs.py` to define your data paths.
    2. `transforms_configs.py` to define your own data transforms.
    
As an example, assume we wish to run encoding using ffhq (`dataset_type=ffhq_encode`). 
We first go to `configs/paths_config.py` and define:
``` 
dataset_paths = {
    'ffhq': '/path/to/ffhq/images256x256'
    'celeba_test': '/path/to/CelebAMask-HQ/test_img',
}
```
The transforms for the experiment are defined in the class `EncodeTransforms` in `configs/transforms_config.py`.   
Finally, in `configs/data_configs.py`, we define:
``` 
DATASETS = {
   'ffhq_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['ffhq'],
        'train_target_root': dataset_paths['ffhq'],
        'test_source_root': dataset_paths['celeba_test'],
        'test_target_root': dataset_paths['celeba_test'],
    },
}
``` 
When defining our datasets, we will take the values in the above dictionary.


#### Training pSp
The main training script can be found in `scripts/train.py`.   
Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs.  
Additionally, if you have tensorboard installed, you can visualize tensorboard logs in `opts.exp_dir/logs`.


##### Line to Sketch
```bash
python scripts/train.py \
--dataset_type=celebs_sketch_to_face \
--exp_dir=finetune_5_8 \
--workers=8 \
--batch_size=4 \
--test_batch_size=4 \
--test_workers=8 \
--val_interval=100 \  
--save_interval=100 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0 \
--w_norm_lambda=0.005 \
--label_nc=1 \
--input_nc=1 \
--checkpoint_path PATH_TO_MODEL \
--max_steps 5000 
```


### Testing
#### Inference
Having trained your model, you can use `scripts/inference.py` to apply the model on a set of images.   
For example, 
```bash
python scripts/inference.py \
--exp_dir test_5_7 \
--checkpoint_path finetune_5_7/checkpoints/best_model.pt \
--data_path ../CGI-PSG-Training_Set/train_line/ \
--resize_output
```
The model can be downloaded at https://jbox.sjtu.edu.cn/l/A1BMJJ

### Repository structure
| Path | Description
| :--- | :---
| pixel2style2pixel | Repository root folder
| &boxvr;&nbsp; configs | Folder containing configs defining model/data paths and data transforms
| &boxvr;&nbsp; criteria | Folder containing various loss criterias for training
| &boxvr;&nbsp; datasets | Folder with various dataset objects and augmentations
| &boxvr;&nbsp; environment | Folder containing Anaconda environment used in our experiments
| &boxvr; models | Folder containting all the models and training objects
| &boxv;&nbsp; &boxvr;&nbsp; encoders | Folder containing our pSp encoder architecture implementation and ArcFace encoder implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
| &boxv;&nbsp; &boxvr;&nbsp; mtcnn | MTCNN implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
| &boxv;&nbsp; &boxvr;&nbsp; stylegan2 | StyleGAN2 model from [rosinality](https://github.com/rosinality/stylegan2-pytorch)
| &boxv;&nbsp; &boxur;&nbsp; psp | Implementation of our pSp framework
| &boxvr;&nbsp; notebook | Folder with jupyter notebook containing pSp inference playground
| &boxvr;&nbsp; options | Folder with training and test command-line options
| &boxvr;&nbsp; scripts | Folder with running scripts for training and inference
| &boxvr;&nbsp; training | Folder with main training logic and Ranger implementation from [lessw2020](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
| &boxvr;&nbsp; utils | Folder with various utility functions

## StyleGAN 2 in PyTorch

### Requirements

I have tested on:

- PyTorch 1.3.1
- CUDA 10.1/10.2

### Usage

First create lmdb datasets:
```bash
python prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH
```

This will convert images to jpeg and pre-resizes it. This implementation does not use progressive growing, but you can create multiple resolution datasets using size arguments with comma separated lists, for the cases that you want to try another resolutions later.

#### Training
```bash
python train.py --size 256 --ckpt checkpoint/face.pt  --n_sample 25 --batch 8 --iter 10000 --wandb ./data_lmdb/
```
`train.py` supports Weights & Biases logging. If you want to use it, add --wandb arguments to the script.

#### Generate samples
```bash
python generate.py --sample N_FACES --pics N_PICS --ckpt PATH_CHECKPOINT
```
You should change your size (--size 256 for example) if you train with another dimension.

The model can be downloaded at https://jbox.sjtu.edu.cn/l/x1NgwP

#### Project images to latent spaces
```bash
python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] FILE1 FILE2 ...
```
## Reference
[1] [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

[2] [Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation](https://arxiv.org/abs/2008.00951)

[3] [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

[4] [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [stylegan2](https://github.com/NVlabs/stylegan2) and [
pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
