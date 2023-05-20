## Image-to-Image Translation: From Line to Sketch

### Abstract
We investigate a specific task of image-to-image translation: line generation sketch task. We delve into two methodologies:  pix2pix and pixel2style2pixel. The pix2pix framework employs a conditional adversarial network, wherein a generator built upon "U-Net" architecture collaborates with a convolutional "PatchGAN" classifier serving as the discriminator. On the other hand, the pixel2style2pixel is based on a novel encoder network, that directly generate series of style vectors which are fed into a pretrained StyleGAN generator, forming the extended $W+$ latent space. We demonstrate that these two approaches are both effective at synthesizing portrait sketches from lines.
### Pix2Pix
#### Prepare dataset
Pix2pix's training requires paired data. Create folder `/path/to/data` with subdirectories `A` and `B`. `A` and `B` should each have their own subdirectories `train`, `test`. In `/path/to/data/A/train`, put training images in style A. In `/path/to/data/B/train`, put the corresponding images in style B. Repeat same for other data splits (`val`, `test`, etc).

Once the data is formatted this way, call:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```

This will combine each pair of images (A,B) into a single image file, ready for training.

#### Training/test options
Please see `options/train_options.py` and `options/base_options.py` for the training flags; see `options/test_options.py` and `options/base_options.py` for the test flags.

#### Training and test
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

### Pixel2Stype2Pixel



### Reference
[1] [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

[2] [Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation](https://arxiv.org/abs/2008.00951)

[3] [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

[4] [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [stylegan2](https://github.com/NVlabs/stylegan2) and [
pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
