# VGAN
PyTorch(v1.7.0) implementation of [Variational Discriminator Bottleneck: improving Imitation Learning, Inverse RL, and GANs by constraining Information Flow](https://arxiv.org/abs/1810.00821) for [Deep Learning and GANs WASP course '21](https://internal.wasp-sweden.org/graduate-school/wasp-graduate-school-courses/deep-learning-and-gans/).

It has been tested on both RGB and grayscale datatypes through CIFAR-10 and FER-2013 datasets.

## Usage

Run train [``train.py``](https://github.com/tmralmeida/VGAN/blob/main/train.py) with the respective options:

```
python train.py [-h] [--dataset  {CIFAR-10, FER-2013}] 
                [--model {VGAN,VGAN-GP}] [--batch_size BATCH_SIZE]
                [--num_workers NUM_WORKERS] [--epochs EPOCHS]
                [--lr_gen LR_GEN] [--lr_disc LR_DISC]
                [--ic IC] [--beta BETA]           
                [--alpha ALPHA] [--save_dir SAVE_DIR]   
                [--nimgs_save NIMGS_SAVE]                                        
```

For help on the optional arguments run: ``python train.py -h``


### Running: Training

```
python train.py --dataset CIFAR-10 --batch_size 128 --epochs 20
```