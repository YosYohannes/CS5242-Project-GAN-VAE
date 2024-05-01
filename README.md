# Overview

This repo references the following:
- https://github.com/ritheshkumar95/pytorch-vqvae
- https://github.com/nerdyrodent/VQGAN-CLIP
- https://github.com/openai/CLIP
- https://github.com/CompVis/taming-transformers

### Training VQ-VAE with AFHQ
Run the following to train VQ-VAE:
```
python vqvae.py --dataset AFHQ --batch-size 128 --output-folder [path/to/output] --num-epochs 500 --beta 0.25
```

Run the following to train PixelCNN prior:
```
python pixelcnn_prior.py --dataset AFHQ --model [path/to/vq-vae/checkpoint] --output-folder [path/to/output] --num-epochs 500 --batch-size 128
```

### Training Custom VQGAN
To train VQGAN on custom `afhq` dataset go into `VQGAN-CLIP/taming-transformer` and read `Training on custom data`. Data pointer `train.txt` and `test.txt` has been added, pointing to `afhq` dataset. And `custom_vqgan.yaml` has been edited. There was a deprection of some sort with `from torch._six import string_classes` which was fixed. I had issues running with `trainer_config["distributed_backend"] = "ddp"` but was able to run it `trainer_config["distributed_backend"] = None`.

To run training, simply follow the command. This will train VQGAN from scratch.
```
python main.py --base configs/custom_vqgan.yaml -t True --gpus "0,"
or for two GPUs
python main.py --base configs/custom_vqgan.yaml -t True --gpus "0,1"
```

### Fine-tuning VQGAN
Go into `VQGAN-CLIP/taming-transformer`. Following this comment https://github.com/CompVis/taming-transformers/issues/107#issuecomment-927097409, checkpoint folder and `configs/model.yaml` has been added for fine-tuning from a pre-trained `vqgan_imagenet_f16_1024` from https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/. To run fine-tuning, run
```
python -m pytorch_lightning.utilities.upgrade_checkpoint --file logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt

python main.py -t True --gpus "0,1" --resume logs/vqgan_imagenet_f16_1024
```

### Running VQGAN-CLIP
Go into `VQGAN-CLIP` and read more under `Set up`. This repo simply points to `.yaml` and `.ckpt` files of desired VQGAN. We can then point to previously trained or fine-tuned result. Afterwhich, we can start generating images `python generate.py -p "A painting painting of a cat"`
