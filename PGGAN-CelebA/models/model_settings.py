# python3.7
"""Contains basic configurations for models used in this project.

Please download the public released models from the following two repositories
OR train your own models, and then put them into `pretrain` folder.

ProgressiveGAN: https://github.com/tkarras/progressive_growing_of_gans

NOTE: Any new model should be registered in `MODEL_POOL` before using.
"""

import os.path

BASE_DIR = os.path.dirname(os.path.relpath(__file__))

MODEL_DIR = BASE_DIR + '/pretrain'

MODEL_POOL = {
    'pggan_celebahq': {
        'tf_model_path': MODEL_DIR + '/karras2018iclr-celebahq-1024x1024.pkl',
        'model_path': MODEL_DIR + '/pggan_celebahq.pth',
        'gan_type': 'pggan',
        'dataset_name': 'celebahq',
        'latent_space_dim': 512,
        'resolution': 1024,
        'min_val': -1.0,
        'max_val': 1.0,
        'output_channels': 3,
        'channel_order': 'RGB',
        'fused_scale': False,
    },
}

# Settings for model running.
USE_CUDA = True

MAX_IMAGES_ON_DEVICE = 4
