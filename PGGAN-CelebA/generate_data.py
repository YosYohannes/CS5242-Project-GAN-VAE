# python3.7
"""Generates a collection of images with specified model.

Commonly, this file is used for data preparation. More specifically, before
exploring the hidden semantics from the latent space, user need to prepare a
collection of images. These images can be used for further attribute prediction.
In this way, it is able to build a relationship between input latent codes and
the corresponding attribute scores.
"""

import os.path
import argparse
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from utils.logger import setup_logger


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Generate images with given model.')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-i', '--latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
  parser.add_argument('-n', '--num', type=int, default=1,
                      help='Number of images to generate. This field will be '
                           'ignored if `latent_codes_path` is specified. '
                           '(default: 1)')

  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  logger = setup_logger(args.output_dir, logger_name='generate_data')

  logger.info(f'Initializing generator.')
  gan_type = MODEL_POOL['pggan_celebahq']['gan_type']
  model = PGGANGenerator('pggan_celebahq', logger)
  kwargs = {}

  logger.info(f'Preparing latent codes.')
  if os.path.isfile(args.latent_codes_path):
    logger.info(f'  Load latent codes from `{args.latent_codes_path}`.')
    latent_codes = np.load(args.latent_codes_path)
    latent_codes = model.preprocess(latent_codes, **kwargs)
  else:
    logger.info(f'  Sample latent codes randomly.')
    latent_codes = model.easy_sample(args.num, **kwargs)
  total_num = latent_codes.shape[0]

  logger.info(f'Generating {total_num} samples.')
  results = defaultdict(list)
  pbar = tqdm(total=total_num, leave=False)
  for latent_codes_batch in model.get_batch_inputs(latent_codes):
    outputs = model.easy_synthesize(latent_codes_batch)
    for key, val in outputs.items():
      if key == 'image':
        for image in val:
          save_path = os.path.join(args.output_dir, f'{pbar.n:06d}.jpg')
          cv2.imwrite(save_path, image[:, :, ::-1])
          pbar.update(1)
      else:
        results[key].append(val)
    if 'image' not in outputs:
      pbar.update(latent_codes_batch.shape[0])
    if pbar.n % 1000 == 0 or pbar.n == total_num:
      logger.debug(f'  Finish {pbar.n:6d} samples.')
  pbar.close()

  logger.info(f'Saving results.')
  for key, val in results.items():
    save_path = os.path.join(args.output_dir, f'{key}.npy')
    np.save(save_path, np.concatenate(val, axis=0))


if __name__ == '__main__':
  main()
