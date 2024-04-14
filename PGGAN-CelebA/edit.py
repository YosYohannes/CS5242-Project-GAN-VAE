# python3.7
"""Edits latent codes with respect to given boundary.

Basically, this file takes latent codes and a semantic boundary as inputs, and
then shows how the image synthesis will change if the latent codes is moved
towards the given boundary.
"""

import os.path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from utils.logger import setup_logger
from utils.manipulator import linear_interpolate


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Edit image synthesis with given semantic boundary.')
  parser.add_argument('-a', '--attribute', type=str, required=True,
                      help='Attribute to classify images on. (required)')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-i', '--input_latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
  parser.add_argument('-c', '--condition', type=str, default='',
                      help='Conditional arguement for boundary (optional)')
  parser.add_argument('-n', '--num', type=int, default=1,
                      help='Number of images for editing. This field will be '
                           'ignored if `input_latent_codes_path` is specified. '
                           '(default: 1)')
  parser.add_argument('--start_distance', type=float, default=-3.0,
                      help='Start point for manipulation in latent space. '
                           '(default: -3.0)')
  parser.add_argument('--end_distance', type=float, default=3.0,
                      help='End point for manipulation in latent space. '
                           '(default: 3.0)')
  parser.add_argument('--steps', type=int, default=10,
                      help='Number of steps for image editing. (default: 10)')

  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  output_dir = os.path.join(args.output_dir, args.attribute)
  if args.condition != '':
    output_dir = os.path.join(args.output_dir, args.attribute, args.condition)
  if os.path.exists(output_dir):
    output_dir = output_dir + '1'
  logger = setup_logger(output_dir, logger_name='generate_data')

  logger.info(f'Initializing generator.')
  model = PGGANGenerator('pggan_celebahq', logger)
  kwargs = {}

  logger.info(f'Preparing boundary.')
  boundary_path = os.path.join('boundaries', f'pggan_celebahq_{args.attribute}_boundary.npy')
  if args.condition != '':
    boundary_path = os.path.join('boundaries', f'pggan_celebahq_{args.attribute}_c_{args.condition}_boundary.npy')
  if not os.path.isfile(boundary_path):
    raise ValueError(f'Boundary `{boundary_path}` does not exist!')
  boundary = np.load(boundary_path)
  np.save(os.path.join(output_dir, 'boundary.npy'), boundary)

  logger.info(f'Preparing latent codes.')
  if os.path.isfile(args.input_latent_codes_path):
    logger.info(f'  Load latent codes from `{args.input_latent_codes_path}`.')
    latent_codes = np.load(args.input_latent_codes_path)
    latent_codes = model.preprocess(latent_codes, **kwargs)
  else:
    logger.info(f'  Sample latent codes randomly.')
    latent_codes = model.easy_sample(args.num, **kwargs)
  np.save(os.path.join(output_dir, 'latent_codes.npy'), latent_codes)
  total_num = latent_codes.shape[0]

  logger.info(f'Editing {total_num} samples.')
  for sample_id in tqdm(range(total_num), leave=False):
    interpolations = linear_interpolate(latent_codes[sample_id:sample_id + 1],
                                        boundary,
                                        start_distance=args.start_distance,
                                        end_distance=args.end_distance,
                                        steps=args.steps)
    interpolation_id = 0
    for interpolations_batch in model.get_batch_inputs(interpolations):
      outputs = model.easy_synthesize(interpolations_batch)
      for image in outputs['image']:
        save_path = os.path.join(output_dir,
                                 f'{sample_id:03d}_{interpolation_id:03d}.jpg')
        cv2.imwrite(save_path, image[:, :, ::-1])
        interpolation_id += 1
    assert interpolation_id == args.steps
    logger.debug(f'  Finished sample {sample_id:3d}.')
  logger.info(f'Successfully edited {total_num} samples.')


if __name__ == '__main__':
  main()
