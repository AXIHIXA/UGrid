import argparse
import typing

import numpy as np
import os

import util


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str)
parser.add_argument('--shape', type=str, choices=['curve', 'square'])
parser.add_argument('--image_size', type=int)
parser.add_argument('--start_index', type=int)
parser.add_argument('--num_instances', type=int)


def generate_punched_random_curve_region(opt: argparse.Namespace) -> None:
    os.makedirs(opt.dataset_root, exist_ok=True)

    for i in range(opt.start_index, opt.start_index + opt.num_instances):
        bc_value: np.ndarray
        bc_mask: np.ndarray

        bc_value, bc_mask = util.gen_punched_random_curve_region(opt.image_size)

        data: np.ndarray = np.stack([bc_value, bc_mask])
        filename = os.path.join(opt.dataset_root, 'curve_{:06d}'.format(i))
        np.save(filename, data)


def generate_random_square_region(opt: argparse.Namespace) -> None:
    os.makedirs(opt.dataset_root, exist_ok=True)

    for i in range(opt.start_index, opt.start_index + opt.num_instances):
        bc_value: np.ndarray
        bc_mask: np.ndarray

        bc_value, bc_mask = util.gen_random_square_region(opt.image_size)

        data: np.ndarray = np.stack([bc_value, bc_mask])
        filename = os.path.join(opt.dataset_root, 'square_{:06d}'.format(i))
        np.save(filename, data)


def main() -> None:
    opt: argparse.Namespace = parser.parse_args()

    if opt.shape == 'curve':
        generate_punched_random_curve_region(opt)
    elif opt.shape == 'square':
        generate_random_square_region(opt)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
