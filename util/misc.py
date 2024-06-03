import argparse
import os
import typing

import cv2
import matplotlib.pyplot as plt
import numpy as np


def merge_namespace(n1: argparse.Namespace, n2: argparse.Namespace) -> argparse.Namespace:
    """
    Merge two namespaces.
    In the scene of a conflict, n2 shall prevail.
    :param n1:
    :param n2:
    :return: merged namespace
    """
    opt = argparse.Namespace(**vars(n1))

    for k, v in vars(n2).items():
        setattr(opt, k, v)

    return opt


def plt_subplot(dic: typing.Dict[str, np.ndarray],
                suptitle: typing.Optional[str] = None,
                unit_size: int = 5,
                show: bool = True,
                dump: typing.Optional[str] = None,
                dpi: typing.Optional[int] = None,
                show_axis: bool = True) -> None:
    fig = plt.figure(figsize=(unit_size * len(dic), unit_size), dpi=dpi)

    if suptitle:
        plt.suptitle(suptitle)

    for i, (k, v) in enumerate(dic.items()):
        plt.subplot(1, len(dic), i + 1)
        plt.axis(show_axis)
        plt.title(k)

        if v is not None:
            plt.imshow(v.squeeze())

        plt.colorbar()

    if dump is not None:
        os.makedirs(dump[:dump.rfind('/')], exist_ok=True)
        plt.savefig(dump, bbox_inches='tight')

    # show must follow savefig otherwise the saved image would be blank
    if show:
        plt.show()

    plt.close(fig)


def plt_dump(dic: typing.Dict[str, np.ndarray],
             unit_size: int = 5,
             colorbar: bool = False,
             dump_dir: typing.Optional[str] = None,
             dpi: int = 100) -> None:
    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)

    for i, (k, v) in enumerate(dic.items()):
        if v is not None:
            fig = plt.figure(figsize=(unit_size * len(dic), unit_size), dpi=dpi)
            plt.axis('off')
            plt.imshow(v.squeeze())

            if colorbar:
                plt.colorbar()

            plt.savefig(os.path.join(dump_dir, k) + '.png', bbox_inches='tight')
            plt.close(fig)


def get_number_of_files(path: str) -> int:
    cnt: int = 0

    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            cnt += 1

    return cnt

