import typing

import torch


def l1_error(pred: torch.Tensor, gt: torch.Tensor, reduction: str = 'all') -> torch.Tensor:
    """
    L1 error beteen pred and gt.
    Note: For this project gt is a single image (optimization target)
    @param pred:      Size (batch, ...).
    @param gt:        Size (1, ...).
    @param reduction: Whether we apply mean reduction.
                      Choices: 'all', 'batch', 'none'
    @return:          Relative error of size
                      (1,) if reduction == 'all', or
                      (batch,) if reduction == 'batch', or
                      (batch, ...) if reduction == 'none'
    """
    l1: torch.Tensor = torch.abs(pred - gt)

    if reduction == 'all':
        return torch.mean(l1)
    elif reduction == 'batch':
        return torch.mean(l1, dim=l1.size()[1:])
    elif reduction == 'none':
        return l1
    else:
        raise ValueError


def relative_error(pred: torch.Tensor, gt: torch.Tensor, reduction: str = 'all') -> torch.Tensor:
    """
    Relative error beteen pred and gt.
    Note: For this project gt is a single image (optimization target)
    @param pred:      Size (batch, ...).
    @param gt:        Size (1, ...).
    @param reduction: Whether we apply mean reduction.
                      Choices: 'all', 'batch', 'none'
    @return:          Relative error of size
                      (1,) if reduction == 'all', or
                      (batch,) if reduction == 'batch', or
                      (batch, ...) if reduction == 'none'
    """
    numerator: torch.Tensor = torch.abs(pred - gt)

    gt_abs: torch.Tensor = torch.abs(gt)
    # gt_abs_mean: torch.Tensor = torch.mean(gt_abs)
    denominator: torch.Tensor = torch.where(gt != 0, gt_abs, 1)

    # Size (batch, ...)
    quotient: torch.Tensor = numerator / denominator

    if reduction == 'all':
        return torch.mean(quotient)
    elif reduction == 'batch':
        return torch.mean(quotient, dim=quotient.size()[1:])
    elif reduction == 'none':
        return quotient
    else:
        raise ValueError


def relative_vertex_difference_100_single_polygon(
        pred: torch.Tensor,
        gt: torch.Tensor,
        reduction: str = 'all'
) -> torch.Tensor:
    """
    Relative error beteen two set of 2D vertices pred and gt,
    which is defined as the delta-x and delta-y coordinates
    relative with respect to the x-span and y-span of the ground-truth polygon.
    @param pred:      Size (nVertices, 2).
    @param gt:        Size (nVertices, 2).
    @param reduction: Whether we apply mean reduction.
                      Choices: 'all', 'none'
    @return:          Relative error (%) of size
                      (1,) if reduction == 'all', or
                      (nVertices, 2) if reduction == 'none'
    """
    x_min, y_min = torch.amin(gt, 0)
    x_max, y_max = torch.amax(gt, 0)

    err: torch.Tensor = torch.abs(pred - gt).cpu() / torch.FloatTensor([[x_max - x_min, y_max - y_min]]) * 100

    if reduction == 'all':
        return torch.mean(err)
    elif reduction == 'none':
        return err
    else:
        raise ValueError


def relative_color_difference_100_single_polygon(
        pred: torch.Tensor,
        gt: torch.Tensor,
        reduction: str = 'all'
) -> torch.Tensor:
    """
    Relative error beteen two set of 1D colors pred and gt,
    which is defined as the delta-x and delta-y coordinates
    relative with respect to the x-span and y-span of the ground-truth polygon.
    @param pred:      Size (nVertices, 1).
    @param gt:        Size (nVertices, 1).
    @param reduction: Whether we apply mean reduction.
                      Choices: 'all', 'none'
    @return:          Relative error (%) of size
                      (1,) if reduction == 'all', or
                      (nVertices, 1) if reduction == 'none'
    """
    x_min = torch.amin(gt)
    x_max = torch.amax(gt)

    err: torch.Tensor = torch.abs(pred - gt).cpu() / (x_max - x_min).cpu() * 100

    if reduction == 'all':
        return torch.mean(err)
    elif reduction == 'none':
        return err
    else:
        raise ValueError


def relative_vertex_difference_100(
        pred_lst: typing.List[torch.Tensor],
        gt_lst: typing.List[torch.Tensor],
) -> torch.Tensor:
    err_lst: typing.List[torch.Tensor] = []

    for pred, gt in zip(pred_lst, gt_lst):
        err_lst.append(relative_vertex_difference_100_single_polygon(pred, gt, 'all'))

    return torch.mean(torch.tensor(err_lst))


def relative_color_difference_100(
        pred_lst: typing.List[torch.Tensor],
        gt_lst: typing.List[torch.Tensor],
) -> torch.Tensor:
    err_lst: typing.List[torch.Tensor] = []

    for pred, gt in zip(pred_lst, gt_lst):
        err_lst.append(relative_color_difference_100_single_polygon(pred, gt, 'all'))

    return torch.mean(torch.tensor(err_lst))