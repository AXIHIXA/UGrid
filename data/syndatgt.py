from .syndat import SynDat

import numpy as np
import torch


class SynDatGt(SynDat):
    def __init__(self, dataset_root: str):
        super().__init__(dataset_root)

    def __getitem__(self, index: int):
        data: np.ndarray = np.load(self.instance_path_lst[index])
        bc_value, bc_mask, gt = data
        bc_value: torch.Tensor = torch.from_numpy(bc_value).float().unsqueeze(0)
        bc_mask: torch.Tensor = torch.from_numpy(bc_mask).float().unsqueeze(0)
        gt: torch.Tensor = torch.from_numpy(gt).float().unsqueeze(0)
        return {'bc_value': bc_value,
                'bc_mask':  bc_mask,
                'gt':       gt}

    def __len__(self):
        return len(self.instance_path_lst) // 4
