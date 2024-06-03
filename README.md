# UGrid: An Efficient-And-Rigorous Neural Multigrid Solver for Linear PDEs

This repository is the official implementation of our ICML 2024 paper named

***UGrid: An Efficient-And-Rigorous Neural Multigrid Solver for Linear PDEs***.

## Data Generation

To generate the dataset, run this command:

```bash
bash ./script/generate.sh
```

Please modify `generate.sh` to generate the `train`, `evaluate` and `test` datasets of the desired size. 

## Training

To train the model(s) in the paper, run this command:

```bash
bash ./script/train.sh
```

## Evaluation and Testing

To re-produce the testing results of UGrid, please run this command: 

```bash
bash ./script/test.sh
```

To compare with 
[AMGCL](https://github.com/ddemidov/amgcl) and [NVIDIA AmgX](https://developer.nvidia.com/amgx), 
please first compile the Python bindings for AMGCL and AmgX (see `./comparasion/cpmg/`), 
then run the following command:

```bash
bash ./script/compare.sh
```

To compare with [(Hsieh et al., 2019)](https://openreview.net/forum?id=rklaWn0qK7), 
please refer to [their offical repository](https://github.com/ermongroup/Neural-PDE-Solver). 

## Pre-trained Models

Self-contained in `var/checkpoint/22/`. 


