import typing
import os

import torch
import torch.utils.data.dataloader

from data import SynDat
import util


class Trainer:
    def __init__(self,
                 experienment_name: str, experienment_checkpoint_path: str, device: torch.device,
                 model, logger,
                 optimizer: str, scheduler: str, initial_lr: float, lambda_1: float, lambda_2: float,
                 start_epoch: int, max_epoch: int, save_every: int, evaluate_every: int,
                 dataset_root: str, num_workers: int, batch_size: int,):
        self.experienment_name: str = experienment_name
        self.experienment_checkpoint_path: str = experienment_checkpoint_path

        self.model = model
        self.logger = logger

        self.device: torch.device = device

        self.initial_lr: float = initial_lr
        self.lambda_1: float = lambda_1
        self.lambda_2: float = lambda_2

        self.start_epoch: int = start_epoch
        self.max_epoch: int = max_epoch
        self.save_every: int = save_every
        self.evaluate_every: int = evaluate_every

        self.dataset_root: str = dataset_root
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.initial_lr)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.initial_lr)
        else:
            raise NotImplementedError

        if 'step' in scheduler:
            _, step_size, gamma = scheduler
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, int(step_size), float(gamma))
        else:
            raise NotImplementedError

        train_dataset_path: str = os.path.join(dataset_root, 'train')
        self.train_dataset = SynDat(train_dataset_path)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        num_workers=self.num_workers,
                                                        batch_size=batch_size,
                                                        pin_memory=True,
                                                        shuffle=True)

        logger.info(f'[Trainer] {len(self.train_dataset)} training data loaded from {train_dataset_path}')

        evaluate_dataset_path: str = os.path.join(dataset_root, 'evaluate')
        self.evaluate_dataset = SynDat(evaluate_dataset_path)
        self.evaluate_loader = torch.utils.data.DataLoader(self.evaluate_dataset,
                                                           num_workers=self.num_workers,
                                                           batch_size=batch_size,
                                                           pin_memory=True)

        logger.info(f'[Trainer] {len(self.evaluate_dataset)} evaluation data loaded from {evaluate_dataset_path}\n')

    # noinspection DuplicatedCode
    def train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            # Train
            self.model.train()
            train_loss_dict: typing.Dict[str, typing.List[torch.Tensor]] = {}

            for batch in self.train_loader:
                x: typing.Optional[torch.Tensor] = None
                bc_value: torch.Tensor = batch['bc_value'].to(self.device)
                bc_mask: torch.Tensor = batch['bc_mask'].to(self.device)
                f: typing.Optional[torch.Tensor] = None

                tup: typing.Tuple[torch.Tensor, int] = self.model(x, bc_value, bc_mask, f)
                y, iterations_used = tup

                residue: torch.Tensor = util.absolute_residue(y, bc_mask, f, reduction='none')

                # abs_residual_norm, rel_residual_norm = util.relative_residue(y, bc_value, bc_mask, f)
                # abs_residual_norm = abs_residual_norm.mean()
                # rel_residual_norm = rel_residual_norm.mean()

                loss_x: torch.Tensor = util.norm(residue).mean()

                iterations_used = torch.tensor([iterations_used], dtype=torch.float32).to(self.device)

                if 'loss_x' in train_loss_dict:
                    train_loss_dict['loss_x'].append(loss_x)
                else:
                    train_loss_dict['loss_x']: typing.List[torch.Tensor] = [loss_x]

                if 'iterations_used' in train_loss_dict:
                    train_loss_dict['iterations_used'].append(iterations_used)
                else:
                    train_loss_dict['iterations_used']: typing.List[torch.Tensor] = [iterations_used]

                loss = self.lambda_1 * loss_x  # + self.lambda_2 * rel_residual_norm

                if 'loss' in train_loss_dict:
                    train_loss_dict['loss'].append(loss)
                else:
                    train_loss_dict['loss']: typing.List[torch.Tensor] = [loss]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for k, v in train_loss_dict.items():
                self.logger.info('[Epoch {}/{}] {} = {}'.format(epoch, self.max_epoch - 1,
                                                                k, torch.mean(torch.tensor(v))))

            # Evaluate
            if 0 < self.evaluate_every and (epoch + 1) % self.evaluate_every == 0:
                with torch.no_grad():
                    self.model.eval()
                    evaluate_loss_dict: typing.Dict[str, typing.List[torch.Tensor]] = {}

                    for batch in self.evaluate_loader:
                        x: typing.Optional[torch.Tensor] = None
                        bc_value: torch.Tensor = batch['bc_value'].to(self.device)
                        bc_mask: torch.Tensor = batch['bc_mask'].to(self.device)
                        f: typing.Optional[torch.Tensor] = None

                        tup: typing.Tuple[torch.Tensor, int] = self.model(x, bc_value, bc_mask, f)
                        y, iterations_used = tup

                        abs_residual_norm, rel_residual_norm = util.relative_residue(y, bc_value, bc_mask, f)
                        abs_residual_norm = abs_residual_norm.mean()
                        rel_residual_norm = rel_residual_norm.mean()

                        if 'abs_residual_norm' in evaluate_loss_dict:
                            evaluate_loss_dict['abs_residual_norm'].append(abs_residual_norm)
                        else:
                            evaluate_loss_dict['abs_residual_norm']: typing.List[torch.Tensor] = [abs_residual_norm]

                        if 'rel_residual_norm' in evaluate_loss_dict:
                            evaluate_loss_dict['rel_residual_norm'].append(rel_residual_norm)
                        else:
                            evaluate_loss_dict['rel_residual_norm']: typing.List[torch.Tensor] = [rel_residual_norm]

                    for k, v in evaluate_loss_dict.items():
                        self.logger.info('[Evaluation] {} = {}'.format(k, torch.mean(torch.tensor(v))))

                    self.model.train()

            # Scheduler step
            self.logger.info('[Epoch {}/{}] Current learning rate = {}'.format(epoch,
                                                                               self.max_epoch - 1,
                                                                               self.optimizer.param_groups[0]['lr']))
            self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0 or epoch == self.max_epoch - 1:
                self.logger.info('[Epoch {}/{}] Model saved.\n'.format(epoch, self.max_epoch - 1))
                self.model.save(self.experienment_checkpoint_path, epoch + 1)
            else:
                self.logger.info('')
