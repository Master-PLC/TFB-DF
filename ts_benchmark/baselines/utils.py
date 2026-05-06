# -*- coding: utf-8 -*-

import math
import os
import torch

import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple

from ts_benchmark.baselines.time_series_library.utils.timefeatures import time_features
from ts_benchmark.utils.data_processing import split_time


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: args.lr if epoch < 3 else args.lr * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == "constant":
        lr_adjust = {epoch: args.lr}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class Scheduler:
    def __init__(self, optimizer, args, train_steps, fixed_epoch=None, lradj=None):
        self.optimizer = optimizer
        self.lradj = args.lradj if lradj is None else lradj
        self.num_epochs = args.num_epochs
        self.train_steps = train_steps

        self.step_size = args.step_size
        self.lr_decay = args.lr_decay
        self.min_lr = args.min_lr
        self.mode = args.mode
        self.pct_start = args.pct_start
        self.fixed_epoch = 3 if fixed_epoch is None else fixed_epoch

        if self.lradj is None or self.lradj == 'constant':
            self.scheduler = None

        elif self.lradj == 'reduce':
            _mode = 'min' if self.mode == 0 else 'max'
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=_mode, factor=self.lr_decay, patience=self.step_size, min_lr=self.min_lr)

        elif self.lradj == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.step_size, eta_min=self.min_lr)

        elif self.lradj == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.lr_decay)

        elif self.lradj == 'type1':
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** ((epoch - 1) // 1))

        elif self.lradj == 'type2':
            lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
            lr_lambda = {epoch: lr / args.learning_rate for epoch, lr in lr_adjust.items()}
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda.get(epoch, 1.0))

        elif self.lradj == 'type3':
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 if epoch < self.fixed_epoch else 0.9 ** ((epoch - self.fixed_epoch) // 1))

        elif self.lradj == 'cosine2':
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 + math.cos(epoch / args.num_epochs * math.pi)) / 2)

        elif self.lradj == 'TST':
            max_lr = [g['lr'] for g in optimizer.param_groups]
            if len(max_lr) == 1:
                max_lr = max_lr[0]
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, steps_per_epoch=self.train_steps, epochs=self.num_epochs,
                max_lr=max_lr, pct_start=self.pct_start
            )

        elif self.lradj == 'sigmoid':
            k = 0.5 # logistic growth rate
            s = 10  # decreasing curve smoothing rate
            w = 10
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + np.exp(-k * (epoch - w))) - 1 / (1 + np.exp(-k/s * (epoch - w*s))))

        else:
            raise NotImplementedError

        if self.scheduler is not None:
            self.last_lr = self.scheduler._last_lr[0] if len(self.scheduler._last_lr) == 1 else list(self.scheduler._last_lr)
        else:
            lrs = [g['lr'] for g in optimizer.param_groups]
            self.last_lr = lrs[0] if len(lrs) == 1 else lrs
        print(f'Initial learning rates: {self.last_lr}')

    def get_lr(self):
        return self.last_lr

    def step(self, val_loss=None, epoch=None, verbose=True):
        if self.lradj is None or self.lradj == 'constant':
            return
        elif self.lradj == 'reduce':
            self.scheduler.step(val_loss, epoch)
        elif epoch is not None:
            self.scheduler.step(epoch)
        else:
            self.scheduler.step()
        self.lr_info(verbose=verbose)

    def lr_info(self, verbose=True):
        last_lrs = self.scheduler._last_lr[0] if len(self.scheduler._last_lr) == 1 else list(self.scheduler._last_lr)
        if last_lrs != self.last_lr:
            if verbose:
                print(f'Updating learning rate from {self.last_lr} to {last_lrs}')
            self.last_lr = last_lrs


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        improved = False
        if self.best_score is None:
            self.best_score = score
            improved = True
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            improved = True
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
            self.val_loss_min = val_loss
            self.counter = 0
        return improved


class LocalBufferWriter:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        # 数据结构: { 'scalars': {...}, 'figures': {...} }
        self.data = {
            'scalars': {},
            'figures': {}
        }

    def add_scalar(self, tag, value, step):
        if tag not in self.data['scalars']:
            self.data['scalars'][tag] = {'steps': [], 'values': []}
        
        if torch.is_tensor(value):
            value = value.item()
        
        self.data['scalars'][tag]['steps'].append(step)
        self.data['scalars'][tag]['values'].append(value)

    def add_figure(self, tag, figure, step):
        """
        将 matplotlib figure 转为 RGB numpy 数组缓存
        """
        if tag not in self.data['figures']:
            self.data['figures'][tag] = {'steps': [], 'images': []}

        # === 核心黑魔法：Figure -> Numpy Array ===
        # 1. 强制重绘，确保内容是最新的
        figure.canvas.draw()
        
        # 2. 获取 RGB 数据 (H, W, 3)
        # 注意：frombuffer 拿出来是 flat 的，需要 reshape
        data = np.frombuffer(figure.canvas.tostring_argb(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        data = data.reshape((h, w, 4))
        data = data[:, :, 1:4]
        
        # 3. 存入列表 (为了节省空间，建议在这里 copy 一份，防止引用问题)
        self.data['figures'][tag]['steps'].append(step)
        self.data['figures'][tag]['images'].append(data.copy())

    def close(self):
        os.makedirs(self.log_dir, exist_ok=True)
        file_path = os.path.join(self.log_dir, 'events.pth')
        torch.save(self.data, file_path)


class SlidingWindowDataLoader:
    """
    SlidingWindDataLoader class.

    This class encapsulates a sliding window data loader for generating time series training samples.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        batch_size: int = 1,
        history_length: int = 10,
        prediction_length: int = 2,
        shuffle: bool = True,
    ):
        """
        Initialize SlidingWindDataLoader.

        :param dataset: Pandas DataFrame containing time series data.
        :param batch_size: Batch size.
        :param history_length: The length of historical data.
        :param prediction_length: The length of the predicted data.
        :param shuffle: Whether to shuffle the dataset.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.shuffle = shuffle
        self.current_index = 0

    def __len__(self) -> int:
        """
        Returns the length of the data loader.

        :return: The length of the data loader.
        """
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __iter__(self) -> "SlidingWindowDataLoader":
        """
        Create an iterator and return.

        :return: Data loader iterator.
        """
        if self.shuffle:
            self._shuffle_dataset()
        self.current_index = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate data for the next batch.

        :return: A tuple containing input data and target data.
        """
        if self.current_index >= len(self):
            raise StopIteration

        batch_inputs = []
        batch_targets = []
        for _ in range(self.batch_size):
            window_data = self.dataset.iloc[
                self.current_index : self.current_index
                + self.history_length
                + self.prediction_length,
                :,
            ]
            if len(window_data) < self.history_length + self.prediction_length:
                raise StopIteration  # Stop iteration when the dataset is less than one window size and prediction step size

            inputs = window_data.iloc[: self.history_length].values
            targets = window_data.iloc[
                self.history_length : self.history_length + self.prediction_length
            ].values

            batch_inputs.append(inputs)
            batch_targets.append(targets)
            self.current_index += 1

        # Convert NumPy array to PyTorch tensor
        batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32)
        batch_targets = torch.tensor(batch_targets, dtype=torch.float32)

        return batch_inputs, batch_targets

    def _shuffle_dataset(self):
        """
        Shuffle the dataset.
        """
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)


def train_val_split(train_data, ratio, seq_len):
    if ratio == 1:
        return train_data, None

    elif seq_len is not None:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_time(train_data, border)
        train_data_rest, valid_data = split_time(train_data, border - seq_len)
        return train_data_value, valid_data
    else:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_time(train_data, border)
        return train_data_value, valid_data_rest


def decompose_time(
    time: np.ndarray,
    freq: str,
) -> np.ndarray:
    """
    Split the given array of timestamps into components based on the frequency.

    :param time: Array of timestamps.
    :param freq: The frequency of the time stamp.
    :return: Array of timestamp components.
    """
    df_stamp = pd.DataFrame(pd.to_datetime(time), columns=["date"])
    freq_scores = {
        "m": 0,
        "w": 1,
        "b": 2,
        "d": 2,
        "h": 3,
        "t": 4,
        "s": 5,
    }
    max_score = max(freq_scores.values())
    df_stamp["month"] = df_stamp.date.dt.month
    if freq_scores.get(freq, max_score) >= 1:
        df_stamp["day"] = df_stamp.date.dt.day
    if freq_scores.get(freq, max_score) >= 2:
        df_stamp["weekday"] = df_stamp.date.dt.weekday
    if freq_scores.get(freq, max_score) >= 3:
        df_stamp["hour"] = df_stamp.date.dt.hour
    if freq_scores.get(freq, max_score) >= 4:
        df_stamp["minute"] = df_stamp.date.dt.minute
    if freq_scores.get(freq, max_score) >= 5:
        df_stamp["second"] = df_stamp.date.dt.second
    return df_stamp.drop(["date"], axis=1).values


def get_time_mark(
    time_stamp: np.ndarray,
    timeenc: int,
    freq: str,
) -> np.ndarray:
    """
    Extract temporal features from the time stamp.

    :param time_stamp: The time stamp ndarray.
    :param timeenc: The time encoding type.
    :param freq: The frequency of the time stamp.
    :return: The mark of the time stamp.
    """
    if timeenc == 0:
        origin_size = time_stamp.shape
        data_stamp = decompose_time(time_stamp.flatten(), freq)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    elif timeenc == 1:
        origin_size = time_stamp.shape
        data_stamp = time_features(pd.to_datetime(time_stamp.flatten()), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    else:
        raise ValueError("Unknown time encoding {}".format(timeenc))
    return data_stamp.astype(np.float32)


def forecasting_data_provider(data, config, timeenc, batch_size, shuffle, drop_last):
    dataset = DatasetForTransformer(
        dataset=data,
        history_len=config.seq_len,
        prediction_len=config.pred_len,
        label_len=config.label_len,
        timeenc=timeenc,
        freq=config.freq,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
    )

    return dataset, data_loader


class DatasetForTransformer:
    def __init__(
        self,
        dataset: pd.DataFrame,
        history_len: int = 10,
        prediction_len: int = 2,
        label_len: int = 5,
        timeenc: int = 1,
        freq: str = "h",
    ):
        # init

        self.dataset = dataset
        self.history_length = history_len
        self.prediction_length = prediction_len
        self.label_length = label_len
        self.current_index = 0
        self.timeenc = timeenc
        self.freq = freq
        self.__read_data__()

    def __len__(self) -> int:
        """
        Returns the length of the data loader.

        :return: The length of the data loader.
        """
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __read_data__(self):
        df_stamp = self.dataset.reset_index()
        df_stamp = df_stamp[["date"]].values.transpose(1, 0)
        data_stamp = get_time_mark(df_stamp, self.timeenc, self.freq)[0]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.history_length
        r_begin = s_end - self.label_length
        r_end = s_end + self.prediction_length

        seq_x = self.dataset[s_begin:s_end]
        seq_y = self.dataset[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.tensor(seq_x.values, dtype=torch.float32)
        seq_y = torch.tensor(seq_y.values, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark


class SegLoader(object):
    def __init__(self, data, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.data = data
        self.test_labels = data

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.data[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "val":
            return np.float32(self.data[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.data[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.data[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


def anomaly_detection_data_provider(
    data, batch_size, win_size=100, step=100, mode="train"
):
    dataset = SegLoader(data, win_size, 1, mode)

    shuffle = False
    if mode == "train" or mode == "val":
        shuffle = True

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )
    return data_loader
