import tensorboardX
import os
from core.config import config


def build_summary_op():
    train_log_path = os.path.join(config.train.output_path, 'train_summary')
    val_log_path = os.path.join(config.train.output_path, 'val_summary')
    if not os.path.isdir(train_log_path):
        os.makedirs(train_log_path)
    if not os.path.isdir(val_log_path):
        os.makedirs(val_log_path)

    train_summary_op = tensorboardX.SummaryWriter(log_dir=train_log_path)
    val_summary_op = tensorboardX.SummaryWriter(log_dir=val_log_path)
    return train_summary_op, val_summary_op


