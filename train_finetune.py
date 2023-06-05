import argparse
import os
import random
import sys
import numpy as np
import toml
import torch
from torch.utils.data import DataLoader
from dataloader import Dataset_DNS_finetune
from tools.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train DTLN")
    parser.add_argument("-C", "--train_config", default="configs/train_finetune.toml", help="train configuration")
    parser.add_argument("-R", "--resume", action = "store_true", help="Resume the experiment")
    args = parser.parse_args()
    config = toml.load(args.train_config)

    # set random seed, and visible GPU
    torch.manual_seed(config['meta']['seed'])
    np.random.seed(config['meta']['seed'])
    random.seed(config['meta']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # create Dataset

    train_dataset = Dataset_DNS_finetune(finetune_clean_data_dirname = config['meta']['finetune_clean_data_dirname'],
                                finetune_noise_data_dirname = config['meta']['finetune_noise_data_dirname'],
                                initial_clean_data_dirname = config['meta']['initial_clean_data_dirname'],
                                initial_noisy_data_dirname = config['meta']['initial_noisy_data_dirname'],
                                samplerate = config['audio']['samplerate'],
                                num_workers = config['meta']['num_workers'],
                                duration=config['audio']['duration'],
                                add_silence=False,
                                how_many_finetune_h=config['meta']['how_many_finetune_h'],
                                how_many_initial_h=config['meta']['how_many_initial_h']
                                )
    
    train_dataloader = DataLoader(dataset = train_dataset,
                                batch_size = config['meta']['batch_size'],
                                shuffle = True,
                                num_workers = config['meta']['num_workers'],
                                pin_memory = config['meta']['pin_memory'])

    # create trainer and train network.
    trainer = Trainer(config = config,
                    resume = args.resume,
                    train_dataloader=train_dataloader,
                    eval_dataloader = None,
                    )

    trainer.train()