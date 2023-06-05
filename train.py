import argparse
import os
import random
import sys
import numpy as np
import toml
import torch
from torch.utils.data import DataLoader
from dataloader import DatasetNS, Dataset_DNS
from tools.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train DTLN")
    parser.add_argument("-C", "--train_config", default="configs/train_npu.toml", help="train configuration")
    parser.add_argument("-R", "--resume", action = "store_true", help="Resume the experiment")
    args = parser.parse_args()
    config = toml.load(args.train_config)

    # set random seed, and visible GPU
    torch.manual_seed(config['meta']['seed'])
    np.random.seed(config['meta']['seed'])
    random.seed(config['meta']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # create Dataset
    '''
    train_dataset = DatasetNS(clean_datalist = config['meta']['train_list_path'],
                            noise_datalist = config['meta']['noise_list_path'],
                            rir_datalist = config['meta']['rir_list_path'],
                            snr_range = [3, 10],
                            target_dB = config['audio']['target_dB'],
                            floatRange_dB = config['audio']['floatRange_dB'],
                            reverb_proportion = config['audio']['rir_prob'],
                            samplerate = config['audio']['samplerate'],
                            sub_sample_length = config['audio']['duration'],
                            num_workers = config['meta']['num_workers']
                            )
    '''

    train_dataset = Dataset_DNS(clean_data_dirname = config['meta']['clean_data_dirname'],
                                noisy_data_dirname = config['meta']['noisy_data_dirname'],
                                #noise_data_dirname = config['meta']['noise_data_dirname'],
                                samplerate = config['audio']['samplerate'],
                                num_workers = config['meta']['num_workers'],
                                add_silence=False
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


