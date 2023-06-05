import os
import sys
sys.path.append("/home/lizhinan/project/lightse/DTLNPytorch")
import torch
import tqdm
import numpy as np
import argparse
import librosa
import soundfile
import toml
from initial_model import initialize_module
from modules.dtlnModel_ns import Pytorch_DTLN_stateful

def Norm(data):
    if data.any()>0.999:
        return data / np.max(data+0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C",
                        "--config_path",
                        default='/home/lizhinan/project/lightse/DTLNPytorch/models/DTLN_0518_32k_train_200epochs/2023-05-19 02:38:29.toml',
                        type = str,
                        help = 'config path')
    
    parser.add_argument("-I",
                        "--wav_in",
                        default="/home/lizhinan/project/lightse/DTLNPytorch/Testaudio/32k/nn_0603_input.wav",
                        type = str,
                        help = "input path")

    parser.add_argument("-O",
                        "--wav_out",
                        default="/home/lizhinan/project/lightse/DTLNPytorch/Testaudio/32k/nn_0603_output_python.wav",
                        type = str,
                        help = "output path")
    parser.add_argument("-G",
                        "--use_gpu",
                        default=True,
                        type=bool
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parser.parse_args()
    config = toml.load(args.config_path)

    model = initialize_module(config['model']['path'], args = config['model']['args'])
    model.load_state_dict(torch.load("/home/lizhinan/project/lightse/DTLNPytorch/models/DTLN_0602_finetune_lr0.00005_onlynewdata/checkpoints/model_0042.pth", map_location='cpu'))
    model.eval()

    audioData, sr = librosa.load(args.wav_in, sr=32000)
    outData = np.zeros_like(audioData)
    block_len = config['model']['args']['frameLength']
    block_shift = config['model']['args']['hopLength']
    hidden_size = config['model']['args']['hidden_size']
    
    
    # create buffer

    in_buffer = np.zeros((block_len))
    out_buffer = np.zeros((block_len))
    num_blocks = (audioData.shape[0] - (block_len - block_shift)) // block_shift
    if args.use_gpu == False:
        inState1 = torch.zeros(2,1,hidden_size,2)
        inState2 = torch.zeros(2,1,hidden_size,2)
        for idx in tqdm.tqdm(range(num_blocks)):
            # shift values and write to buffer
            in_buffer[:-block_shift] = in_buffer[block_shift:]
            in_buffer[-block_shift:] = audioData[idx * block_shift:(idx * block_shift) + block_shift]
            in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
            x = torch.from_numpy(in_block)
            with torch.no_grad():
                out_block, inState1, inState2 = model.forward_realtime(x, inState1, inState2)
            out_block = out_block.numpy()
            # shift values and write to buffer
            out_buffer[:-block_shift] = out_buffer[block_shift:]
            out_buffer[-block_shift:] = np.zeros((block_shift))
            out_buffer += np.squeeze(out_block)
            # write block to output file
            outData[idx * block_shift:(idx * block_shift) + block_shift] = out_buffer[:block_shift]

        soundfile.write(args.wav_out, outData, samplerate=32000)
    else:
        inState1 = torch.zeros(2,1,hidden_size,2).cuda()
        inState2 = torch.zeros(2,1,hidden_size,2).cuda()
        model = model.cuda()
        for idx in tqdm.tqdm(range(num_blocks)):
            # shift values and write to buffer
            in_buffer[:-block_shift] = in_buffer[block_shift:]
            in_buffer[-block_shift:] = audioData[idx * block_shift:(idx * block_shift) + block_shift]
            in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
            x = torch.from_numpy(in_block).cuda()
            with torch.no_grad():
                out_block, inState1, inState2 = model.forward_realtime(x, inState1, inState2)
            out_block = out_block.detach().cpu().numpy()
            # shift values and write to buffer
            out_buffer[:-block_shift] = out_buffer[block_shift:]
            out_buffer[-block_shift:] = np.zeros((block_shift))
            out_buffer += np.squeeze(out_block)
            # write block to output file
            outData[idx * block_shift:(idx * block_shift) + block_shift] = out_buffer[:block_shift]

        soundfile.write(args.wav_out, outData, samplerate=32000)
