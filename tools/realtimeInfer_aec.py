import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import tqdm
import numpy as np
import argparse
import librosa
import soundfile
import toml
from initial_model import initialize_module
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C",
                        "--config_path",
                        default='/home/lizhinan/project/lightse/DTLNPytorch/models/DTLN-aec TrainTest 0324_stateful/2023-03-25 10:08:51.toml',
                        type = str,
                        help = 'config path')
    
    parser.add_argument("-I",
                        "--wav_in",
                        default="/home/lizhinan/project/lightse/DTLNPytorch/Testaudio/aec_in_doubletalk",
                        type = str,
                        help = "input dir path")

    parser.add_argument("-O",
                        "--wav_out",
                        default="/home/lizhinan/project/lightse/DTLNPytorch/Testaudio/aec_out_doubletalk",
                        type = str,
                        help = "output dir path")
    args = parser.parse_args()
    config = toml.load(args.config_path)

    model = initialize_module(path=config['model']['path'], args=config['model']['args'])
    model.load_state_dict(torch.load("/home/lizhinan/project/lightse/DTLNPytorch/models/DTLN-aec Trainfinetune_singletalk 0327_stateful/checkpoints/model_0037.pth", map_location='cpu'))
    model.eval()

    ref_wavpath = glob.glob(os.path.join(args.wav_in, "*lpb.wav"))[0]
    mic_wavpath = glob.glob(os.path.join(args.wav_in, "*mic.wav"))[0]

    samplerate = config['audio']['samplerate']
    refData, sr = librosa.load(ref_wavpath, sr=samplerate)
    micData, sr = librosa.load(mic_wavpath, sr=samplerate)
    #refData = refData[:sr*10]
    #micData = micData[:sr*10]

    block_len = config['model']['args']['frameLength']
    block_shift = config['model']['args']['hopLength']
    hiddensize_RNN = config['model']['args']['hiddenSizeOfRNN']

    outData = np.zeros((len(micData)))

    # create buffer
    in_buffer_ref = np.zeros((block_len))
    in_buffer_mic = np.zeros((block_len))
    out_buffer = np.zeros((block_len))
    num_blocks = (micData.shape[0] - (block_len - block_shift)) // block_shift

    inState1 = torch.zeros(2,1,hiddensize_RNN,2)
    inState2 = torch.zeros(2,1,hiddensize_RNN,2)

    for idx in tqdm.tqdm(range(num_blocks)):
        # shift values and write to buffer
        in_buffer_ref[:-block_shift:] = in_buffer_ref[block_shift:]
        in_buffer_ref[-block_shift:] = refData[idx * block_shift:(idx * block_shift) + block_shift]
        in_buffer_mic[:-block_shift] = in_buffer_mic[block_shift:]
        in_buffer_mic[-block_shift:] = micData[idx * block_shift:(idx * block_shift) + block_shift]

        in_block_ref = np.expand_dims(in_buffer_ref, axis=0).astype('float32')
        in_block_mic = np.expand_dims(in_buffer_mic, axis=0).astype('float32')

        in_block_ref_tensor = torch.from_numpy(in_block_ref)
        in_block_mic_tensor = torch.from_numpy(in_block_mic)


        with torch.no_grad():
            out_block, inState1, inState2 = model.forward(in_block_mic_tensor, in_block_ref_tensor, inState1, inState2)
        out_block = out_block.detach().numpy()

        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        # write block to output file
        outData[idx * block_shift: (idx * block_shift) + block_shift] = out_buffer[:block_shift]

    outDatapath = os.path.join(args.wav_out,'out.wav')
    soundfile.write(outDatapath, outData, samplerate=samplerate)











