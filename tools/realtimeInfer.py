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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C",
                        "--config_path",
                        default='/home/lizhinan/project/lightse/DTLNPytorch/models/DTLN 0411 test_preloaded/2023-04-11 14:48:16.toml',
                        type = str,
                        help = 'config path')
    
    parser.add_argument("-I",
                        "--wav_in",
                        default="/home/lizhinan/project/lightse/DTLNPytorch/Testaudio/ns_in/input.wav",
                        type = str,
                        help = "input path")

    parser.add_argument("-O",
                        "--wav_out",
                        default="/home/lizhinan/project/lightse/DTLNPytorch/Testaudio/ns_out/output.wav",
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

    model = initialize_module(path=config['model']['path'], args=config['model']['args'])
    model.load_state_dict(torch.load("/home/lizhinan/project/lightse/DTLNPytorch/modules/ScriptModel/model.pth", map_location='cpu'))
    model.eval()

    audioData, sr = librosa.load(args.wav_in, sr=16000)
    block_len = config['model']['args']['frameLength']
    block_shift = config['model']['args']['hopLength']
    hidden_size = config['model']['args']['hidden_size']

    outData = np.zeros((len(audioData)))

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

        soundfile.write(args.wav_out, outData, samplerate=16000)
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

        soundfile.write(args.wav_out, outData, samplerate=16000)





