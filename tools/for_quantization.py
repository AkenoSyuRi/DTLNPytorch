# 此文件为了DTLN模型量化计算输入的均值和方差使用
# 使用模型为./models/DTLN_0518_32k_train_200epochs
# 使用数据集为./train_dataset_sample/noisy, gogogogo!!!
import os
import librosa
import torch
import sys
import numpy as np
import torch.nn as nn
sys.path.append("/home/lizhinan/project/lightse/DTLNPytorch")
class Simple_STFT_Layer(nn.Module):
    def __init__(self, frame_len=1024, frame_hop=256):
        super(Simple_STFT_Layer, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop

    def forward(self, x):
        if len(x.shape) != 2:
            print("x must be in [B, T]")
        y = torch.stft(x, n_fft=self.frame_len, hop_length=self.frame_hop,
                       win_length=self.frame_len, return_complex=True, center=False)
        r = y.real
        i = y.imag
        mag = torch.clamp(r ** 2 + i ** 2, self.eps) ** 0.5
        phase = torch.atan2(i + self.eps, r + self.eps)
        return mag, phase

class SeperationBlock_Stateful(nn.Module):
    def __init__(self, input_size=513, hidden_size=128, dropout=0.25):
        super(SeperationBlock_Stateful, self).__init__()
        self.rnn1 = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.rnn2 = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.drop = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, in_states):
        h1_in, c1_in = in_states[:1, :, :, 0], in_states[:1, :, :, 1]
        h2_in, c2_in = in_states[1:, :, :, 0], in_states[1:, :, :, 1]
        h1_in = h1_in.contiguous()
        c1_in = c1_in.contiguous()
        h2_in = h2_in.contiguous()
        c2_in = c2_in.contiguous()
        x1, (h1, c1) = self.rnn1(x, (h1_in, c1_in))
        x1 = self.drop(x1)
        x2, (h2, c2) = self.rnn2(x1, (h2_in, c2_in))
        x2 = self.drop(x2)
        mask = self.dense(x2)
        mask = self.sigmoid(mask)
        h = torch.cat((h1, h2), dim=0)
        c = torch.cat((c1, c2), dim=0)
        out_states = torch.stack((h, c), dim=-1)
        return mask, out_states

class DTLN_Stateful_RK(nn.Module):
    def __init__(self, frameLength=1024, hopLength=256, hidden_size=128, encoder_size=256):
        super(DTLN_Stateful_RK, self).__init__()
        self.frame_len = frameLength
        self.frame_hop = hopLength
        self.stft = Simple_STFT_Layer(frameLength, hopLength)
        self.sep1 = SeperationBlock_Stateful()
        self.encoder_size = encoder_size
        self.encoder_conv1 = nn.Conv1d(in_channels=frameLength, out_channels=self.encoder_size,
                                       kernel_size=1, stride=1, bias=False)
        self.sep2 = SeperationBlock_Stateful(input_size=self.encoder_size, hidden_size=hidden_size, dropout=0.25)

        ## TODO with causal padding like in keras,when ksize > 1
        self.decoder_conv1 = nn.Conv1d(in_channels=self.encoder_size, out_channels=frameLength,
                                       kernel_size=1, stride=1, bias=False)

    def forward(self, x, in_state1, in_state2): # [N, T], [2,N,128,2], [2,N,128,2] -> [N, T], [2,N,128,2], [2,N,128,2]
        batch, n_frames = x.shape

        mag, phase = self.stft(x)
        mag = mag.permute(0, 2, 1)
        phase = phase.permute(0, 2, 1)

        # N, T, hidden_size
        mask, out_state1 = self.sep1(mag, in_state1)
        estimated_mag = mask * mag

        s1_stft = estimated_mag * torch.exp((1j * phase))
        y1 = torch.fft.irfft2(s1_stft, dim=-1)
        y1 = y1.permute(0, 2, 1)

        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)
        mask_2, out_state2 = self.sep2(encoded_f_norm, in_state2)
        estimated = mask_2 * encoded_f
        estimated = estimated.permute(0, 2, 1)
        decoded_frame = self.decoder_conv1(estimated)
        #print(decoded_frame.shape)
        ## overlap and add
        out = torch.nn.functional.fold(
            decoded_frame,
            (n_frames, 1),
            kernel_size=(self.frame_len, 1),
            padding=(0, 0),
            stride=(self.frame_hop, 1),
        )
        out = out.reshape(batch, -1)
        return out, out_state1, out_state2

    def forward_gety1(self, x, in_state1, in_state2):
        batch, n_frames = x.shape

        mag, phase = self.stft(x)
        mag = mag.permute(0, 2, 1)
        phase = phase.permute(0, 2, 1)

        # N, T, hidden_size
        mask, out_state1 = self.sep1(mag, in_state1)
        estimated_mag = mask * mag

        s1_stft = estimated_mag * torch.exp((1j * phase))
        y1 = torch.fft.irfft2(s1_stft, dim=-1)
        y1 = y1.permute(0, 2, 1)

        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)
        mask_2, out_state2 = self.sep2(encoded_f_norm, in_state2)
        estimated = mask_2 * encoded_f
        estimated = estimated.permute(0, 2, 1)
        decoded_frame = self.decoder_conv1(estimated)
        #print(decoded_frame.shape)
        ## overlap and add
        out = torch.nn.functional.fold(
            decoded_frame,
            (n_frames, 1),
            kernel_size=(self.frame_len, 1),
            padding=(0, 0),
            stride=(self.frame_hop, 1),
        )
        out = out.reshape(batch, -1)
        return y1, out_state1, out_state2


class Quantization_CalMeanStd_Tools():
    def __init__(self, data_dirname, model_path, duration=10, fs=32000, use_gpu=True):
        self.model = DTLN_Stateful_RK()
        self.stft = Simple_STFT_Layer()
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.eval()
        self.data_dirname = data_dirname
        self.data_dirlist = os.listdir(data_dirname)
        self.audioname = []
        for audioname in self.data_dirlist:
            self.audioname.append(os.path.join(self.data_dirname, audioname))
        self.duration = duration
        self.fs = fs
        self.use_gpu = use_gpu
    
    def getMag(self): # get mean and std value of mag in stage1
        audio_array = np.zeros((len(self.audioname), self.duration*self.fs ))
        for i in range(len(self.audioname)):
            audioname = self.audioname[i]
            audiodata, sr = librosa.load(audioname, sr=self.fs)
            audio_array[i, :] = audiodata
            audio_array = torch.FloatTensor(audio_array).cuda()
        self.stftLayer.cuda()
        audio_stft_array, _ = self.stftLayer(audio_array)
        audio_stft_array = audio_stft_array.permute(0,2,1)
        audio_stft_array = audio_stft_array.reshape(-1,513)
        mean_mag = torch.mean(audio_array)
        std_mag = torch.std(audio_array)
        return mean_mag, std_mag
    
    def gety1_and_states(self): # get mean and std value of y1 in stage2, and states in the whole stage
        audio_array = np.zeros((len(self.audioname), self.duration*self.fs ))
        for i in range(len(self.audioname)):
            audioname = self.audioname[i]
            audiodata, sr = librosa.load(audioname, sr=self.fs)
            audio_array[i, :] = audiodata

        audio_array = torch.FloatTensor(audio_array).cuda()
        self.model.cuda()
        block_len = 1024
        block_shift = 256
        hidden_size = 128
        # 两个阶段， 每个阶段四个状态需要计算mean, std
        y1 = np.array([])
        h1_1 = np.array([])
        c1_1 = np.array([])
        h2_1 = np.array([])
        c2_1 = np.array([])
        h1_2 = np.array([])
        c1_2 = np.array([])
        h2_2 = np.array([])
        c2_2 = np.array([])


        for i in range(len(self.audioname)):
            audioData = audio_array[i,:]
            # create buffer
            in_buffer = np.zeros((block_len))
            num_blocks = (audioData.shape[0] - (block_len - block_shift)) // block_shift
            inState1 = torch.zeros(2,1,hidden_size,2).cuda()
            inState2 = torch.zeros(2,1,hidden_size,2).cuda()
            for idx in range(num_blocks):
                # shift values and write to buffer
                in_buffer[:-block_shift] = in_buffer[block_shift:]
                in_buffer[-block_shift:] = audioData[idx * block_shift:(idx * block_shift) + block_shift]
                in_block = np.expand_dims(in_buffer, axis=0).astype(np.float32)
                x = in_buffer
                out_block, inState1, inState2 = self.model.forward_gety1(x, inState1, inState2)
                y1 = np.append(y1, out_block.squeeze(0).detach().numpy())
                h1_1 = np.append(h1_1, inState1[0,0,:,0].detach().numpy())
                c1_1 = np.append(c1_1, inState1[0,0,:,1].detach().numpy())
                h2_1 = np.append(h2_1, inState1[1,0,:,0].detach().numpy())
                c2_1 = np.append(c2_1, inState1[1,0,:,1].detach().numpy())
                h1_2 = np.append(h1_2, inState2[0,0,:,0].detach().numpy())
                c1_2 = np.append(c1_2, inState2[0,0,:,1].detach().numpy())
                h2_2 = np.append(h2_2, inState2[1,0,:,0].detach().numpy())
                c2_2 = np.append(c2_2, inState1[1,0,:,1].detach().numpy())
        return (np.mean(y1), np.std(y1)), (np.mean(h1_1), np.std(h1_1)), (np.mean(c1_1), np.std(c1_1)), (np.mean(h2_1), np.std(h2_1)), (np.mean(c2_1), np.std(c2_1)), \
            (np.mean(h1_2), np.std(h1_2)) , (np.mean(c1_2), np.std(c1_2)), (np.mean(h2_2), np.std(h2_2)), (np.mean(c2_2), np.std(c2_2))


data_dirname = "/home/lizhinan/project/lightse/DTLNPytorch/train_dataset_sample/noisy"
model_path = "/home/lizhinan/project/lightse/DTLNPytorch/models/DTLN_0518_32k_train_200epochs/checkpoints/model_0200.pth"
MeanStdModel = Quantization_CalMeanStd_Tools(data_dirname=data_dirname, model_path=model_path, duration=10, fs=32000)
mag_mean, mag_std = MeanStdModel.getMag()
y1_data, h1_1_data, c1_1_data, h2_1_data, c2_1_data, h1_2_data, c1_2_data, h2_2_data, c2_2_data = MeanStdModel.gety1_and_states()


















