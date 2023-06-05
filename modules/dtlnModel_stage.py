import torch
import torch.nn as nn
import torch.nn.functional as F

class stftLayer(nn.Module):
    def __init__(self, frameLength, hopLength):
        super(stftLayer, self).__init__()
        self.eps = 1e-7
        self.frameLength = frameLength
        self.hopLength = hopLength
    
    def forward(self, x):
        y = torch.stft(x, n_fft=self.frameLength, hop_length=self.hopLength,
                    win_length=self.frameLength, return_complex=False, center=False)
        real = y[:,:,:,0]
        imag = y[:,:,:,1]
        mag = torch.clamp(real ** 2 + imag ** 2, self.eps) ** 0.5
        phase = torch.atan2(imag + self.eps, real + self.eps)
        return mag, phase


class SeperationKernel(nn.Module):
    def __init__(self, input_size=257, hidden_size = 128):
        super(SeperationKernel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True, dropout=0.0, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True, dropout=0.0, bidirectional=False)
        self.drop = nn.Dropout(0.25)
        self.linear = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, states_hc):
        h1, c1 = states_hc[0:1, :, :], states_hc[1:2, :, :]
        h2, c2 = states_hc[2:3, :, :], states_hc[3:4, :, :]
        _, batchsize, hidden_size = states_hc.shape

        x1, (h1, c1) = self.lstm1(x, (h1, c1))
        x1 = self.drop(x1)
        x2, (h2, c2) = self.lstm2(x1, (h2, c2))
        x2 = self.drop(x2)

        mask = self.linear(x2)
        mask = self.sigmoid(mask)

        states_hc_out = torch.zeros(4, batchsize, hidden_size)
        states_hc_out[0:1, :, :], states_hc_out[1:2, :, :] = h1, c1
        states_hc_out[2:3, :, :], states_hc_out[3:4, :, :] = h2, c2
        
        return mask, states_hc_out


        


class DTLN_NS(nn.Module):
    def __init__(self, frameLength=512, hopLength=128, hidden_size=512):
        super(DTLN_NS, self).__init__()
        self.frameLength = frameLength
        self.hopLength = hopLength

        self.stftLayer = stftLayer(frameLength, hopLength)
        self.kernel1 = SeperationKernel(input_size=(frameLength//2 + 1), hidden_size=hidden_size)
        self.encode_size = 256
        self.encoder = nn.Conv1d(in_channels=frameLength, out_channels=self.encode_size, kernel_size=1, stride=1, bias=False)
        self.encode_Layernorm = nn.LayerNorm(self.encode_size)

        self.kernel2 = SeperationKernel(input_size=self.encode_size, hidden_size=hidden_size)
        self.decoder = nn.Conv1d(in_channels=self.encode_size, out_channels=frameLength, kernel_size=1, stride=1, bias=False)

    
    def forward(self, x, states_hc):
        states_hc_kernel1 = states_hc[0:4,:,:]
        states_hc_kernel2 = states_hc[4:,:,:]
        mag, phase = self.stftLayer(x)
        mag = mag.permute(0, 2, 1)
        phase = phase.permute(0, 2, 1)

        mask1, states_hc_kernel1_out = self.kernel1(mag, states_hc_kernel1)
        estimated_mag = mask1 * mag

        estimated_stft = estimated_mag * torch.exp((1j * phase))
        y1 = torch.fft.irfft(estimated_stft, dim=-1)
        y1 = y1.permute(0, 2, 1)

        encoded = self.encoder(y1)
        encoded = encoded.permute(0, 2, 1)
        encoded_norm = self.encode_Layernorm(encoded)

        mask2, states_hc_kernel2_out = self.kernel2(encoded_norm, states_hc_kernel2)
        estimated_mag2 = mask2 * encoded
        estimated_mag2 = estimated_mag2.permute(0, 2, 1)
        decode_frame = self.decoder(estimated_mag2)

        states_hc_out = torch.zeros_like(states_hc)
        states_hc_out[0:4, :, :] = states_hc_kernel1_out
        states_hc_out[4:, :, :] = states_hc_kernel2_out

        return decode_frame, states_hc_out

    def forward_train(self, x, states_hc):
        decode_frames, _ = self.forward(x, states_hc)
        batchsize, n_samples = x.shape
        out = F.fold(
            decode_frames,
            (n_samples, 1),
            kernel_size=(self.frameLength, 1),
            padding=(0,0),
            stride=(self.hopLength, 1)
        )
        out = out.reshape(batchsize, -1)
        return out

class DTLN_NS_P1(nn.Module):
    def __init__(self, frameLength=512, hopLength=128, hidden_size=512):
        super(DTLN_NS_P1, self).__init__()
        self.frameLength = frameLength
        self.hopLength = hopLength
        self.kernel1 = SeperationKernel(input_size=frameLength//2 + 1, hidden_size=hidden_size)
    
    def forward(self, mag, states_hc1):
        mask, states_hc1 = self.kernel1(mag, states_hc1)
        estimated_mag = mask * mag
        return estimated_mag, states_hc1

class DTLN_NS_P2(nn.Module):
    def __init__(self, frameLength=512, hopLength=128, hidden_size=512):
        super(DTLN_NS_P2, self).__init__()
        self.frameLength = frameLength
        self.encoder_size = 256
        self.encoder = nn.Conv1d(in_channels=frameLength, out_channels=self.encoder_size, kernel_size=1, stride=1, bias=False)

        self.encode_Layernorm = nn.LayerNorm(self.encoder_size)
        self.kernel2 = SeperationKernel(input_size=self.encoder_size, hidden_size=hidden_size)
        self.decoder = nn.Conv1d(in_channels=self.encoder_size, out_channels=frameLength, kernel_size=1, stride=1, bias=False)

    def forward(self, x, states_hc2):
        encoded = self.encoder(x)
        encoded = encoded.permute(0, 2, 1)
        encoded_norm = self.encode_Layernorm(encoded)

        mask2, states_hc2 = self.kernel2(encoded_norm, states_hc2)
        estimated_mag2 = mask2 * encoded
        estimated_mag2 = estimated_mag2.permute(0, 2, 1)
        decoded_frame = self.decoder(estimated_mag2)

        return decoded_frame, states_hc2


if __name__ == "__main__":
    model = DTLN_NS()
    input = torch.randn(2,20000)
    states_hc = torch.randn(8,2,512)
    output, _ = model(input, states_hc)
    print(output.shape)
