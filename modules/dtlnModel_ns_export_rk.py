import torch
import torch.nn as nn
import numpy as np
import os


class Pytorch_InstantLayerNormalization_NCNN_Compat(nn.Module):
    def __init__(self, channels):
        """
            Constructor
        """
        super(Pytorch_InstantLayerNormalization_NCNN_Compat, self).__init__()
        self.epsilon = 1e-7
        self.gamma = nn.Parameter(torch.ones(1, 1, channels), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, 1, channels), requires_grad=True)
        self.register_parameter("gamma", self.gamma)
        self.register_parameter("beta", self.beta)

    def forward(self, inputs):
        # calculate mean of each frame
        mean = torch.mean(inputs)
        sub = inputs - mean
        # calculate variance of each frame
        variance = torch.mean(torch.square(sub))
        # calculate standard deviation
        std = torch.sqrt(variance + self.epsilon)
        # normalize each frame independently
        outputs = sub / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs

class SeperationBlock_Stateful(nn.Module):
    def __init__(self, input_size=257, hidden_size=128, dropout=0.25):
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

    def forward(self, x, h1_in, c1_in, h2_in, c2_in):
        x1, (h1, c1) = self.rnn1(x, (h1_in, c1_in))
        x1 = self.drop(x1)
        x2, (h2, c2) = self.rnn2(x1, (h2_in, c2_in))
        x2 = self.drop(x2)

        mask = self.dense(x2)
        mask = self.sigmoid(mask)

        return mask, h1, c1, h2, c2

class DTLN_RK_P1(nn.Module):
    def __init__(self, frame_len=1024, frame_hop=256):
        super(DTLN_RK_P1, self).__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.sep1 = SeperationBlock_Stateful(input_size=513, hidden_size=128, dropout=0.25)
    def forward(self, mag, h1_in, c1_in, h2_in, c2_in):
        '''
        args: mag : [1,1,1,513]
            h_in, c_in : [1,1,128]
        '''
        mag_reshape = mag.reshape(1,1,513)
        h1_in_reshape = h1_in.reshape(1,1,128)
        c1_in_reshape = c1_in.reshape(1,1,128)
        h2_in_reshape = h2_in.reshape(1,1,128)
        c2_in_reshape = c2_in.reshape(1,1,128)
        mask, h1, c1, h2, c2 = self.sep1(mag_reshape, h1_in_reshape, c1_in_reshape, h2_in_reshape, c2_in_reshape)
        estimated_mag = mask * mag_reshape
        return estimated_mag, h1, c1, h2, c2

class DTLN_RK_P2(nn.Module):
    def __init__(self, frame_len=1024):
        super(DTLN_RK_P2, self).__init__()
        self.frame_len = frame_len
        self.encoder_size = 256
        self.encoder_conv1 = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, bias=False)

        self.encoder_norm1 = Pytorch_InstantLayerNormalization_NCNN_Compat(channels=256)
        self.sep2 = SeperationBlock_Stateful(input_size=256, hidden_size=128, dropout=0.25)
        self.decoder_conv1 = nn.Conv1d(in_channels=self.encoder_size, out_channels=frame_len, kernel_size=1, stride=1, bias=False)

    def forward(self, y1, h1_in, c1_in, h2_in, c2_in):
        y1_reshape = y1.reshape(1,1024,1)
        h1_in_reshape = h1_in.reshape(1,1,128)
        c1_in_reshape = c1_in.reshape(1,1,128)
        h2_in_reshape = h2_in.reshape(1,1,128)
        c2_in_reshape = c2_in.reshape(1,1,128)

        encoded_f = self.encoder_conv1(y1_reshape)
        encoded_f = encoded_f.permute(0,2,1)
        encoded_f_norm = self.encoder_norm1(encoded_f)

        mask_2, h1, c1, h2, c2 = self.sep2(encoded_f_norm, h1_in_reshape, c1_in_reshape, h2_in_reshape, c2_in_reshape)
        estimated = mask_2 * encoded_f
        estimated = estimated.permute(0,2,1)
        decoded_frame = self.decoder_conv1(estimated)
        return decoded_frame, h1, c1, h2, c2

if __name__ == "__main__":


    stage = 3 # 1导出p1， 2导出p2, 3导出torchScript精度测试

    if stage == 1:
        model = DTLN_RK_P1()
        model.load_state_dict(torch.load("/home/lizhinan/project/lightse/DTLNPytorch/models/DTLN_0518_32k_train_200epochs/checkpoints/model_0200.pth"), strict=False)
        input = torch.randn(1,1,1,513)
        h1 = torch.randn(1,1,1,128)
        c1 = torch.randn(1,1,1,128)
        h2 = torch.randn(1,1,1,128)
        c2 = torch.randn(1,1,1,128)
        model.eval()
        output, _, _ ,_ ,_ = model(input, h1, c1, h2, c2)
        print(output.shape)

        jit_model = torch.jit.trace(model, example_inputs=[input, h1, c1, h2, c2])
        jit_model.save("./dtln_p1_rk.pt")
    if stage == 2:
        model = DTLN_RK_P2()
        model.load_state_dict(torch.load("/home/lizhinan/project/lightse/DTLNPytorch/models/DTLN_0518_32k_train_200epochs/checkpoints/model_0200.pth"), strict=False)
        input = torch.randn(1,1,1024,1)
        h1 = torch.randn(1,1,1,128)
        c1 = torch.randn(1,1,1,128)
        h2 = torch.randn(1,1,1,128)
        c2 = torch.randn(1,1,1,128)
        model.eval()
        output, _, _ ,_ ,_ = model(input, h1, c1, h2, c2)
        print(output.shape)
        jit_model = torch.jit.trace(model, example_inputs=[input, h1, c1, h2, c2])
        jit_model.save("./dtln_p2_rk.pt")

    if stage == 3:
        input_mag = torch.randn(1,1,1,513)
        h1 = torch.randn(1,1,1,128)
        c1 = torch.randn(1,1,1,128)
        h2 = torch.randn(1,1,1,128)
        c2 = torch.randn(1,1,1,128)
        input_y1 = torch.randn(1,1,1024,1)

        model1_jit = torch.jit.load("./dtln_p1_rk.pt")
        model2_jit = torch.jit.load("./dtln_p2_rk.pt")
        model1 = DTLN_RK_P1()
        model1.load_state_dict(torch.load("/home/lizhinan/project/lightse/DTLNPytorch/models/DTLN_0518_32k_train_200epochs/checkpoints/model_0200.pth"), strict=False)
        model2 = DTLN_RK_P2()
        model2.load_state_dict(torch.load("/home/lizhinan/project/lightse/DTLNPytorch/models/DTLN_0518_32k_train_200epochs/checkpoints/model_0200.pth"), strict=False)
        model1_jit.eval()
        model1.eval()
        model2_jit.eval()
        model2.eval()

        output1_jit, h1_jit, c1_jit, h2_jit, c2_jit = model1_jit(input_mag, h1, c1, h2, c2)
        output1, h1_init, c1_init, h2_init, c2_init = model1(input_mag, h1, c1, h2, c2)
        print("P1 精度测试：")
        print("output1:\n", output1==output1_jit )
        print("h1:\n", h1_init==h1_jit)
        print("c1:\n", c1_init==c1_jit)
        print("h2:\n", h2_init==h2_jit)
        print("c2:\n", c2_init==c2_jit)

        output2_jit, h1_jit, c1_jit, h2_jit, c2_jit = model2_jit(input_y1, h1, c1, h2, c2)
        output2, h1_init, c1_init, h2_init, c2_init = model2(input_y1, h1, c1, h2, c2)

        print("P2 精度测试：")
        print("output2:\n", output2_jit==output2)
        print("h1:\n", h1_init==h1_jit)
        print("c1:\n", c1_init==c1_jit)
        print("h2:\n", h2_init==h2_jit)
        print("c2:\n", c2_init==c2_jit)

