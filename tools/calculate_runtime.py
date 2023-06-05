
from thop import profile
import torch
import torchvision.models as models
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
 
model = Pytorch_DTLN_stateful(1024, 256, hidden_size=128,encoder_size=256)
model.eval()
inState1 = torch.zeros(2,1,128,2)
inState2 = torch.zeros(2,1,128,2)

input = torch.randn((1,32000))
flops, params = profile(model, inputs=(input, inState1, inState2))

print("参数量(M):", params/1e6)
print("GFLOPs:",flops/1e9)

