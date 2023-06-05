import numpy as np
from rknn.api import RKNN
import torch
import os

model = "./dtln_p1_rk.pt"
input_size_list = [[1,1,1,513], [1,1,1,128], [1,1,1,128], [1,1,1,128], [1,1,1,128]]

rknn = RKNN(verbose=True)

#Pre-process config
print("--> Config model")
rknn.config[mean_values=[[0], [0], [0], [0], [0]], std_values=[[1], [1], [1], [1], [1]]]
print("--> Config done")

# Load model
print("--> Loading model")
ret = rknn.load_pytorch(model = model, input_size_list = input_size_list)
if ret != 0:
    print("Load model failed!")
    exit(ret)
print("--> Loading done")

#Build model
print("--> Building model")
ret = rknn.build(do_quantization=True)
if ret != 0:
    print("Build model failed!")
    exit(ret)
print("--> Building done")

# Export model
print("--> Export rknn model")
ret = rknn.export_rknn("./dtln_p1_rk.rknn")
if ret != 0:
    print("Export rknn model failed")
    exit(ret)
print("--> Export done")


# Init runtime environment
print("--> Init runtime environment")
ret = rknn.init_runtime()
if ret != 0:
    print("Init runtime environment failed!")
    exit(ret)
print("--> Init done")

# Inference
print("--> Inference")
inputs_mag = np.random.randn(1,1,1,513)
inputs_h1 = np.random.randn(1,1,1,128)
inputs_c1 = np.random.randn(1,1,1,128)
inputs_h2 = np.random.randn(1,1,1,128)
inputs_c2 = np.random.randn(1,1,1,128)

outputs = rknn.inference(inputs = [inputs_mag, inputs_h1, inputs_c1, inputs_h2, inputs_c2])
print("done")
rknn.release()