import torch
import torch.nn.functional as F
import os
import numpy as np
import torchaudio

def overlapAdd(tensor, hopLength):
    # tensor.shape : [B, frameLength, numOfFrames]
    batchsize, frameLength, numOfFrames = tensor.shape
    out = F.fold(
            tensor,
            (numOfFrames, 1),
            kernel_size = (frameLength, 1),
            padding = (0, 0),
            stride = (hopLength, 1)
        )
    out = out.reshape(batchsize, -1)
    return out

def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))

def clip_check(y):
    return True if y.any() >= 0.999 else False
### audio process
def norm_amplitude(y):
    scalar = np.max(np.abs(y)) + 1e-7
    return y / scalar, scalar

def to_target_dB(y, targetdB = -25):
    rms = np.sqrt(np.mean(y ** 2))
    scalar = 10 ** (targetdB / 20) / (rms + 1e-7)
    y *= scalar
    return y, rms, scalar

def subsample(data, sub_sample_length, start_position: int = -1, return_start_position=False):
    """
    Randomly select fixed-length data from 

    Args:
        data: **one-dimensional data**
        sub_sample_length: how long
        start_position: If start index smaller than 0, randomly generate one index

    """
    assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
    length = len(data)

    if length > sub_sample_length:
        if start_position < 0:
            start_position = np.random.randint(length - sub_sample_length)
        end = start_position + sub_sample_length
        data = data[start_position:end]
    elif length < sub_sample_length:
        shortage = sub_sample_length - length
        data = np.pad(data, [0, shortage], 'wrap')
    else:
        pass

    assert len(data) == sub_sample_length

    if return_start_position:
        return data, start_position
    else:
        return data


if __name__ == "__main__":
    audio1 = torch.randn(2,20000)
    audio2 = torch.randn(2,20000)