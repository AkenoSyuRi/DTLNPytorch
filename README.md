# DTLNPytorch
Hello! 这是一个有关用于语音降噪模型DTLN的Pytorch版仓库，其中包含了预训练模型，较完善的训练代码，支持断点训练，加载预训练模型训练，修改优化器，模型配置等功能。

## Requirements  
Ubuntu 18.04  
Python 3.8  
环境依赖见./requirements.txt
```python
conda create -n se python==3.8
pip install -r ./requirements.txt
```

## Prepare Data
这部分参见./dataloader.py中我为dataset写的几个继承于torch.utils.data.Dataset的类，建议实际训练的时候使用Dataset_DNS, 或者 Dataset_DNS_finetune(如果你需要微调模型的话)两个类。前者使用DNS-Challenge提供的数据格式， 后者支持自己加入数据混合DNS的数据一起训练。  


对于16k的模型训练，建议使用[DNS-Challenge2020](https://github.com/microsoft/DNS-Challenge/tree/interspeech2020/master)。需要git lfs技术进行大文件存储，git clone的时候注意一下分支，选择interspeech2020/master分支。参照官方教程将数据配置好。


对于32k的模型训练，建议使用[DNS-Challenge主分支](https://github.com/microsoft/DNS-Challenge)。使用其中的[download-dns-challenge-4.sh](https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-4.sh)脚本进行下载，其中包含全频段的数据。下载后同样按照官方提供的.cfg配置文件以及.py脚本将数据处理成clean, noise, noisy三个文件夹。  


处理好的数据放置在dataset文件夹中，文件目录结构如下:
- dataset
    - clean
        - clean_fileid_0.wav
        - clean_fileid_1.wav
        - ...
    - noise
        - ...
    - noisy
        - **0.wav
        - **1.wav
        - ...  

其中clean语音和noisy语音一一对应，根据.wav前的那个数字。 noisy文件就是我们送入网络的输入，clean文件就是我们希望网络处理noisy后的目标文件。将clean和noisy的目录填入config/**.toml下的配置文件即可。

## Train Model
使用的是./train.py文件，整个训练代码架构较多参考了[FullSubnet-Plus](https://github.com/RookieJunChen/FullSubNet-plus)训练过程的代码实现。在./train_dns.toml配置好训练参数，执行以下:
```python
python train.py -C ./configs/train_dns.toml
```
或者
```python
nohup python train.py -C ./configs/train_dns.toml &
```

之后会在model下开创一个目录，目录名字即是你配置文件中的**save_model_dir+experiment_name**字段。会在该目录下生成**checkpoints**和**logs**两个子目录，分别存下训练过程中的模型以及loss。  


下面介绍下训练中断后如何续点训练。保证训练配置文件与原训练配置文件一致，执行以下：
```python
python train.py -C ./configs/train_dns.toml -R
```

## Eval Model
我在tools.trainer下留了个eval的接口:  
```python
def eval(self, model , eval_datalist):
    pass
```
供后来者需要在训练过程中评估模型指标使用。但笔者训练所使用的优化器并不是torch.optim.lr_scheduler.ReduceLROnPlateau, 因此自己并未完善。个人比较喜欢在一次完整的训练过程结束后评估模型指标，eval部分可以参考./eval.py及./evaler.py，这部分还未整理，并不是一个完全版。

个人训练32k的模型，帧长/移选用的是1024/256，在loss选取跟原作一样是SI-SDR的情况下，训练集参考[日志](https://github.com/Plutoisme/DTLNPytorch/blob/main/model/DTLN_0531_si-snr_lr%3D0.002/logs/train_log.txt)是在损失在-18+，个人实测测试集损失与训练集差距一般在1以内。

## Realtime Inference
实时推理需要了解Stateful RNN类模型的概念， 在训练过程中使用LSTM是能够自动将隐藏状态进行传递。但是在实际情况下我们接受到的信息是以帧为单位的， 需要人为将状态进行传递。根据DTLN作者原述**如果不考虑状态传递实际性能会下降**，个人理解也是如此，因为帧与帧之间推理的关系是独立的了。 具体实现单音频.wav文件推理可见./tools/realtimeInfer_32k.py, 实现一组音频.wav文件推理可见./tools/realtimeInfer_32k_dir.py。

在./tools/realtimeInfer_32k.py下配置好所选用模型的配置文件，模型.pth文件，输入的.wav文件及输出的.wav文件即可实现推理:
```python
cd ./tools
python realtimeInfer_32k.py
```


## Demonstrate

## Deploy on Arm32 cpu

## Finetune




















## Citing
借鉴了[DTLN的Tensorflow->Pytorch转换及NCNN部署](https://github.com/lhwcv/DTLN_pytorch)，实现模型的NCNN部署，所使用的模型结构为了能和其代码对接，使用的是其模型结构(我自己写的转了移动端推理不对，后面直接用这位大佬的了)。
另外，参考DTLN原作论文:
```python
@inproceedings{Westhausen2020,
  author={Nils L. Westhausen and Bernd T. Meyer},
  title={{Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={2477--2481},
  doi={10.21437/Interspeech.2020-2631},
  url={http://dx.doi.org/10.21437/Interspeech.2020-2631}
}
```