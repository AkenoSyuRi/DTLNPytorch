from torch.utils import data
import random
import numpy as np
from scipy import signal
from tools.calculateTools import expand_path, norm_amplitude, to_target_dB, subsample, clip_check
import librosa, soundfile
import toml
import os
import tqdm
import torch
# 模拟房间中的混响效应， 还可以参考https://www.yingsoo.com/news/devops/56757.html


def get_snrlist(snr_range):
    assert len(snr_range) == 2, f"The range of SNR should be [low, high]"
    assert snr_range[0] <= snr_range[-1], f"The low SNR should not larger than high SNR."

    low, high = snr_range
    snr_list = []
    for i in range(low, high + 1, 1):
        snr_list.append(i)

    return snr_list

def getNumber(str_element):
        str_element = str_element.split("fileid_")[-1]
        str_element = str_element.split(".wav")[0]
        return int(str_element)


class Dataset_DNS(data.Dataset):
    def __init__(self,
                clean_data_dirname,
                noisy_data_dirname,
                noise_data_dirname=None,
                samplerate=32000,
                num_workers=8,
                add_silence=False):
        super().__init__()
        self.sr = samplerate

        self.num_workers = num_workers
        self.clean_data_dirname = clean_data_dirname
        self.noisy_data_dirname = noisy_data_dirname
        #self.noise_data_dirname = noise_data_dirname
        #self.noise_data_dirlist = os.listdir(noise_data_dirname)
        #self.noise_data_length = len(self.noise_data_dirlist)
        self.length = len(os.listdir(clean_data_dirname))
        

        self.clean_data_list = os.listdir(clean_data_dirname)
        self.noisy_data_list = os.listdir(noisy_data_dirname)

        self.clean_data_list.sort(key=self.getNumber)
        self.noisy_data_list.sort(key=self.getNumber)
        assert len(self.clean_data_list)==len(self.noisy_data_list), "带噪语音和纯净语音数量对不上！"
        self.add_silence = add_silence

    def getNumber(self, str_element):
        str_element = str_element.split("fileid_")[-1]
        str_element = str_element.split(".wav")[0]
        return int(str_element)

    def __getitem__(self, index):

        clean_data_name = os.path.join(self.clean_data_dirname, self.clean_data_list[index])
        noisy_data_name = os.path.join(self.noisy_data_dirname, self.noisy_data_list[index])
        clean_data, _ = librosa.load(clean_data_name, sr = self.sr)
        noisy_data, _ = librosa.load(noisy_data_name, sr = self.sr)
        prob = np.random.rand()
        if self.add_silence == True:
            prob = np.random.rand()
            if prob <= 0.15:
                random_number_index = np.random.randint(0, self.noise_data_length)
                noise_data_name = self.noise_data_dirlist[random_number_index]
                noise_data_name = os.path.join(self.noise_data_dirname, noise_data_name)
                noise_data, _ = librosa.load(noise_data_name, sr=self.sr)
                if noise_data.shape[0] >= self.sr * 10: noise_data = noise_data[:self.sr * 10]
                else: noise_data = np.pad(noise_data, (0, self.sr*10 - noise_data.shape[0]), mode="wrap")
                clean_data = noise_data * 0.01
                return noise_data, clean_data
        else:
            clean_data = clean_data[:self.sr * 10]
        noisy_data = noisy_data[:self.sr * 10]

        return noisy_data, clean_data

    def __len__(self):
        return self.length
    
    def check(self):
        # 生成50条dataset的音频，到目标语音库中，看看有无出错
        des_clean = "/home/lizhinan/project/lightse/DTLNPytorch/train_dataset_sample/clean"
        des_noisy = "/home/lizhinan/project/lightse/DTLNPytorch/train_dataset_sample/noisy"
        clean_data_list = os.listdir(self.clean_data_dirname)
        noisy_data_list = os.listdir(self.noisy_data_dirname)
        clean_data_list.sort(key=self.getNumber)
        noisy_data_list.sort(key=self.getNumber)

        for i in range(50):
            clean_data_name = os.path.join(self.clean_data_dirname, clean_data_list[i])
            noisy_data_name = os.path.join(self.noisy_data_dirname, noisy_data_list[i])
            clean_data, _ = librosa.load(clean_data_name, sr = self.sr)
            noisy_data, _ = librosa.load(noisy_data_name, sr = self.sr)
            print(clean_data.shape)
            print(clean_data_name)
            soundfile.write(os.path.join(des_clean, clean_data_name.split("/")[-1]), clean_data, samplerate = self.sr)
            soundfile.write(os.path.join(des_noisy, noisy_data_name.split("/")[-1]), noisy_data, samplerate = self.sr)

class Dataset_DNS_finetune(data.Dataset):
    # 不再制定clean-noisy语音对， 这边是给定clean, noise自己生成clean, noisy的语音对进行处理
    def __init__(self,
            finetune_clean_data_dirname,
            finetune_noise_data_dirname,
            initial_clean_data_dirname,
            initial_noisy_data_dirname,
            samplerate=32000,
            num_workers=8,
            duration = 10,
            add_silence=False,
            how_many_finetune_h = 10,
            how_many_initial_h = 20):
        super().__init__()
        self.sr = samplerate
        self.num_workers = num_workers
        self.clean_data_dirname = finetune_clean_data_dirname
        self.noise_data_dirname = finetune_noise_data_dirname
        self.clean_data_list = os.listdir(finetune_clean_data_dirname)
        self.noise_data_list = os.listdir(finetune_noise_data_dirname)
        self.how_many_finetune_h = how_many_finetune_h
        self.how_many_initial_h = how_many_initial_h
        self.duration = 10 # 持续时间长10s

        self.init_dataset = Dataset_DNS(clean_data_dirname=initial_clean_data_dirname, noisy_data_dirname=initial_noisy_data_dirname, samplerate=samplerate,
                                    num_workers=num_workers, add_silence=add_silence)
    def __getitem__(self, index):
        threshold_index = int(self.how_many_finetune_h * 60 * 60 / self.duration)
        if index <= threshold_index:
            clean_data_name = os.path.join(self.clean_data_dirname, self.clean_data_list[index])
            noise_data_name = np.random.choice(self.noise_data_list)
            noise_data_name = os.path.join(self.noise_data_dirname, noise_data_name)
            if index == 0:
                print(noise_data_name)
            clean_data, sr = librosa.load(clean_data_name, sr=self.sr)
            noise_data, sr = librosa.load(noise_data_name, sr=self.sr)
            noise_data = noise_data * 1.2
            clean_data = subsample(data = clean_data, sub_sample_length=self.duration * self.sr)
            noise_data = subsample(data = noise_data, sub_sample_length=self.duration * self.sr)
            noisy_data = clean_data + noise_data
            if clip_check(noisy_data):
                noisy_data, scalar = norm_amplitude(noisy_data)
                clean_data = clean_data / scalar
                return noisy_data, clean_data
            else:
                return noisy_data, clean_data
        else:
            return self.init_dataset[index-threshold_index]

    def __len__(self):
        return int((self.how_many_finetune_h + self.how_many_initial_h) * 60 * 60 / self.duration) # 返回多少个文件对





class DatasetNS(data.Dataset):
    def __init__(self,
                clean_datalist,
                noise_datalist,
                rir_datalist,
                snr_range,
                target_dB,
                floatRange_dB,
                reverb_proportion = 0.2,
                samplerate = 16000,
                sub_sample_length = 3,
                num_workers = 8
                ):
        super().__init__()

        self.sr = samplerate
        self.num_workers = num_workers

        # prepare datalist
        self.clean_dataset_list = [line.rstrip('\n') for line in open(expand_path(clean_datalist), "r")]
        self.noise_dataset_list = [line.rstrip('\n') for line in open(expand_path(noise_datalist), "r")]
        self.rir_dataset_list = [line.rstrip('\n') for line in open(expand_path(rir_datalist), "r")]

        self.noise_dataset_list = self.noise_dataset_list[:15000]

        # set the enhancement of data
        self.target_dB = target_dB
        self.floatRange_dB = floatRange_dB
        self.snr_list = get_snrlist(snr_range)
        self.reverb_proportion = reverb_proportion
        self.sub_sample_length = sub_sample_length

        self.length = len(self.clean_dataset_list)
    
    def __len__(self):
        return self.length

    def _random_select_from(self, datalist):
        return random.choice(datalist)

    def _mixAudios(self, cleanData, noiseData, snr, target_dB, floatRange_dB = 15, rir=None):
        '''
        args:
            cleanData: pure audio
            noiseData: noise audio
            snr: SNR
            rir: room impulse response, None or np.array

        Returns:
            (mixData, cleanData)
        '''
        
        # transform dB to target dB for cleanData.
        cleanData, _ = norm_amplitude(cleanData)
        cleanData, cleanData_RMS, _ = to_target_dB(cleanData, targetdB=target_dB)

        # transform dB to target dB for noiseData.
        noiseData, _ = norm_amplitude(noiseData)
        noiseData, noiseData_RMS, _ = to_target_dB(noiseData, targetdB=target_dB)

        # get SNR and mix Data.
        if rir != False:
            rirPath = random.choice(self.rir_dataset_list)
            rir, sr = librosa.load(rirPath, sr=self.sr)
            
            cleanData_rir = signal.fftconvolve(cleanData, rir)[:len(cleanData)]
            cleanData_rir, _ = norm_amplitude(cleanData_rir)
            cleanData_rir, cleanDatarir_RMS, _ = to_target_dB(cleanData_rir, targetdB=target_dB)

            snr_scalar = 10 ** (snr / 20)
            noiseData = noiseData / (snr_scalar+1e-7)
            mixData = cleanData_rir + noiseData

        else:
            snr_scalar = 10 ** (snr / 20)
            noiseData = noiseData / (snr_scalar+1e-7)
            mixData = cleanData + noiseData

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        mixData_target_dB = np.random.randint(
            target_dB - floatRange_dB,
            target_dB + floatRange_dB
        )
        mixData, _, scalar = to_target_dB(mixData, mixData_target_dB)
        cleanData *= scalar

        # avoid clipping
        if any(np.abs(mixData) > 0.999):
            clip_scalar = np.max(np.abs(mixData)) / (0.99 + 1e-7)
            mixData = mixData / clip_scalar
            cleanData = cleanData / clip_scalar

        return mixData, cleanData

    def __getitem__(self, index):
        clean_file = self.clean_dataset_list[index]
        cleanData, sr = librosa.load(clean_file, sr=self.sr)
        cleanData = subsample(cleanData, sub_sample_length = int(self.sub_sample_length * self.sr))

        noise_file = self._random_select_from(self.noise_dataset_list)
        noiseData, sr = librosa.load(noise_file, sr=self.sr)
        noiseData = subsample(noiseData, sub_sample_length = len(cleanData))

        snr = self._random_select_from(self.snr_list)
        use_rir = bool(np.random.random(1) < self.reverb_proportion)

        mixData, cleanData = self._mixAudios(cleanData = cleanData,
                                            noiseData = noiseData,
                                            snr = snr,
                                            target_dB = self.target_dB,
                                            floatRange_dB = self.floatRange_dB,
                                            rir = use_rir)

        mixData = mixData.astype(np.float32)
        cleanData = cleanData.astype(np.float32)
        
        return mixData, cleanData

class DatasetAEC_pretrain(data.Dataset):
    def __init__(self,
                clean_datalist,
                mic_datalist,
                ref_datalist,
                noise_datalist = None,
                addnoiseprob = 0.2,
                samplerate = 48000,
                sub_sample_length = 10,
                num_workers = 8
                ):
        super().__init__()

        self.sr = samplerate
        self.num_workers = num_workers
        self.sub_sample_length = sub_sample_length

        # prepare datalist
        self.clean_dataset_list = [line.rstrip('\n') for line in open(expand_path(clean_datalist), "r")]
        self.mic_dataset_list = [line.rstrip('\n') for line in open(expand_path(mic_datalist), "r")]
        self.ref_dataset_list = [line.rstrip('\n') for line in open(expand_path(ref_datalist), "r")]
        if noise_datalist != None and addnoiseprob > 0:
            self.noise_dataset_list = [line.rstrip('\n') for line in open(expand_path(noise_datalist), "r")]

        # Test
        self.clean_dataset_list = self.clean_dataset_list[:10000]
        self.mic_dataset_list = self.mic_dataset_list[:10000]
        self.ref_dataset_list = self.ref_dataset_list[:10000]
            

        self.length = len(self.clean_dataset_list)
        self.addnoiseprob = addnoiseprob
    
    def __len__(self):
        return self.length
    
    def _mixAudios(self, micData, noiseData, snr=[6,20]): #add some noise to microphone
        def calRMS(array):
            return np.sqrt(np.mean(array ** 2))
        snr_list = [i for i in range(snr[0],snr[1]+1,1)]
        snr = random.choice(snr_list)
        noiseRMS = calRMS(noiseData)
        micRMS = calRMS(micData)
        noiseRMS_target = micRMS/ (10 ** (snr / 20))
        noiseData_target = noiseData / (noiseRMS / noiseRMS_target)
        micData = micData + noiseData_target

        # avoid clipping
        if any(np.abs(micData) > 0.999):
            clip_scalar = np.max(np.abs(mixData)) / (0.99 + 1e-7)
            mixData = mixData / clip_scalar
        
        return mixData

    def __getitem__(self, index):
        clean_file = self.clean_dataset_list[index]
        cleanData, _ = librosa.load(clean_file, sr = self.sr)
        cleanData = subsample(cleanData, sub_sample_length = int(self.sub_sample_length * self.sr))

        mic_file = self.mic_dataset_list[index]
        micData, _ = librosa.load(mic_file, sr = self.sr)
        micData = subsample(micData, sub_sample_length = int(self.sub_sample_length * self.sr))

        ref_file = self.ref_dataset_list[index]
        refData, _ = librosa.load(ref_file, sr = self.sr)
        refData = subsample(refData, sub_sample_length = int(self.sub_sample_length * self.sr))

        use_add_noise = bool(np.random.random(1) < self.addnoiseprob)
        if use_add_noise:
            noisefile = random.choice(self.noise_dataset_list)
            noiseData, _ = librosa.load(noisefile, sr=self.sr)
            micData = self._mixAudios(micData, noiseData, snr=[6,20])

        cleanData = cleanData.astype(np.float32)
        micData = micData.astype(np.float32)
        refData = refData.astype(np.float32)

        return cleanData, micData, refData


class DatasetAEC_finetune_singletalk(data.Dataset):
    def __init__(self,
                clean_datalist,
                mic_datalist,
                ref_datalist,
                noise_datalist = None,
                addnoiseprob = 0,
                samplerate = 48000,
                sub_sample_length = 10,
                num_workers = 8
                ):
        super().__init__()

        self.sr = samplerate
        self.num_workers = num_workers
        self.sub_sample_length = sub_sample_length

        # prepare datalist
        self.clean_dataset_list = [line.rstrip('\n') for line in open(expand_path(clean_datalist), "r")]
        self.mic_dataset_list = [line.rstrip('\n') for line in open(expand_path(mic_datalist), "r")]
        self.ref_dataset_list = [line.rstrip('\n') for line in open(expand_path(ref_datalist), "r")]
        if noise_datalist != None and addnoiseprob > 0:
            self.noise_dataset_list = [line.rstrip('\n') for line in open(expand_path(noise_datalist), "r")]

        self.clean_dataset_list = self.clean_dataset_list[17500:]
        self.mic_dataset_list = self.mic_dataset_list[17500:]
        self.ref_dataset_list = self.ref_dataset_list[17500:]
            

        self.length = len(self.clean_dataset_list)
        self.addnoiseprob = addnoiseprob
    
    def __len__(self):
        return self.length
    
    def _mixAudios(self, micData, noiseData, snr=[6,20]): #add some noise to microphone
        def calRMS(array):
            return np.sqrt(np.mean(array ** 2))
        snr_list = [i for i in range(snr[0],snr[1]+1,1)]
        snr = random.choice(snr_list)
        noiseRMS = calRMS(noiseData)
        micRMS = calRMS(micData)
        noiseRMS_target = micRMS/ (10 ** (snr / 20))
        noiseData_target = noiseData / (noiseRMS / noiseRMS_target)
        micData = micData + noiseData_target

        # avoid clipping
        if any(np.abs(micData) > 0.999):
            clip_scalar = np.max(np.abs(mixData)) / (0.99 + 1e-7)
            mixData = mixData / clip_scalar
        
        return mixData

    def __getitem__(self, index):
        clean_file = self.clean_dataset_list[index]
        if clean_file != 'None': # double talk
            cleanData, _ = librosa.load(clean_file, sr = self.sr)
            cleanData = subsample(cleanData, sub_sample_length = int(self.sub_sample_length * self.sr))
        
        else:
            # for single talk situation, the target audio shoule be 0.
            cleanData = np.zeros((int(self.sub_sample_length * self.sr))) 

        mic_file = self.mic_dataset_list[index]
        micData, _ = librosa.load(mic_file, sr = self.sr)
        micData = subsample(micData, sub_sample_length = int(self.sub_sample_length * self.sr))

        ref_file = self.ref_dataset_list[index]
        refData, _ = librosa.load(ref_file, sr = self.sr)
        refData = subsample(refData, sub_sample_length = int(self.sub_sample_length * self.sr))

        use_add_noise = bool(np.random.random(1) < self.addnoiseprob)
        if use_add_noise:
            noisefile = random.choice(self.noise_dataset_list)
            noiseData, _ = librosa.load(noisefile, sr=self.sr)
            micData = self._mixAudios(micData, noiseData, snr=[6,20])

        cleanData = cleanData.astype(np.float32)
        micData = micData.astype(np.float32)
        refData = refData.astype(np.float32)

        return cleanData, micData, refData


def checkAECtrainfile(clean_datalist, mic_datalist, ref_datalist):
    clean_dataset_list = [line.rstrip('\n') for line in open(expand_path(clean_datalist), "r")]
    mic_dataset_list = [line.rstrip('\n') for line in open(expand_path(mic_datalist), "r")]
    ref_dataset_list = [line.rstrip('\n') for line in open(expand_path(ref_datalist), "r")]

    i, j, k = len(clean_dataset_list), len(mic_dataset_list), len(ref_dataset_list)
    assert i==j and j==k, "the lengths should be equal!"
    for index in range(i):
        clean_filename = clean_dataset_list[index]
        clean_filename = os.path.basename(clean_filename)
        clean_filename = clean_filename.split('.')[0]
        mic_filename = os.path.basename(mic_dataset_list[i]).split('.')[0]
        ref_filename = os.path.basename(ref_dataset_list[i]).split('.')[0]

        name1 = clean_filename.split('_')[0]
        name2 = mic_filename.split('_')[0]
        name3 = ref_filename.split('_')[0]
        assert name1==name2 and name2 == name3, 'datalist sort wrong!'
    print('check datalist successful!')

if __name__ == "__main__":
    stage = 3
    if stage == 1:
        config = toml.load("./configs/train_dns.toml")
        train_dataset = Dataset_DNS(clean_data_dirname = config['meta']['clean_data_dirname'],
                                    noisy_data_dirname = config['meta']['noisy_data_dirname'],
                                    samplerate = config['audio']['samplerate'],
                                    num_workers = config['meta']['num_workers'])
        train_dataset.check()
        print("check over!")
    if stage == 2:
        config = toml.load("./configs/train_dns.toml")
        train_dataset = Dataset_DNS(clean_data_dirname = config['meta']['clean_data_dirname'],
                                    noisy_data_dirname = config['meta']['noisy_data_dirname'],
                                    samplerate = config['audio']['samplerate'],
                                    num_workers = config['meta']['num_workers'])
        for i in range(len(train_dataset)):
            noisy_data, clean_data = train_dataset[i]
            if noisy_data.shape[0] != (10 * 32000) or  clean_data.shape[0] != (10 * 32000):
                print("error")
                print("i is", i)
                print(noisy_data.shape[0])
                print(clean_data.shape[0])

    if stage == 3:
        config = toml.load("./configs/train_finetune.toml")
        train_dataset = Dataset_DNS_finetune(finetune_clean_data_dirname = config['meta']['finetune_clean_data_dirname'],
                                finetune_noise_data_dirname = config['meta']['finetune_noise_data_dirname'],
                                initial_clean_data_dirname = config['meta']['initial_clean_data_dirname'],
                                initial_noisy_data_dirname = config['meta']['initial_noisy_data_dirname'],
                                samplerate = config['audio']['samplerate'],
                                num_workers = config['meta']['num_workers'],
                                duration=config['audio']['duration'],
                                add_silence=False,
                                how_many_finetune_h=config['meta']['how_many_finetune_h'],
                                how_many_initial_h=config['meta']['how_many_initial_h']
                                )
        des_clean = "/home/lizhinan/project/lightse/DTLNPytorch/train_dataset_sample/finetune_clean"
        des_noisy = "/home/lizhinan/project/lightse/DTLNPytorch/train_dataset_sample/finetune_noisy"
        for i in range(100):
            noisy_data, clean_data = train_dataset[i]
            soundfile.write(os.path.join(des_noisy, f"noise_{i}.wav"), noisy_data, samplerate=32000)
            soundfile.write(os.path.join(des_clean, f"clean_{i}.wav"), clean_data, samplerate=32000)
            



        



        
            









        
        
