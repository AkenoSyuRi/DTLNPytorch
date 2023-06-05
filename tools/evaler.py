import time
from pathlib import Path
from .initial_model import initialize_module
from datetime import datetime
import torch
import torchaudio.functional as F
import toml
import tqdm
import os

def calMSE(input, target):
    batchsize, _, _ = input.shape
    loss = torch.FloatTensor(0.0)
    for i in range(batchsize):
        sub_input = input[i,:,:]
        sub_length = target[i,:,:]
        loss += torch.sum((sub_input - sub_length) ** 2)
    return loss / batchsize

class Evaler():
    def __init__(self, config, resume, eval_dataloader):
        model = initialize_module(config['model']['path'], args = config['model']['args'])
        self.config = config
        self.device = torch.device("cuda")
        self.model = model.to(self.device)

        self.save_dir = Path(config['meta']['save_model_dir']).expanduser().absolute() / config["meta"]['experiment_name']
        self.checkpoints_dir = self.save_dir / 'checkpoints'
        self.logs_dir = self.save_dir / "logs"

        # for realtime inference
        self.frameLength = config['model']['args']['frameLength']
        self.hopLength = config['model']['args']['hopLength']
        self.hidden_size = config['model']['args']['hidden_size']
        self.encoder_size = config['model']['args']['encoder_size']

    def calloss(self, wavTrue, wavPred, method = "SI-SDR", alpha=0.5):
        if method == "SDR":
            wavTrueRMS = torch.mean(wavTrue**2, axis=-1, keepdims=True)
            wavNoiseRMS = torch.mean((wavTrue-wavPred)**2, axis=-1, keepdims=True) + 1e-6
            snr = torch.log10((wavTrueRMS+1e-7)/wavNoiseRMS)
            loss = -10*snr
            loss = torch.mean(loss)
            return loss

        if method == "MSE":
            mseCalulator = torch.nn.MSELoss()
            freqTrue = F.spectrogram(waveform=wavTrue, n_fft=self.frameLength, hop_length=self.hopLength)
            freqPred = F.spectrogram(waveform=wavPred, n_fft=self.frameLength, hop_length=self.hopLength)
            return mseCalulator(freqPred, freqTrue)

        def l2norm(mat, keepdim=True):
            return torch.norm(mat, dim=-1, keepdim=keepdim)
        
        if method == "SI-SDR":
            wavTrue_zm = wavTrue - torch.mean(wavTrue, dim=-1, keepdim=True)
            wavPred_zm = wavPred - torch.mean(wavPred, dim=-1, keepdim=True)
            t = torch.sum(wavPred_zm * wavTrue_zm, dim=-1, keepdim=True) * wavTrue_zm / (l2norm(wavTrue_zm) ** 2 +1e-7)
            return -torch.mean(20 * torch.log10(1e-7 + l2norm(t) / l2norm(wavPred_zm - t)))
        
        if method == "dtln-snr":
            wavTrue_energy = wavTrue ** 2
            wavTrue_energy = torch.mean(wavTrue_energy, dim=-1, keepdim=True)
            noise_energy = (wavTrue - wavPred) ** 2
            noise_energy = torch.mean(noise_energy, dim=-1, keepdims=True)+1e-7
            return torch.mean(-10 * torch.log10(wavTrue_energy/noise_energy))


        if method == "MSE+dtln-snr":
            freqTrue_stft = torch.stft(wavTrue, n_fft=self.frameLength, hop_length=self.hopLength, win_length=self.frameLength, return_complex=True, center=False)
            freqPred_stft = torch.stft(wavPred, n_fft=self.frameLength, hop_length=self.hopLength, win_length=self.frameLength, return_complex=True, center=False)
            freqTrue_stft_mag = torch.clamp(freqTrue_stft.real ** 2 + freqTrue_stft.imag ** 2, torch.finfo(torch.float32).eps) ** 0.5
            freqPred_stft_mag = torch.clamp(freqPred_stft.real ** 2 + freqPred_stft.imag ** 2, torch.finfo(torch.float32).eps) ** 0.5
            loss1 = calMSE(freqPred_stft_mag, freqTrue_stft_mag)

            wavTrue_energy = wavTrue ** 2
            wavTrue_energy = torch.mean(wavTrue_energy, dim=-1, keepdim=True)
            noise_energy = (wavTrue - wavPred) ** 2
            noise_energy = torch.mean(noise_energy, dim=-1, keepdims=True)+1e-7
            loss2 = torch.mean(-10 * torch.log10(wavTrue_energy/noise_energy))

            return loss1 + alpha * loss2
        
        raise("pls set right loss")

    def eval(self):
        for i in os.listdir(self.checkpoints_dir):
            if i.endswith(".pth"):
                self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, i)))
                self.model.eval()
                loss = 0.0
                for batch_id, (mixAudio, cleanAudio) in enumerate(self.eval_dataloader, start=1):
                    instates_hc_1 = torch.zeros(size=(2,1,self.hidden_size,2)).to(self.device)
                    instates_hc_2 = torch.zeros(size=(2,1,self.hidden_size,2)).to(self.device)

                    mixAudio = mixAudio.to(self.device)
                    outAudio = self.model.forward(mixAudio, instates_hc_1, instates_hc_2)
                    cleanAudio = torch.FloatTensor(cleanAudio).to(self.device)

                    self.loss_method = self.config['loss']['method']
                    nloss = self.calloss(wavPred=outAudio, wavTrue=cleanAudio, method = self.loss_method)
                    loss += nloss.item()
                f_eval = open(os.path.join(self.logs_dir, 'eval_log.txt'), 'a+')
                f_eval.write(f'[{datetime.now()}]' + i + "  "+ f'loss:{loss / len(self.eval_dataloader):.5f}' + '\n')
                f_eval.close()
                
            
    

    





        

        

        