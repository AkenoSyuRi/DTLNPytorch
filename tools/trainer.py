import time
from pathlib import Path
from .initial_model import initialize_module
from datetime import datetime
import torch
import torchaudio.functional as F
import toml
import tqdm
import os
from torch.cuda.amp import GradScaler, autocast

def prepare_empty_dir(dirs, resume=False):
    """
    if resume the experiment, assert the dirs exist. If not the resume experiment, set up new dirs.
    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists(), "In resume mode, you must be have an old experiment dir."
        else:
            dir_path.mkdir(parents=True, exist_ok=True)

class Trainer():
    def __init__(self, config, resume,  train_dataloader, eval_dataloader):
        model = initialize_module(config['model']['path'], args = config['model']['args'])
        self.config = config
        # set model, loss, optimizer, scheduler
        self.device = torch.device("cuda")
        self.train_dataloader = train_dataloader
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                        lr = config['optimizer']['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.97)
        self.scaler = GradScaler()

        self.train_config = config['trainer']['args']
        self.epochs = config['meta']['num_epochs']

        self.start_epoch = 1
        self.save_dir = Path(config['meta']['save_model_dir']).expanduser().absolute() / config["meta"]['experiment_name']
        self.checkpoints_dir = self.save_dir / 'checkpoints'
        self.logs_dir = self.save_dir / "logs"

        # clip and AMP
        self.clipset = config['trainer']['args']['clip_grad_norm_ornot']
        self.clip_norm_value = config['trainer']['args']['clip_grad_norm_value']

        # for realtime inference
        self.frameLength = config['model']['args']['frameLength']
        self.hopLength = config['model']['args']['hopLength']
        self.hidden_size = config['model']['args']['hidden_size']
        if resume:
            self._resume_checkpoint()

        if config['meta']['preload_model_path']:
            self._preload_model(config['meta']['preload_model_path'])
        
        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume = resume)
        with open((self.save_dir / f"{time.strftime('%Y-%m-%d %H:%M:%S')}.toml").as_posix(), "w") as handle:
            toml.dump(config, handle)
        self._print_networks([self.model])

    def _preload_model(self, model_path):
        """
        Preload model parameters (in "*.tar" format) at the start of experiment.

        Args:
            model_path (Path): The file path of the *.tar file
        """
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        print(f"Model preloaded successfully from {model_path}.")

    def _resume_checkpoint(self):
        """
        Resume the experiment from the latest checkpoint.
        """
        latest_model_path = self.checkpoints_dir.expanduser().absolute() / "latest_model.tar"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can not load latest checkpoint."
        checkpoint = torch.load(latest_model_path.as_posix(), map_location="cpu")

        self.start_epoch = checkpoint["epoch"] + 1
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.model.load_state_dict(checkpoint["model"])
        self.model.cuda()
        print(f"Model checkpoint loaded. Training will begin at {self.start_epoch} epoch.")

    def _save_checkpoint(self, epoch):
        """
        Save checkpoint to "<save_dir>/<config name>/checkpoints" directory, which consists of:
            - epoch
            - optimizer parameters
            - model parameters
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")
        state_dict = {
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        state_dict["model"] = self.model.state_dict()
        torch.save(state_dict["model"], (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth").as_posix())
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())  # Latest

    @staticmethod
    def _print_networks(models: list):
        print(f"This project contains {len(models)} models, the number of the parameters is: ")
        params_of_all_networks = 0
        for idx, model in enumerate(models, start=1):
            params_of_network = 0
            for param in model.parameters():
                params_of_network += param.numel()
            print(f"\tNetwork {idx}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network
        print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")

    def calloss(self, wavTrue, wavPred, method = "dtln-snr", alpha=15):
        if method == "MSE":
            mseCalulator = torch.nn.MSELoss()
            freqTrue_stft = torch.stft(wavTrue, n_fft=self.frameLength, hop_length=self.hopLength, win_length=self.frameLength, return_complex=True, center=False)
            freqPred_stft = torch.stft(wavPred, n_fft=self.frameLength, hop_length=self.hopLength, win_length=self.frameLength, return_complex=True, center=False)
            freqTrue_stft_mag = torch.clamp(freqTrue_stft.real ** 2 + freqTrue_stft.imag ** 2, torch.finfo(torch.float32).eps) ** 0.5
            freqPred_stft_mag = torch.clamp(freqPred_stft.real ** 2 + freqPred_stft.imag ** 2, torch.finfo(torch.float32).eps) ** 0.5
            loss1 = mseCalulator(freqPred_stft_mag, freqTrue_stft_mag)
            return loss1
        
        if method == "dtln-snr":
            wavTrue_energy = wavTrue ** 2
            wavTrue_energy = torch.mean(wavTrue_energy, dim=-1, keepdim=True)
            noise_energy = (wavTrue - wavPred) ** 2
            noise_energy = torch.mean(noise_energy, dim=-1, keepdims=True)+1e-7
            return -10*torch.mean(torch.log10(wavTrue_energy/noise_energy))


        if method == "MSE+dtln-snr":
            mseCalulator = torch.nn.MSELoss()
            freqTrue_stft = torch.stft(wavTrue, n_fft=self.frameLength, hop_length=self.hopLength, win_length=self.frameLength, return_complex=True, center=False)
            freqPred_stft = torch.stft(wavPred, n_fft=self.frameLength, hop_length=self.hopLength, win_length=self.frameLength, return_complex=True, center=False)
            freqTrue_stft_mag = torch.clamp(freqTrue_stft.real ** 2 + freqTrue_stft.imag ** 2, torch.finfo(torch.float32).eps) ** 0.5
            freqPred_stft_mag = torch.clamp(freqPred_stft.real ** 2 + freqPred_stft.imag ** 2, torch.finfo(torch.float32).eps) ** 0.5
            loss1 = mseCalulator(freqPred_stft_mag, freqTrue_stft_mag)
            #print("loss1 is", loss1)

            wavTrue_energy = wavTrue ** 2
            wavTrue_energy = torch.mean(wavTrue_energy, dim=-1, keepdim=True)
            noise_energy = (wavTrue - wavPred) ** 2
            noise_energy = torch.mean(noise_energy, dim=-1, keepdims=True)+1e-7
            loss2 = torch.mean(-10 * torch.log10(wavTrue_energy/noise_energy))
            #print("loss2 is", loss2)

            return 15 * loss1 + loss2
        
        raise("pls set right loss")
            
                
    def eval(self, model , eval_datalist):
        pass

    def _train_epoch_ns(self, epoch, num_epochs):
        loss = 0.0
        progress_bar = tqdm.tqdm(total=len(self.train_dataloader), desc='Training')
        for batch_id, (mixAudio, cleanAudio) in enumerate(self.train_dataloader, start=1):
            #self.scheduler.step(epoch-1)
            self.optimizer.zero_grad()
            batchsize = mixAudio.shape[0]
            instates_hc_1 = torch.zeros(size=(2, batchsize, self.hidden_size, 2)).to(self.device)
            instates_hc_2 = torch.zeros(size=(2, batchsize, self.hidden_size, 2)).to(self.device)

            mixAudio = mixAudio.to(self.device)
            outAudio = self.model.forward(mixAudio, instates_hc_1, instates_hc_2)
            cleanAudio = torch.FloatTensor(cleanAudio).to(self.device)

            # calculate loss and update parameters
            self.loss_method = self.config['loss']['method']
            nloss = self.calloss(wavPred=outAudio, wavTrue=cleanAudio, method = self.loss_method)
            nloss.backward()
            if self.clipset:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm_value, norm_type=2)
            self.optimizer.step()
            #self.scheduler.step()
            loss += nloss.item()
            # train log
            progress_bar.update(1)
            progress_bar.refresh()
            if batch_id % 100 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f'[{datetime.now()}]'
                    f'Train epoch [{epoch} / {num_epochs}],'
                    f'batch: [{batch_id} / {len(self.train_dataloader)}],'
                    f'loss: {(loss / (batch_id)):.5f},'
                    f'lr: {lr:.5f}''\n')
                f_train = open(os.path.join(self.logs_dir, 'train_log.txt'), 'a+')
                f_train.write(f'[{datetime.now()}]'
                    f'Train epoch [{epoch} / {num_epochs}],'
                    f'batch: [{batch_id} / {len(self.train_dataloader)}],'
                    f'loss: {(loss / (batch_id)):.5f},'
                    f'lr: {lr:.5f}''\n')
                f_train.close()
        self.scheduler.step()

    def _train_epoch_ns_amp(self, epoch, num_epochs):
        loss = 0.0
        progress_bar = tqdm.tqdm(total=len(self.train_dataloader), desc='Training')
        for batch_id, (mixAudio, cleanAudio) in enumerate(self.train_dataloader, start=1):
            #self.scheduler.step(epoch-1)
            self.optimizer.zero_grad()
            batchsize = mixAudio.shape[0]
            instates_hc_1 = torch.zeros(size=(2, batchsize, self.hidden_size, 2)).to(self.device)
            instates_hc_2 = torch.zeros(size=(2, batchsize, self.hidden_size, 2)).to(self.device)

            mixAudio = mixAudio.to(self.device)
            outAudio = self.model.forward(mixAudio, instates_hc_1, instates_hc_2)
            cleanAudio = torch.FloatTensor(cleanAudio).to(self.device)

            # calculate loss and update parameters
            self.loss_method = self.config['loss']['method']
            nloss = self.calloss(wavPred=outAudio, wavTrue=cleanAudio, method = self.loss_method)
            self.scaler.scale(nloss).backward()
            if self.clipset:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm_value, norm_type=2)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss += nloss.item()
            # train log
            progress_bar.update(1)
            progress_bar.refresh()
            if batch_id % 100 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f'[{datetime.now()}]'
                    f'Train epoch [{epoch} / {num_epochs}],'
                    f'batch: [{batch_id} / {len(self.train_dataloader)}],'
                    f'loss: {(loss / (batch_id)):.5f},'
                    f'lr: {lr:.5f}''\n')
                f_train = open(os.path.join(self.logs_dir, 'train_log.txt'), 'a+')
                f_train.write(f'[{datetime.now()}]'
                    f'Train epoch [{epoch} / {num_epochs}],'
                    f'batch: [{batch_id} / {len(self.train_dataloader)}],'
                    f'loss: {(loss / (batch_id)):.5f},'
                    f'lr: {lr:.5f}''\n')
                f_train.close()
        # self.optimizer.step() , self.scaler.step(self.optimizer) 已经能够代替了。
        self.scheduler.step()


    def _train_epoch_stateful_aec(self, epoch, num_epochs):
        loss = 0.0
        progress_bar = tqdm.tqdm(total=len(self.train_dataloader), desc='Training')
        for batch_id, (cleanData, micData, refData) in enumerate(self.train_dataloader, start=1):
            self.scheduler.step(epoch-1)
            self.optimizer.zero_grad()
            # initial State.
            batchsize = cleanData.shape[0]
            inState1 = torch.zeros(size=(2,batchsize, self.hiddenSize_RNN, 2)).to(self.device) # numLayers, batchsize, hiddensiceofrnn, 2. where 2 means h and c.
            inState2 = torch.zeros(size=(2,batchsize, self.hiddenSize_RNN, 2)).to(self.device)

            micData = micData.to(self.device)
            refData = refData.to(self.device)

            predData, _, _ = self.model(micData, refData, inState1, inState2)
            cleanData = cleanData.to(self.device)

            # calculate loss and update parameters
            nloss = self.calloss(wavPred=predData, wavTrue=cleanData)
            print("loss is :", nloss)
            nloss.backward()
            if self.clipset:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm_value, norm_type=2)
            self.optimizer.step()
            self.scheduler.step()
            loss += nloss.item()

            # train log
            progress_bar.update(1)
            progress_bar.refresh()
            if batch_id % 100 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f'[{datetime.now()}]'
                    f'Train epoch [{epoch} / {num_epochs}],'
                    f'batch: [{batch_id} / {len(self.train_dataloader)}],'
                    f'loss: {(loss / (batch_id)):.5f},'
                    f'lr: {lr:.5f}''\n')
                f_train = open(os.path.join(self.logs_dir, 'train_log.txt'), 'a+')
                f_train.write(f'[{datetime.now()}]'
                    f'Train epoch [{epoch} / {num_epochs}],'
                    f'batch: [{batch_id} / {len(self.train_dataloader)}],'
                    f'loss: {(loss / (batch_id)):.5f},'
                    f'lr: {lr:.5f}''\n')
                f_train.close()


    def train_aec(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print("[0 seconds]Begin training...")
            self.model.train()
            self._train_epoch_stateful_aec(epoch, self.epochs)
            self.model.eval()
            self._save_checkpoint(epoch)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print("[0 seconds]Begin training...")
            self.model.train()
            if self.config["meta"]["use_amp"]:
                self._train_epoch_ns_amp(epoch, self.epochs)
            else:
                self._train_epoch_ns(epoch, self.epochs)
            self.model.eval()
            self._save_checkpoint(epoch)


        





    