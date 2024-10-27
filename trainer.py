import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # tqdm 불러오기
from datetime import datetime
from sklearn.metrics import r2_score
from model.model import APCMLP
import matplotlib.pyplot as plt
# from data.dataset import APCDataset

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def setup(self):
        self.VM_model = APCMLP(**self.cfg['model_cfg'])
        self.criterion = nn.L1Loss(reduction='none')
        self.optimizer = optim.Adam(self.VM_model.parameters(), lr=self.cfg['lr'])  
        # self.warmup_scheduler = WarmUpLR(self.optimizer, iter_per_epoch * self.args.warm_epoch)

        # train, validation, test DataLoader 생성
        self.train_loader = DataLoader(self.cfg['train_set'], batch_size=self.cfg['batch_size'], shuffle=True)
        self.valid_loader = DataLoader(self.cfg['valid_set'], batch_size=self.cfg['batch_size'], shuffle=False)
        self.test_loader = DataLoader(self.cfg['test_set'], batch_size=self.cfg['batch_size'], shuffle=False)

        now = datetime.now()
        dir_name = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = os.path.join(r'result', dir_name)
        os.makedirs(self.save_dir, exist_ok=True)
        # os.makedirs(f'{self.output_path}/checkpoint')
    
    def run(self):
        num_epochs = self.cfg['epochs']
        epoch_val_loss, epoch_val_loss = 0, 0
        train_losses = []
        val_losses = []
        train_r2_scores_list = {'flatten': [], 'res': [], 'thk': []}
        val_r2_scores_list = {'flatten': [], 'res': [], 'thk': []}
        for epoch in tqdm(range(num_epochs), desc="Epochs Progress", leave=True):
            epoch_train_loss, train_r2_scores  = self.train_one_epoch(epoch)
            if (epoch + 1) % 5 == 0:
                epoch_val_loss, val_r2_scores = self.valid_one_epoch(epoch)
            else:
                val_r2_scores = {'flatten': np.nan, 'res': np.nan, 'thk': np.nan}
                epoch_val_loss = np.nan
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            for key in ['flatten', 'res', 'thk']:
                train_r2_scores_list[key].append(train_r2_scores.get(key, np.nan))
                val_r2_scores_list[key].append(val_r2_scores.get(key, np.nan))
            
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
        # model save
        checkpoiont_dict = {
                        'epoch':epoch,
                        'model_state_dict': self.VM_model.state_dict(),
                    }
        torch.save(checkpoiont_dict, f'{self.save_dir}/checkpoint-{epoch}.pt')
        self.plot_curve(train_r2_scores_list, val_r2_scores_list, train_losses, val_losses)
    
    def train_one_epoch(self, epoch):
        self.VM_model.train()
        running_loss = 0.0
        y_trues = {'flatten': [], 'res': [], 'thk': []}
        y_preds = {'flatten': [], 'res': [], 'thk': []}
        for sample_batch in self.train_loader:
            self.optimizer.zero_grad()

            shared_X = torch.cat((sample_batch['share_control'], sample_batch['share_not_control']), dim=1)
            flatten_X = sample_batch['flatten_not_control']
            thk_X = torch.cat((sample_batch['thk_control'], sample_batch['thk_not_control']), dim=1)
            
            # Forward pass
            outputs = self.VM_model(
                shared_X=shared_X,
                flatten_X=flatten_X,
                # res_X=batch['res_X'],
                thk_and_res_X=thk_X
            )

            # Masked Loss 계산 (Flatten, Res, Thk 공정별로 계산 후 합산)
            loss_flatten = torch.nan_to_num(self.criterion(outputs['flatten'], sample_batch['target_flatten_before']), nan=0.0)
            loss_res = torch.nan_to_num(self.criterion(outputs['res'], sample_batch['target_res']), nan=0.0)
            loss_thk = torch.nan_to_num(self.criterion(outputs['thk'], sample_batch['target_thk']), nan=0.0)

            total_loss = loss_flatten.mean(dim=1) + loss_res.mean(dim=1) + loss_thk.mean(dim=1)
            total_loss = total_loss.mean()

            # Backward pass 및 옵티마이저 스텝
            total_loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                mask_flatten = ~torch.isnan(sample_batch['target_flatten_before'])
                y_true_flatten = sample_batch['target_flatten_before'][mask_flatten]
                y_pred_flatten = outputs['flatten'][mask_flatten]
                y_trues['flatten'].append(y_true_flatten.cpu())
                y_preds['flatten'].append(y_pred_flatten.cpu())

                # Res
                mask_res = ~torch.isnan(sample_batch['target_res'])
                y_true_res = sample_batch['target_res'][mask_res]
                y_pred_res = outputs['res'][mask_res]
                y_trues['res'].append(y_true_res.cpu())
                y_preds['res'].append(y_pred_res.cpu())

                # Thk
                mask_thk = ~torch.isnan(sample_batch['target_thk'])
                y_true_thk = sample_batch['target_thk'][mask_thk]
                y_pred_thk = outputs['thk'][mask_thk]
                y_trues['thk'].append(y_true_thk.cpu())
                y_preds['thk'].append(y_pred_thk.cpu())

            running_loss += total_loss.item()
        r2_scores = {}
        for key in ['flatten', 'res', 'thk']:
            y_true = torch.cat(y_trues[key], dim=0).numpy()
            y_pred = torch.cat(y_preds[key], dim=0).numpy()
            if y_true.size > 0:
                r2_scores[key] = r2_score(y_true, y_pred)
            else:
                r2_scores[key] = np.nan  # 데이터가 없는 경우 NaN 처리

        return running_loss / len(self.train_loader), r2_scores
    
    def valid_one_epoch(self, epoch):
        self.VM_model.eval()
        val_loss = 0.0
        y_trues = {'flatten': [], 'res': [], 'thk': []}
        y_preds = {'flatten': [], 'res': [], 'thk': []}
        with torch.no_grad():
            for sample_batch in self.valid_loader:
                shared_X = torch.cat((sample_batch['share_control'], sample_batch['share_not_control']), dim=1)
                flatten_X = sample_batch['flatten_not_control']
                thk_X = torch.cat((sample_batch['thk_control'], sample_batch['thk_not_control']), dim=1)
                
                outputs = self.VM_model(
                    shared_X=shared_X,
                    flatten_X=flatten_X,
                    # res_X=batch['res_X'],
                    thk_and_res_X=thk_X
                )

                loss_flatten = torch.nan_to_num(self.criterion(outputs['flatten'], sample_batch['target_flatten_before']), nan=0.0)
                loss_res = torch.nan_to_num(self.criterion(outputs['res'], sample_batch['target_res']), nan=0.0)
                loss_thk = torch.nan_to_num(self.criterion(outputs['thk'], sample_batch['target_thk']), nan=0.0)

                total_loss = loss_flatten.mean(dim=1) + loss_res.mean(dim=1) + loss_thk.mean(dim=1)
                total_loss = total_loss.mean()
                val_loss += total_loss.item()

                # R2 스코어 계산을 위한 실제 값과 예측 값 저장
                # Flatten
                mask_flatten = ~torch.isnan(sample_batch['target_flatten_before'])
                y_true_flatten = sample_batch['target_flatten_before'][mask_flatten]
                y_pred_flatten = outputs['flatten'][mask_flatten]
                y_trues['flatten'].append(y_true_flatten.cpu())
                y_preds['flatten'].append(y_pred_flatten.cpu())

                # Res
                mask_res = ~torch.isnan(sample_batch['target_res'])
                y_true_res = sample_batch['target_res'][mask_res]
                y_pred_res = outputs['res'][mask_res]
                y_trues['res'].append(y_true_res.cpu())
                y_preds['res'].append(y_pred_res.cpu())

                # Thk
                mask_thk = ~torch.isnan(sample_batch['target_thk'])
                y_true_thk = sample_batch['target_thk'][mask_thk]
                y_pred_thk = outputs['thk'][mask_thk]
                y_trues['thk'].append(y_true_thk.cpu())
                y_preds['thk'].append(y_pred_thk.cpu())
        r2_scores = {}
        for key in ['flatten', 'res', 'thk']:
            y_true = torch.cat(y_trues[key], dim=0).numpy()
            y_pred = torch.cat(y_preds[key], dim=0).numpy()
            if y_true.size > 0:
                r2_scores[key] = r2_score(y_true, y_pred)
            else:
                r2_scores[key] = np.nan 


        return val_loss / len(self.valid_loader), r2_scores
    
    def control_X(self):
        # sample data
        test_sample = self.test_loader.__iter__().__next__()

        thk_control = test_sample['thk_control'].clone().detach().requires_grad_(True)
        thk_not_control = test_sample['thk_not_control'].detach()

        share_control = test_sample['share_control'].clone().detach().requires_grad_(True)
        share_not_control = test_sample['share_not_control'].detach()

        flatten_X = test_sample['flatten_not_control'].detach()

        # Optimizer 설정
        optimizer = optim.Adam(
            [{'params': [share_control]}, {'params': [thk_control]}], 
            lr=0.01
        )
        criterion = nn.L1Loss(reduction='none')

        self.VM_model.eval()
        for step in range(10): # control step
            optimizer.zero_grad()
            
            thk_X = torch.cat((thk_control, thk_not_control), dim=1)
            thk_X.requires_grad_(True)

            share_X = torch.cat((share_control, share_not_control), dim=1)
            share_X.requires_grad_(True)
            
            # 업데이트된 입력으로 예측 수행
            outputs = self.VM_model(share_X, flatten_X, thk_X)  
            
            # 손실 계산
            # after가 원하는 품질이라고 이해함 ->
            loss_flatten = torch.nan_to_num(criterion(outputs['flatten'], test_sample['target_flatten_after']), nan=0.0)
            loss_res = torch.nan_to_num(criterion(outputs['res'], test_sample['target_res']), nan=0.0)
            loss_thk = torch.nan_to_num(criterion(outputs['thk'], test_sample['target_thk']), nan=0.0)

            total_loss = loss_flatten.mean(dim=1) + loss_res.mean(dim=1) + loss_thk.mean(dim=1)
            loss = total_loss.mean()
            
            # 역전파
            loss.backward()  
            
            # 컨트롤 피처 업데이트
            optimizer.step()

        # 최종 조정된 control feature 확인
        print(f"origin Control Features = {test_sample['share_control']}")
        print(f"Final Adjusted Control Features = {share_control}")
    
    def plot_curve(self, train_r2_scores_list, val_r2_scores_list, train_losses, val_losses):
        num_epochs = self.cfg['epochs']
     
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        # Flatten 공정 R2 스코어
        axs[0, 0].plot(range(1, num_epochs + 1), [train_r2_scores_list['flatten'][i] for i in range(num_epochs)], label='Train R2 Flatten')
        axs[0, 0].plot(range(1, num_epochs + 1), [val_r2_scores_list['flatten'][i] for i in range(num_epochs)], label='Val R2 Flatten')
        axs[0, 0].set_title('Flatten R2 Score')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('R2 Score')
        axs[0, 0].legend()

        # Res 공정 R2 스코어
        axs[0, 1].plot(range(1, num_epochs + 1), [train_r2_scores_list['res'][i] for i in range(num_epochs)], label='Train R2 Res')
        axs[0, 1].plot(range(1, num_epochs + 1), [val_r2_scores_list['res'][i] for i in range(num_epochs)], label='Val R2 Res')
        axs[0, 1].set_title('Res R2 Score')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('R2 Score')
        axs[0, 1].legend()

        # Thk 공정 R2 스코어
        axs[0, 2].plot(range(1, num_epochs + 1), [train_r2_scores_list['thk'][i] for i in range(num_epochs)], label='Train R2 Thk')
        axs[0, 2].plot(range(1, num_epochs + 1), [val_r2_scores_list['thk'][i] for i in range(num_epochs)], label='Val R2 Thk')
        axs[0, 2].set_title('Thk R2 Score')
        axs[0, 2].set_xlabel('Epoch')
        axs[0, 2].set_ylabel('R2 Score')
        axs[0, 2].legend()

        # 기존의 손실 그래프는 아래에 배치
        # Training Loss
        axs[1, 0].plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
        axs[1, 0].set_title('Training Loss')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()

        # Validation Loss
        axs[1, 1].plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        axs[1, 1].set_title('Validation Loss')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_plots_per_process.png')