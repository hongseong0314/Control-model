import torch
import numpy as np
from torch.utils.data import Dataset
import copy

share_control_col = ['control_factor_1', 'control_factor_2', 'control_factor_3',
                        'control_factor_4', 'control_factor_5', 'control_factor_6',
                        'control_factor_7', 'control_factor_8', 'control_factor_9']
share_not_control_col = ['x_factor_1', 'x_factor_2', 'x_factor_3','x_factor_4']

## flatten
flatten_not_control_col = ['x_factor_5', 'x_factor_6', 'x_factor_7','x_factor_8','x_factor_9']

# thk and res
thk_control_col = ['control_factor_10']
thk_not_control_col = ['x_factor_10']

# y 
target_col_before = ['y_factor_1_before', 'y_factor_2_before', 'y_factor_4_before',
                    'y_factor_5_before', 'y_factor_6_before']
target_col_after = ['y_factor_1_after', 'y_factor_2_after', 'y_factor_4_after',
                    'y_factor_5_after', 'y_factor_6_after']
all_res_cols = [f'RES_y_factor_{i}' for i in range(1, 10)]
all_thk_cols = [f'THK_y_factor_{i}' for i in range(1, 32)]

# PyTorch Dataset 클래스 정의
class APCDataset(Dataset):
    ## share
    
    def __init__(self, raw_df, device, nan_replace=-1):
        self.device = device
        df = copy.deepcopy(raw_df)
        df[share_control_col + share_not_control_col + \
                flatten_not_control_col+\
                thk_control_col + thk_not_control_col] = df[share_control_col + share_not_control_col + \
                                                        flatten_not_control_col+\
                                                        thk_control_col + thk_not_control_col].fillna(nan_replace)

        # 공유 X 데이터
        self.share_control = df[share_control_col].values  
        self.share_not_control = df[share_not_control_col].values 
        
        self.flatten_not_control = df[flatten_not_control_col].values  # Flatten 독립 X 데이터
        self.thk_control = df[thk_control_col].values  # Thk and Res 독립 X 데이터
        self.thk_not_control = df[thk_not_control_col].values  # Thk and Res 독립 X 데이터
        
        self.target_flatten_before = df[target_col_before].values
        self.target_flatten_after = df[target_col_after].values
        self.target_res = df[all_res_cols].values
        self.target_thk = df[all_thk_cols].values

        # NaN 여부를 알려주는 마스크 생성 (Y가 NaN이면 0, 아니면 1)
        self.mask_flatten_before = (~np.isnan(self.target_flatten_before)).astype(int)
        self.mask_flatten_after = (~np.isnan(self.target_flatten_after)).astype(int)
        self.mask_res = (~np.isnan(self.target_res)).astype(int)
        self.mask_thk = (~np.isnan(self.target_thk)).astype(int)

    def __len__(self):
        return len(self.share_control)

    def __getitem__(self, idx):
        return {
            'share_control': torch.tensor(self.share_control[idx], dtype=torch.float32),
            'share_not_control': torch.tensor(self.share_not_control[idx], dtype=torch.float32),
            'flatten_not_control': torch.tensor(self.flatten_not_control[idx], dtype=torch.float32),
            'thk_control': torch.tensor(self.thk_control[idx], dtype=torch.float32).to(self.device),
            'thk_not_control': torch.tensor(self.thk_not_control[idx], dtype=torch.float32),

            'target_flatten_before': torch.tensor(self.target_flatten_before[idx], dtype=torch.float32),
            'target_flatten_after': torch.tensor(self.target_flatten_after[idx], dtype=torch.float32),
            'target_res': torch.tensor(self.target_res[idx], dtype=torch.float32),
            'target_thk': torch.tensor(self.target_thk[idx], dtype=torch.float32),

            'mask_flatten_before': torch.tensor(self.mask_flatten_before[idx], dtype=torch.float32),
            'mask_flatten_after': torch.tensor(self.mask_flatten_after[idx], dtype=torch.float32),
            'mask_res': torch.tensor(self.mask_res[idx], dtype=torch.float32),
            'mask_thk': torch.tensor(self.mask_thk[idx], dtype=torch.float32),
        }