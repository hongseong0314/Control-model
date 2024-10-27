import pandas as pd

from sklearn.model_selection import train_test_split
from data.dataset import *

def make_dataloader(root_path):
    meta_df = pd.read_csv(root_path)

    # Flatten 공정 식별
    flatten_df = meta_df.dropna(subset=target_col_before + target_col_after, how='all')

    # Res 공정 식별
    res_df = meta_df.dropna(subset=all_res_cols, how='all')

    # Thk 공정 식별
    thk_df = meta_df.dropna(subset=all_thk_cols, how='all')

    # 각 공정별로 Train, Validation, Test 분할 (비율: 60% Train, 20% Validation, 20% Test)
    def split_data(df, test_size=0.2, valid_size=0.25, random_state=42):
        train_df, temp_df = train_test_split(df, test_size=test_size + valid_size, random_state=random_state)
        valid_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + valid_size), random_state=random_state)
        return train_df, valid_df, test_df

    flatten_train, flatten_valid, flatten_test = split_data(flatten_df)
    res_train, res_valid, res_test = split_data(res_df)
    thk_train, thk_valid, thk_test = split_data(thk_df)

    # 최종적으로 각 공정의 Train, Validation, Test 세트를 결합
    train_set = pd.concat([flatten_train, res_train, thk_train])
    valid_set = pd.concat([flatten_valid, res_valid, thk_valid])
    test_set = pd.concat([flatten_test, res_test, thk_test])

    # 결과 확인
    print(f"Train Set Size: {train_set.shape}")
    print(f"Validation Set Size: {valid_set.shape}")
    print(f"Test Set Size: {test_set.shape}")

    trainset = APCDataset(raw_df=train_set,)
    validset = APCDataset(raw_df=valid_set,)
    testset = APCDataset(raw_df=test_set,)
    return  trainset, validset, testset
