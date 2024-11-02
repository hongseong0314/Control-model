import torch
import math
import random
import numpy as np

from trainer import Trainer
from config import cfg
from data.loaddata import make_dataloader

if __name__ == '__main__':
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED) 
    root_dir = r'data/df_전체데이터.csv'
    train_set, valid_set, test_set = make_dataloader(root_dir)
    cfg['train_set'] = train_set; cfg['valid_set'] = valid_set; cfg['test_set'] = test_set;
    trainer = Trainer(cfg) 
    trainer.setup()

    trainer.run()

    # contorl 해보기
    # trainer.control_X()