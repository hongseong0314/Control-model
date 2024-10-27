import torch
import torch.nn as nn
from model.sub_model import PeriodicEmbeddings

class APCMLP(nn.Module):
    def __init__(self, shared_input_dim, flatten_input_dim, res_input_dim, thk_input_dim, 
                 flatten_y_dims, res_y_dims, thk_y_dims, 
                 embedding_dim=32):
        super(APCMLP, self).__init__()

        # 공유 X를 처리하는 레이어
        self.shared_layer = nn.Sequential(
            PeriodicEmbeddings(n_features=shared_input_dim, d_embedding=embedding_dim, lite=False),
            nn.Flatten(),
            nn.Linear(shared_input_dim * embedding_dim, embedding_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Flatten 공정의 레이어
        self.flatten_layer = nn.Sequential(
            PeriodicEmbeddings(n_features=flatten_input_dim, d_embedding=embedding_dim, lite=False),
            nn.Flatten(),
            nn.Linear(embedding_dim*flatten_input_dim, embedding_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Res 공정의 레이어
        self.res_layer = nn.Sequential(
            PeriodicEmbeddings(n_features=thk_input_dim, d_embedding=embedding_dim, lite=False),
            nn.Flatten(),
            nn.Linear(embedding_dim*res_input_dim, embedding_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Thk 공정의 레이어
        self.thk_layer = nn.Sequential(
            PeriodicEmbeddings(n_features=thk_input_dim, d_embedding=embedding_dim, lite=False),
            nn.Flatten(),
            nn.Linear(embedding_dim*thk_input_dim, embedding_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # 각 공정의 최종 출력 헤드 (Flatten, Res, Thk)
        self.flatten_head = nn.Linear(embedding_dim*2, flatten_y_dims)
        self.res_head = nn.Linear(embedding_dim*2, res_y_dims)
        self.thk_head = nn.Linear(embedding_dim*2, thk_y_dims)

    def forward(self, shared_X, flatten_X, thk_and_res_X):
        # 공유 X를 처리
        shared_out = self.shared_layer(shared_X)

        # 각 공정의 독립 X와 공유 결과를 결합하여 처리
        flatten_out = self.flatten_layer(flatten_X)
        flatten_input = torch.cat((shared_out, flatten_out), dim=1)
        flatten_result = self.flatten_head(flatten_input)

        thk_out = self.thk_layer(thk_and_res_X)
        thk_input = torch.cat((shared_out, thk_out), dim=1)
        thk_result = self.thk_head(thk_input)

        res_out = self.res_layer(thk_and_res_X)
        res_input = torch.cat((shared_out, res_out), dim=1)
        res_result = self.res_head(res_input)


        return {
            'flatten': flatten_result,
            'res': res_result,
            'thk': thk_result
        }