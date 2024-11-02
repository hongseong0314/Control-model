import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sub_model import PeriodicEmbeddings, PositionalEncoding

class APCforemr(nn.Module):
    def __init__(self, shared_input_dim, flatten_input_dim, res_input_dim, thk_input_dim, 
                 flatten_y_dims, res_y_dims, thk_y_dims, 
                 embedding_dim=32):
        super(APCforemr, self).__init__()
        total_dim = shared_input_dim + (flatten_input_dim+5) + thk_input_dim
        # 공유 X를 처리하는 레이어
        self.shared_layer = PeriodicEmbeddings(n_features=shared_input_dim, 
                                               d_embedding=embedding_dim, lite=False)

        # Flatten 공정의 레이어
        self.flatten_layer = PeriodicEmbeddings(n_features=flatten_input_dim+5, 
                                                d_embedding=embedding_dim, lite=False)

        # Res and Thk 공정의 레이어
        self.thk_and_res_layer = PeriodicEmbeddings(n_features=thk_input_dim, 
                                                    d_embedding=embedding_dim, 
                                                    lite=False)
            
        # 학습 가능한 특수 토큰 임베딩 생성
        self.thk_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))      # (1, 1, E)
        self.flatten_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))  # (1, 1, E)
        self.res_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))  
        self.pos_encoder = PositionalEncoding(embedding_dim)
        nhead = max(1, embedding_dim // 16) 
        # self.backborn = nn.Sequential(nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, 
        #                                dim_feedforward=embedding_dim*2, 
        #                                dropout=0.1,batch_first=True,
        #                                activation='gelu'),
        #     num_layers=3),
        # nn.Flatten(),
        # )
        encoder_layer = PreLNTransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, 
                                                     dim_feedforward=embedding_dim*4, 
                                                     dropout=0.2, activation='relu')
        self.backborn = PreLNTransformerEncoder(encoder_layer, num_layers=3)
        
        # 각 공정의 최종 출력 헤드 (Flatten, Res, Thk)
        self.flatten_head = nn.Linear(embedding_dim, flatten_y_dims)
        self.res_head = nn.Linear(embedding_dim, res_y_dims)
        self.thk_head = nn.Linear(embedding_dim, thk_y_dims)

        self.split_sizes = [shared_input_dim, flatten_input_dim + 5, thk_input_dim]

    def forward(self, sample_batch):
        
        shared_X = torch.cat((sample_batch['share_control'], sample_batch['share_not_control']), dim=1)
        thk_and_res_X = torch.cat((sample_batch['thk_control'], 
                                   sample_batch['thk_not_control']), dim=1)
        
        flatten_X = sample_batch['flatten_not_control']
        flatten_pred_y = torch.nan_to_num(sample_batch['target_flatten_before'], nan=-1)
        flatten_X_and_before = torch.cat((flatten_X, flatten_pred_y), dim=1)
        
        # Embedding
        shared_out = self.shared_layer(shared_X)
        flatten_out = self.flatten_layer(flatten_X_and_before)
        thk_and_res_out = self.thk_and_res_layer(thk_and_res_X)

        # 배치 크기 가져오기
        B = shared_out.size(0)

        # 특수 토큰을 배치 크기에 맞게 복제
        thk_token = self.thk_token.expand(B, 1, -1)       # (1, B, E)
        flatten_token = self.flatten_token.expand(B, 1, -1) # (1, B, E)
        res_token = self.res_token.expand(B, 1, -1)      # (1, B, E)

        # embedding_x = torch.cat((shared_out, flatten_out, thk_and_res_out), dim=1)
        embedding_x = torch.cat((thk_token, flatten_token, res_token, shared_out, flatten_out, thk_and_res_out), dim=1)  # (seq_len, B, E)

        # embedding_x = embedding_x.permute(1, 0, 2)  # (3F, B, E)

        # # 포지셔널 인코딩 추가
        # embedding_x = self.pos_encoder(embedding_x)
        # 각 공정의 독립 X와 공유 결과를 결합하여 처리
        out = self.backborn(embedding_x.transpose(0, 1))
        # out_split = torch.split(out, self.split_sizes, dim=0)  # tuple of 3 tensors, 각 (F_i, B, E)
        
        # # out_means = [part.mean(dim=0) for part in out_split]  # list of 3 tensors, each (B, E)
        # flatten_out = torch.cat((out_means[0], out_means[1]), dim=1)
        # res_and_thk_out = torch.cat((out_means[0], out_means[2]), dim=1)
        # print(res_and_thk_out.shape)
        # print(flatten_out.shape)
        thk_output = out[0, :, :]        # (B, E)
        flatten_output = out[1, :, :]    # (B, E)
        res_output = out[2, :, :]        # (B, E)

        # 각 헤드에 전달하여 최종 출력 생성
        flatten_result = self.flatten_head(flatten_output)  # (B, flatten_y_dims)
        thk_result = self.thk_head(thk_output)      # (B, thk_y_dims)
        res_result = self.res_head(res_output)  

        # out = out.mean(dim=0)
        # flatten_result = self.flatten_head(out)
        # thk_result = self.thk_head(out)
        # res_result = self.res_head(out)

        return {
            'flatten': flatten_result,
            'res': res_result,
            'thk': thk_result
        }

class PreLNTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(PreLNTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward 네트워크
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 레이어 정규화
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 활성화 함수
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 서브 레이어 전에 LayerNorm 적용
        src2 = self.norm1(src)
        attn_output, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(attn_output)
        
        # 서브 레이어 전에 LayerNorm 적용
        src2 = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout(ff_output)
        return src

class PreLNTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(PreLNTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output



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
            PeriodicEmbeddings(n_features=(flatten_input_dim+5), d_embedding=embedding_dim, lite=False),
            nn.Flatten(),
            nn.Linear(embedding_dim*(flatten_input_dim+5), embedding_dim),
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

    def forward(self, sample_batch):
        
        shared_X = torch.cat((sample_batch['share_control'], sample_batch['share_not_control']), dim=1)
        thk_and_res_X = torch.cat((sample_batch['thk_control'], sample_batch['thk_not_control']), dim=1)
        
        flatten_X = sample_batch['flatten_not_control']
        flatten_pred_y = torch.nan_to_num(sample_batch['target_flatten_before'], nan=-1)
        flatten_X_and_before = torch.cat((flatten_X, flatten_pred_y), dim=1)
        # 공유 X를 처리
        shared_out = self.shared_layer(shared_X)

        # 각 공정의 독립 X와 공유 결과를 결합하여 처리
        flatten_out = self.flatten_layer(flatten_X_and_before)
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