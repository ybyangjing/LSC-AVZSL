# system, numpy
import os
import sys
import numpy as np
import math
from einops import rearrange, repeat
import einops
import opt_einsum
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# user defined
from src.optimizer import SAM
from einops.layers.torch import Rearrange
torch.set_printoptions(threshold=10_000)
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)





class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, hidden_size=-1):
        super(EmbeddingNet, self).__init__()
        modules = []

        if hidden_size > 0:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)




class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class AutoFusion(nn.Module):
    """docstring for AutoFusion"""
    def __init__(self, dim, hidden_dim):
        super(AutoFusion, self).__init__()

        self.fuse_in = nn.Sequential(
            nn.Linear(dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, dim),
            nn.ReLU()
            )
        self.fuse_out = nn.Sequential(
            nn.Linear(dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, dim)
            )
        self.criterion = nn.MSELoss()

    def forward(self, z):
        compressed_z = self.fuse_in(z)
        loss = self.criterion(self.fuse_out(compressed_z), z)
        return compressed_z,loss

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, input_vectors, target_vectors):
        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(input_vectors, target_vectors)

        # 构建损失函数，使余弦相似度接近1
        loss = 1 - cosine_similarity

        return loss.mean()

class MLP_block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MLP_Communicator(nn.Module):
    def __init__(self, token, channel, hidden_size, depth=1):
        super(MLP_Communicator, self).__init__()
        self.depth = depth
        self.token_mixer = nn.Sequential(
            Rearrange('b n d -> b d n'),
            MLP_block(input_size=channel, hidden_size=hidden_size),
            Rearrange('b n d -> b d n')
        )
        self.channel_mixer = nn.Sequential(
            MLP_block(input_size=token, hidden_size=hidden_size)
        )

    def forward(self, x):
        for _ in range(self.depth):
            x = x + self.token_mixer(x)
            x = x + self.channel_mixer(x)
        return x


class ClipClap_model(nn.Module):
    def __init__(self, params_model, input_size_audio, input_size_video):
        super(ClipClap_model, self).__init__()

        print('Initializing model variables...', end='')
        # Dimension of embedding
        self.dim_out = params_model['dim_out']
        self.input_dim_audio = input_size_audio
        self.input_dim_video = input_size_video

        self.hidden_size_decoder=params_model['decoder_hidden_size']
        self.drop_proj_o=params_model['dropout_decoder']
        self.drop_proj_w=params_model['additional_dropout']
        self.reg_loss=params_model['reg_loss']
        self.cross_entropy_loss=params_model['cross_entropy_loss']
        self.hidden_size_encoder=params_model['encoder_hidden_size']
        self.drop_enc=params_model['dropout_encoder']


        self.rec_loss = params_model['rec_loss']

        self.lr_scheduler = params_model['lr_scheduler']

        print('Initializing trainable models...', end='')


        self.modality = params_model['modality']
        self.word_embeddings = params_model['word_embeddings']

        if self.modality == 'audio':
            self.O_enc = EmbeddingNet(
                input_size=1024,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
            self.W_enc = EmbeddingNet(
                input_size=1024,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
        elif self.modality == 'video':
            self.O_enc = EmbeddingNet(
                input_size=512,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
            self.W_enc = EmbeddingNet(
                input_size=512,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
        else:
            self.O_enc = EmbeddingNet(
                input_size=1536,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
            w_in_dim = 1536
            if self.word_embeddings == 'wavcaps':
                w_in_dim = 1024
            elif self.word_embeddings == 'clip':
                w_in_dim = 512

            self.W_enc = EmbeddingNet(
                input_size=w_in_dim,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )




        word_embedding_dim = 512
        self.O_proj = EmbeddingNet(
            input_size=512,
            hidden_size=self.hidden_size_decoder,
            output_size=self.dim_out,
            dropout=self.drop_proj_o,
            use_bn=params_model['embeddings_batch_norm']
        )
        self.D_o = EmbeddingNet(
            input_size=self.dim_out,
            hidden_size=self.hidden_size_decoder,
            output_size=word_embedding_dim,
            dropout=self.drop_proj_o,
            use_bn=params_model['embeddings_batch_norm']
        )


        self.W_proj= EmbeddingNet(
            input_size=word_embedding_dim,
            output_size=self.dim_out,
            dropout=self.drop_proj_w,
            use_bn=params_model['embeddings_batch_norm']
        )

        self.D_w = EmbeddingNet(
            input_size=self.dim_out,
            output_size=word_embedding_dim,
            dropout=self.drop_proj_w,
            use_bn=params_model['embeddings_batch_norm']
        )
        self.cross_attention = Transformer(512, 6, 8, 64, 512, dropout=0.2)
        self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 512))
        self.fu = AutoFusion(
            dim=512,
            hidden_dim=512,
        )
        self.mlp = MLP_Communicator(
            token=512,  # token 的大小
            channel=1,  # 通道的大小
            hidden_size=64,  # 隐藏层的大小
            depth=1  # 深度
        )









        # Optimizers
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']

        optimizer = params_model['optimizer']
        self.is_sam_optim = False
        if optimizer == 'adam':
            self.optimizer_gen = optim.Adam(
                self.parameters(),
                lr=self.lr, weight_decay=1e-5
            )
            if self.lr_scheduler:
                self.scheduler_learning_rate =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max', patience=3, verbose=True)

        elif optimizer == 'adam-sam':
            self.optimizer_gen = SAM(self.parameters(), optim.Adam, lr=self.lr, weight_decay=1e-5)
            self.is_sam_optim = True
            if self.lr_scheduler:
                # lr scheduling on base optimizer
                self.scheduler_learning_rate =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen.base_optimizer, 'max', patience=3, verbose=True)
        else:
            raise NotImplementedError

        print('Done')

        # Loss function
        print('Defining losses...', end='')
        self.criterion_cyc = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        print('Done')

    def optimize_scheduler(self, value):
        if self.lr_scheduler:
            self.scheduler_learning_rate.step(value)

    def forward(self, a, v, w, masks, timesteps):
        b, _ = a.shape
        device = a.device
        v = v.type(torch.float32)
        if self.modality == 'audio':
            w = w[:,512:]
            model_input = a

        elif self.modality == 'video':
            w = w[:,:512]
            model_input = v
        else:
            if self.word_embeddings == 'wavcaps':
                w = w[:,512:]
            elif self.word_embeddings == 'clip':
                w = w[:,:512]
            model_input = torch.cat((v, a), dim=1)


        o = self.O_enc(model_input)
        # o = o.unsqueeze(1)
        # #print(o.shape)
        # o = self.mlp(o).squeeze(1)

        m, l = self.fu(o)
        m = m + self.pos_emb1D[0, :]
        m = m.unsqueeze(1)
        #print(m.shape)
        o = self.cross_attention(m).squeeze(1)
        o = o.unsqueeze(1)
        # print(o.shape)s
        o = self.mlp(o).squeeze(1)

        w = self.W_enc(w)



        theta_o = self.O_proj(o)


        rho_o = self.D_o(theta_o)


        theta_w = self.W_proj(w)


        rho_w=self.D_w(theta_w)


        output = {
            "theta_w": theta_w,
            "w": w,
            "rho_w": rho_w,
            "theta_o": theta_o,
            "rho_o": rho_o,
            "l_m": l,
        }


        return output


    def compute_loss(self, outputs, embeddings_crossentropy, gt_cross_entropy):

        l_m = outputs['l_m']
        theta_w = outputs['theta_w']

        w = outputs['w']
        rho_w = outputs['rho_w']

        theta_o = outputs['theta_o']

        rho_o = outputs['rho_o']


        device = theta_w.device

        if self.cross_entropy_loss==True:
            if self.modality == 'audio':
                embeddings_crossentropy = embeddings_crossentropy[:,512:]
            elif self.modality == 'video':
                embeddings_crossentropy = embeddings_crossentropy[:,:512]
            else:
                if self.word_embeddings == 'wavcaps':
                    embeddings_crossentropy = embeddings_crossentropy[:,512:]
                elif self.word_embeddings == 'clip':
                    embeddings_crossentropy = embeddings_crossentropy[:,:512]

            embedding_cross_entropy=self.W_proj(self.W_enc(embeddings_crossentropy))
            Cross_loss=nn.CrossEntropyLoss()
            scores=torch.matmul(theta_o, embedding_cross_entropy.t()) # (bs, 64) x (K_seen, 64).T = (bs, 64) x (64, K_seen) = (bs, K_seen)
            # gt_cross_entropy = [1, 3, 2, 55, 97, 45, ...] list of gt class labels -> shape (bs,)
            l_ce=Cross_loss(scores, gt_cross_entropy)
        else:
            l_ce = torch.tensor(0., device=device)

        if self.reg_loss==True:
            l_reg = (
                self.MSE_loss(theta_o, theta_w)
            )
        else:
            l_reg = torch.tensor(0., device=device)



        if self.rec_loss == True:
            l_rec = (
                    self.MSE_loss(w, rho_o) +
                    self.MSE_loss(w, rho_w)
            )
        else:
            l_rec = torch.tensor(0., device=device)
        kl_div = nn.KLDivLoss(reduction='batchmean')  # KL散度的PyTorch实现
        p = torch.softmax(theta_o, dim=-1)  # 模型的预测分布
        q = torch.softmax(theta_w, dim=-1)  # 目标分布
        l_kl = kl_div(p.log(), q)
        cosine_loss = CosineSimilarityLoss()
        l_cos = cosine_loss(theta_o, theta_w)

        #loss_total = l_rec+l_reg+l_ce + l_kl + l_cos*0.5+l_m*0.5
        #loss_total = l_rec + l_reg + l_ce + l_kl + l_m*0.6
        loss_total = l_rec + l_reg + l_ce + l_kl + l_m * 0.5
        #loss_total = l_rec+l_reg+l_ce #+ l_kl + l_cos*0.7
        loss_dict = {
            "Loss/total_loss": loss_total.detach().cpu(),
            "Loss/loss_reg": l_reg.detach().cpu(),
            "Loss/loss_cmd_rec": l_rec.detach().cpu(),
            "Loss/cross_entropy": l_ce.detach().cpu()

        }
        return loss_total, loss_dict

    # cls_numeric = class index
    # cls_embedding = w2v embedding of the target
    def optimize_params(self, audio, video, cls_numeric, cls_embedding, masks, timesteps, embedding_crossentropy, optimize=False):
        if not self.is_sam_optim:
            # Forward pass
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)

            # Backward pass
            loss_numeric, loss = self.compute_loss(outputs, embedding_crossentropy,  cls_numeric)

            if optimize == True:
                self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.step()

        else:
            # SAM optimizer requires two forward / backward

            enable_running_stats(self)
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)
            loss_numeric, loss = self.compute_loss(outputs, embedding_crossentropy,  cls_numeric)

            if optimize:
                # first forward-backward step
                # self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(self)
                outputs_second = self.forward(audio, video, cls_embedding, masks, timesteps)
                second_loss, _ = self.compute_loss(outputs_second, embedding_crossentropy,  cls_numeric)
                second_loss.backward()
                self.optimizer_gen.second_step(zero_grad=True)

        return loss_numeric, loss

    def get_embeddings(self, a, v, w, masks, timesteps):
        b, _ = a.shape
        device = a.device
        v = v.type(torch.float32)



        if self.modality == 'audio':
            w = w[:,512:]
            model_input = a

        elif self.modality == 'video':
            w = w[:,:512]
            model_input = v
        else:
            if self.word_embeddings == 'wavcaps':
                w = w[:,512:]
            elif self.word_embeddings == 'clip':
                w = w[:,:512]
            model_input = torch.cat((v, a), dim=1)


        o = self.O_enc(model_input)
        # o = o.unsqueeze(1)
        # #print(o.shape)s
        # o = self.mlp(o).squeeze(1)
        m, l = self.fu(o)
        m = m + self.pos_emb1D[0, :]
        m = m.unsqueeze(1)
        o = self.cross_attention(m).squeeze(1)
        o = o.unsqueeze(1)
        # print(o.shape)s
        o = self.mlp(o).squeeze(1)
        w = self.W_enc(w)



        theta_o = self.O_proj(o)

        theta_w=self.W_proj(w)

        return theta_o, theta_o, theta_w