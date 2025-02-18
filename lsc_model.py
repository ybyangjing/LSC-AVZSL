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
import hyptorch.pmath as hypmath
# user defined
from src.optimizer import SAM

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
























class Contrastive_Loss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x / self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t() / self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j

def normalize_embeddings(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm


# 相似度矩阵
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a = normalize_embeddings(a, eps)
    b = normalize_embeddings(b, eps)
    # print("a",a.shape)
    # print("b",b.shape)
    sim_mt = torch.mm(a, b.transpose(0, 1))
    # sim_mt = torch.mm(a, b)
    return sim_mt

class WeightedContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义可学习的权重参数
        self.weight_ta = nn.Parameter(torch.tensor(0.4))
        self.weight_tv = nn.Parameter(torch.tensor(0.4))
        self.weight_av = nn.Parameter(torch.tensor(0.1))
        self.weight_t_av = nn.Parameter(torch.tensor(1.0))
        self.weight_a_av = nn.Parameter(torch.tensor(0.1))
        self.weight_v_av = nn.Parameter(torch.tensor(0.1))
        #print("ta = ",0.7)
        #print("tv = ", 0.8)

    def forward(self, loss_ta, loss_tv, loss_av, loss_t_av, loss_a_av, loss_v_av):
        # 对损失函数加权
        weighted_loss_ta = self.weight_ta * loss_ta
        weighted_loss_tv = self.weight_tv * loss_tv
        weighted_loss_av = self.weight_av * loss_av
        weighted_loss_t_av = self.weight_t_av * loss_t_av
        weighted_loss_a_av = self.weight_a_av * loss_a_av
        weighted_loss_v_av = self.weight_v_av * loss_v_av

        # 计算总的加权损失
        total_loss = weighted_loss_ta + weighted_loss_tv + weighted_loss_av + weighted_loss_t_av + weighted_loss_a_av + weighted_loss_v_av
        return total_loss

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, input_vectors, target_vectors):
        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(input_vectors, target_vectors)

        # 构建损失函数，使余弦相似度接近1
        loss = 1 - cosine_similarity

        return loss.mean()




class Alignment_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSE = nn.MSELoss()

    def forward(self, x, y):
        sim_mt_x = Sim_matrix(x, x)
        sim_mt_y = Sim_matrix(y, y)

        loss = self.MSE(sim_mt_x, sim_mt_y)
        return loss

def Sim_matrix(a, b):
    a_norm, b_norm = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

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
        self.O_enc_a = EmbeddingNet(
                    input_size=1024,
                    output_size=512,
                    dropout=0.1,
                    use_bn=True
                )
        self.W_enc_a = EmbeddingNet(
                input_size=1024,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
        self.O_enc_v = EmbeddingNet(
                    input_size=512,
                    output_size=512,
                    dropout=0.1,
                    use_bn=True
                )
        self.W_enc_v = EmbeddingNet(
                    input_size=512,
                    output_size=512,
                    dropout=0.1,
                    use_bn=True
                )
        self.O_enc = EmbeddingNet(
                input_size=1536,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )


        # if self.modality == 'audio':
        #     self.O_enc = EmbeddingNet(
        #         input_size=1024,
        #         output_size=512,
        #         dropout=0.1,
        #         use_bn=True
        #     )
        #     self.W_enc = EmbeddingNet(
        #         input_size=1024,
        #         output_size=512,
        #         dropout=0.1,
        #         use_bn=True
        #     )
        # elif self.modality == 'video':
        #     self.O_enc = EmbeddingNet(
        #         input_size=512,
        #         output_size=512,
        #         dropout=0.1,
        #         use_bn=True
        #     )
        #     self.W_enc = EmbeddingNet(
        #         input_size=512,
        #         output_size=512,
        #         dropout=0.1,
        #         use_bn=True
        #     )
        # else:
        #     self.O_enc = EmbeddingNet(
        #         input_size=1536,
        #         output_size=512,
        #         dropout=0.1,
        #         use_bn=True
        #     )
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
        self.cross_attention = Transformer(512, 6, 8, 64, 512, dropout=0.4)
        self.pos_emb1D = torch.nn.Parameter(torch.randn(3, 512))









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
        # if self.modality == 'audio':
        #     w = w[:,512:]
        #     model_input = a
        #
        # elif self.modality == 'video':
        #     w = w[:,:512]
        #     model_input = v
        # else:
        #     if self.word_embeddings == 'wavcaps':
        #         w = w[:,512:]
        #     elif self.word_embeddings == 'clip':
        #         w = w[:,:512]
        #     model_input = torch.cat((v, a), dim=1)
        ###############
        w_a = w[:,512:]
        w_v = w[:,:512]
        model_input_a = a
        model_input_v = v
        model_input = torch.cat((v, a), dim=1)
        o_a = self.O_enc_a(model_input_a)
        o_v = self.O_enc_v(model_input_v)
        o = self.O_enc(model_input)

        o_a = o_a + self.pos_emb1D[0, :]
        o_v = o_v + self.pos_emb1D[1, :]
        o = o + self.pos_emb1D[2, :]
        o_a = o_a.unsqueeze(1)
        o_v = o_v.unsqueeze(1)
        o = o.unsqueeze(1)

        o_a = self.cross_attention(o_a).squeeze(1)
        o_v = self.cross_attention(o_v).squeeze(1)
        o = self.cross_attention(o).squeeze(1)

        w_a = self.W_enc_a(w_a)
        w_v = self.W_enc_v(w_v)
        w = self.W_enc(w)

        theta_o = self.O_proj(o)
        theta_o_a = self.O_proj(o_a)
        theta_o_v = self.O_proj(o_v)
        rho_o = self.D_o(theta_o)
        rho_o_a = self.D_o(theta_o_a)
        rho_o_v = self.D_o(theta_o_v)

        theta_w = self.W_proj(w)
        theta_w_a = self.W_proj(w_a)
        theta_w_v = self.W_proj(w_v)


        rho_w=self.D_w(theta_w)
        rho_w_a=self.D_w(theta_w_a)
        rho_w_v=self.D_w(theta_w_v)







        #print("model_input", model_input.shape)

        # o = self.O_enc(model_input)
        #
        # w = self.W_enc(w)
        #
        #
        #
        # theta_o = self.O_proj(o)
        #
        #
        # rho_o = self.D_o(theta_o)
        #
        #
        # theta_w = self.W_proj(w)
        #
        #
        # rho_w=self.D_w(theta_w)


        output = {
            "theta_w": theta_w,
            "w": w,
            "w_a":w_a,
            "w_v":w_v,
            "rho_w": rho_w,
            "theta_o": theta_o,
            "rho_o": rho_o,
            "theta_o_a": theta_o_a,
            "rho_o_a": rho_o_a,
            "theta_o_v": theta_o_v,
            "rho_o_v": rho_o_v,
            "theta_w_a":theta_w_a,
            "rho_w_a":rho_w_a,
            "theta_w_v":theta_w_v,
            "rho_w_v":rho_w_v,
            "o_a":o_a,
            "o_v":o_v,

        }


        return output


    def compute_loss(self, outputs, embeddings_crossentropy, gt_cross_entropy):

        o_a= outputs['o_a']
        o_v= outputs['o_v']

        theta_w = outputs['theta_w']
        theta_w_a=outputs['theta_w_a']
        theta_w_v=outputs['theta_w_v']

        w = outputs['w']
        w_a= outputs['w_a']
        w_v= outputs['w_v']

        rho_w = outputs['rho_w']
        rho_w_a=outputs['rho_w_a']
        rho_w_v=outputs['rho_w_v']

        theta_o_a= outputs['theta_o_a']
        theta_o_v= outputs['theta_o_v']
        theta_o = outputs['theta_o']

        rho_o = outputs['rho_o']
        rho_o_a = outputs['rho_o_a']
        rho_o_v = outputs['rho_o_v']



        device = theta_w.device


        if self.cross_entropy_loss==True:
            a_h = hypmath.logmap0(hypmath.project(o_a, c=0.2), c=0.2)
            v_h = hypmath.logmap0(hypmath.project(o_v, c=0.2), c=0.2)
            align_loss = Alignment_loss()
            l_align = align_loss(v_h, a_h)
            loss_align = l_align * 1e-3
            # if self.modality == 'audio':
            #     embeddings_crossentropy = embeddings_crossentropy[:,512:]
            # elif self.modality == 'video':
            #     embeddings_crossentropy = embeddings_crossentropy[:,:512]
            # else:
            #     if self.word_embeddings == 'wavcaps':
            #         embeddings_crossentropy = embeddings_crossentropy[:,512:]
            #     elif self.word_embeddings == 'clip':
            #         embeddings_crossentropy = embeddings_crossentropy[:,:512]
            #
            # embedding_cross_entropy=self.W_proj(self.W_enc(embeddings_crossentropy))
            # Cross_loss=nn.CrossEntropyLoss()
            # scores=torch.matmul(theta_o, embedding_cross_entropy.t()) # (bs, 64) x (K_seen, 64).T = (bs, 64) x (64, K_seen) = (bs, K_seen)
            # # gt_cross_entropy = [1, 3, 2, 55, 97, 45, ...] list of gt class labels -> shape (bs,)
            #l_ce = Cross_loss(scores, gt_cross_entropy)
            loss_ta = Contrastive_Loss()(sim_matrix(theta_o_a, theta_w_a))
            loss_tv = Contrastive_Loss()(sim_matrix(theta_o_v, theta_w_v))
            loss_av = Contrastive_Loss()(sim_matrix(theta_o_a, theta_o_v))
            loss_t_av = Contrastive_Loss()(sim_matrix(theta_o, theta_w))
            loss_a_av = Contrastive_Loss()(sim_matrix(theta_o_a, theta_o))
            loss_v_av = Contrastive_Loss()(sim_matrix(theta_o_v, theta_o))
            weighted_loss_fn = WeightedContrastiveLoss()
            l_ce2 = weighted_loss_fn(loss_ta, loss_tv, loss_av, loss_t_av, loss_a_av, loss_v_av)
            cosine_loss = CosineSimilarityLoss()
            l_cos = cosine_loss(theta_o, theta_w) + cosine_loss(theta_o_a, theta_w_a) + cosine_loss(theta_o_v,
                                                                                                  theta_w_v) + cosine_loss(
                theta_o_a, theta_o_v)
            l_ce = l_ce2 + l_align*0.4 #+ l_cos*0.01ws
            #l_ce = loss_ta + loss_tv + loss_av + loss_t_av + loss_a_av + loss_v_av



        else:
            l_ce = torch.tensor(0., device=device)

        if self.reg_loss==True:
            l_reg = (
                self.MSE_loss(theta_o, theta_w) + self.MSE_loss(theta_o_a, theta_w_a) + self.MSE_loss(theta_o_v, theta_w_v)
            )
        else:
            l_reg = torch.tensor(0., device=device)


        if self.rec_loss == True:
            l_rec = (
                    self.MSE_loss(w, rho_o) +
                    self.MSE_loss(w, rho_w) +
                    self.MSE_loss(w_a, rho_o_a) + self.MSE_loss(w_v, rho_o_v) +
                    self.MSE_loss(w_a, rho_w_a) + self.MSE_loss(w_v, rho_w_v)
            )
        else:
            l_rec = torch.tensor(0., device=device)


        loss_total = l_rec+l_reg+l_ce
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

        w_a = w[:, 512:]
        w_v = w[:, :512]
        model_input_a = a
        model_input_v = v
        model_input = torch.cat((v, a), dim=1)
        o_a = self.O_enc_a(model_input_a)
        o_v = self.O_enc_v(model_input_v)
        o = self.O_enc(model_input)

        o_a = o_a + self.pos_emb1D[0, :]
        o_v = o_v + self.pos_emb1D[1, :]
        o = o + self.pos_emb1D[2, :]
        o_a = o_a.unsqueeze(1)
        o_v = o_v.unsqueeze(1)
        o = o.unsqueeze(1)

        o_a = self.cross_attention(o_a).squeeze(1)
        o_v = self.cross_attention(o_v).squeeze(1)
        o = self.cross_attention(o).squeeze(1)

        w_a = self.W_enc_a(w_a)
        w_v = self.W_enc_v(w_v)
        w = self.W_enc(w)

        theta_o = self.O_proj(o)
        theta_o_a = self.O_proj(o_a)
        theta_o_v = self.O_proj(o_v)
        rho_o = self.D_o(theta_o)
        rho_o_a = self.D_o(theta_o_a)
        rho_o_v = self.D_o(theta_o_v)

        theta_w = self.W_proj(w)
        theta_w_a = self.W_proj(w_a)
        theta_w_v = self.W_proj(w_v)

        rho_w = self.D_w(theta_w)
        rho_w_a = self.D_w(theta_w_a)
        rho_w_v = self.D_w(theta_w_v)

        return theta_o, theta_o, theta_w