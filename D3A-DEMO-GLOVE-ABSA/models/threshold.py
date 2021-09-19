from layers.decnn_conv import DECNN_CONV
import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import *


class DECNN(nn.Module):
    def __init__(self, global_emb, opt):
        super(DECNN, self).__init__()
        self.opt = opt
        global_emb = torch.tensor(global_emb, dtype=torch.float32).to(self.opt.device)

        self.global_emb = nn.Embedding.from_pretrained(global_emb)

        if self.opt.lm != 'none':
            self.conv_op = DECNN_CONV(300, self.opt)
            self.gate1 = torch.nn.Linear(300*2, 1)
            self.gate2 = torch.nn.Linear(300*2, 1)
        else:
            self.conv_op = DECNN_CONV(300, self.opt)
            self.linear_gatecat = torch.nn.Linear(300, 300)

        self.linear_ae256 = torch.nn.Linear(256, self.opt.class_num)
        self.dropout = torch.nn.Dropout(self.opt.keep_prob)

    def forward(self, inputs, epoch, tau_now, is_training=False, train_y=None):
        self.epoch = epoch
        self.tau_now = tau_now

        if self.opt.lm != 'none':
            x, mask, lmwords, lmprobs, data_type = inputs
            x_emb = self.global_emb(x)

            lm_word = self.global_emb(lmwords)
            # lm_prob = torch.softmax(lmprobs, dim=-1).view(lmprobs.shape[0], lmprobs.shape[1], 1, -1) # RR prototype

            lm_prob = lmprobs.view(lmprobs.shape[0], lmprobs.shape[1], 1, -1) # LM prototype
            lm_emb = torch.matmul(lm_prob, lm_word).squeeze(-2)

            concat_emb = torch.cat([x_emb, lm_emb], -1)

            TH= 0.5
            concat_gate1 = torch.clamp(self.gate1(concat_emb), TH, 1.0)
            concat_gate2 = torch.clamp(self.gate2(concat_emb), 0.0, TH)
            # x_emb = concat_emb * concat_gate

            x_emb_grad = (concat_gate1 * x_emb + concat_gate2 * lm_emb).unsqueeze(-2) # b, len, 1, 600
            x_emb_nograd = x_emb_grad.detach() # b, len, 1, 600

            x_emb_candidate = torch.cat([x_emb_grad, x_emb_nograd], -2)# b, len, 2, 600
            select_signal = data_type.reshape(x.shape[0], 1, 1, 2).repeat(1, x.shape[1], 1, 1) # b, len, 1, 2

            x_emb = torch.matmul(select_signal, x_emb_candidate).squeeze(-2)

            pass






        else:
            x, mask, lmwords, lmprobs, data_type = inputs
            x_emb = self.global_emb(x)

        x_emb_tran = self.dropout(x_emb).transpose(1, 2)
        x_conv = self.conv_op(x_emb_tran)

        x_logit = self.linear_ae256(x_conv)
        return F.softmax(x_logit, -1)




    def softmask(self, score, mask):
        mask_3dim = mask.view(mask.shape[0], 1, -1).repeat(1, self.opt.max_sentence_len, 1)
        score_exp = torch.mul(torch.exp(score), mask_3dim)
        sumx = torch.sum(score_exp, dim=-1, keepdim=True)
        return score_exp / (sumx + 1e-5)

