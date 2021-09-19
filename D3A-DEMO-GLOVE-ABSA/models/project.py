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

        # if self.opt.lm != 'none':
        #     self.conv_op = DECNN_CONV(300*2, self.opt)
        #     self.linear_gatecat = torch.nn.Linear(300*2, 300*2)
        # else:
        #     self.conv_op = DECNN_CONV(300, self.opt)
        #     self.linear_gatecat = torch.nn.Linear(300, 300)

        self.linear_fuse = torch.nn.Linear(600, 300)
        self.linear_gate = torch.nn.Linear(600, 300)
        self.conv_op = DECNN_CONV(300, self.opt)


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
            fuse_emb = self.linear_fuse(concat_emb) # 300
            fuse_gat = torch.sigmoid(self.linear_gate(concat_emb)) # 300
            # fuse_gat = torch.sigmoid(self.linear_gate(fuse_emb))
            # x_emb = fuse_emb * fuse_gat

            x_emb_grad = (fuse_emb * fuse_gat).unsqueeze(-2) # b, len, 1, 600
            x_emb_nograd = x_emb_grad.detach() # b, len, 1, 600

            x_emb_candidate = torch.cat([x_emb_grad, x_emb_nograd], -2)# b, len, 2, 600
            select_signal = data_type.reshape(x.shape[0], 1, 1, 2).repeat(1, x.shape[1], 1, 1) # b, len, 1, 2

            x_emb = torch.matmul(select_signal, x_emb_candidate).squeeze(-2)
        else:
            x, mask, lmwords, lmprobs, data_type = inputs
            x_emb = self.global_emb(x)

        x_emb_tran = self.dropout(x_emb).transpose(1, 2)
        x_conv = self.conv_op(x_emb_tran)

        x_logit = self.linear_ae256(x_conv)
        return F.softmax(x_logit, -1)




        # a = np.random.rand(1, 100)
        # print(a)
        # a[a <= 0.15] = 0
        # a[a >= 0.15] = 1


        # x, mask, position, proto_emb, pos = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        # x_emb = torch.cat((self.global_emb(x), self.domain_emb(x)), dim=2)
        'affine'
        # scaled_part = self.linear_scale(x_emb) * proto_emb
        # shift_part = self.linear_shift(x_emb)
        #
        # affined_emb = scaled_part + shift_part
        # affined_emb_tran = self.dropout(affined_emb).transpose(1, 2)
        # affined_conv = self.conv_op(affined_emb_tran)
        # affined_logit = self.linear_ae256(affined_conv)
        # return F.softmax(affined_logit, -1)
        'gate+cat'
        # scale_value = torch.sigmoid(self.linear_scale(x_emb))
        # scaled_proto = proto_emb * scale_value # b,80,300
        #
        # fuse_emb = torch.tanh(self.linear_shift(torch.cat([x_emb, scaled_proto], -1)))
        # fuse_emb_tran = self.dropout(fuse_emb).transpose(1, 2)
        # fuse_conv = self.conv_op(fuse_emb_tran)
        # fuse_logit = self.linear_ae256(fuse_conv)
        # return F.softmax(fuse_logit, -1)
        'mutual gating'
        # proto_gate = torch.sigmoid(self.linear_shift(proto_emb))
        # x_gate = torch.sigmoid(self.linear_scale(x_emb))
        #
        # proto_emb_gated = proto_emb * x_gate
        # x_emb_gated= x_emb * proto_gate
        #
        # x_emb_tran = self.dropout(x_emb_gated).transpose(1, 2)
        # x_conv = self.conv_op(x_emb_tran)
        #
        # proto_emb_tran = self.dropout(proto_emb_gated).transpose(1,2)
        # proto_conv = self.conv_op(proto_emb_tran)
        #
        # fuse_logic = self.linear_ae(torch.cat([x_conv, proto_conv], -1))
        #
        # return  F.softmax(fuse_logic, -1)
        'similarity'
        # perserve_gate = torch.sigmoid(torch.sum(x_emb * proto_emb, dim=-1, keepdim=True))
        #
        # fuse_emb = perserve_gate * proto_emb + (1-perserve_gate) * x_emb
        # fuse_emb_tran = self.dropout(fuse_emb).transpose(1, 2)
        # fuse_conv = self.conv_op(fuse_emb_tran)
        # fuse_logit = self.linear_ae256(fuse_conv)
        # return F.softmax(fuse_logit, -1)
        'all proto'
        # fuse_emb = proto_emb
        # fuse_emb_tran = self.dropout(fuse_emb).transpose(1, 2)
        # fuse_conv = self.conv_op(fuse_emb_tran)
        # fuse_logit = self.linear_ae256(fuse_conv)
        # return F.softmax(fuse_logit, -1)
        'only emb'
        # fuse_logit = self.linear_ae300(x_emb)
        # return F.softmax(fuse_logit, -1)

        'concat'
        # fuse_emb = torch.tanh(self.linear_shift(torch.cat([x_emb, proto_emb], -1)))
        # fuse_emb_tran = self.dropout(fuse_emb).transpose(1, 2)
        # fuse_conv = self.conv_op(fuse_emb_tran)
        # fuse_logit = self.linear_ae256(fuse_conv)
        # return F.softmax(fuse_logit, -1)

        'rerank'
        # x_emb_broad = torch.unsqueeze(x_emb, 2) # b, 80, 1, 300
        #
        # fw_word = torch.cat((self.global_emb(fw_lmwords), self.domain_emb(fw_lmwords)), dim=-1) # b, 70, 3, 300
        # # fw_score = torch.matmul(self.linear_proj(x_emb_broad), fw_word.transpose(2,3)) # b, 80, 1, 3
        # fw_score = torch.matmul(x_emb_broad, fw_word.transpose(2,3)) # b, 80, 1, 3
        # fw_prob = torch.softmax(fw_score, -1)
        # fw_emb = torch.matmul(fw_prob, fw_word).squeeze()
        # # fw_scale = fw_weight.view(fw_weight.shape[0], fw_weight.shape[1], 1).repeat(1, 1, 300)
        #
        # bw_word = torch.cat((self.global_emb(bw_lmwords), self.domain_emb(bw_lmwords)), dim=-1) # b, 70, 3, 300
        # # bw_score = torch.matmul(self.linear_proj(x_emb_broad), bw_word.transpose(2,3)) # b, 80, 1, 3
        # bw_score = torch.matmul(x_emb_broad, bw_word.transpose(2,3)) # b, 80, 1, 3
        # bw_prob = torch.softmax(bw_score, -1)
        # bw_emb = torch.matmul(bw_prob, bw_word).squeeze()
        # # bw_scale = bw_weight.view(bw_weight.shape[0], bw_weight.shape[1], 1).repeat(1, 1, 300)
        #
        # # lm = fw_emb * fw_scale + bw_emb * bw_scale
        # lm_emb = (fw_emb + bw_emb)/2.
        #
        # # x_emb = (1 - self.opt.interpolation) * x_emb + self.opt.interpolation * lm_emb
        # x_emb = torch.cat([x_emb, lm_emb], -1)





    def softmask(self, score, mask):
        mask_3dim = mask.view(mask.shape[0], 1, -1).repeat(1, self.opt.max_sentence_len, 1)
        score_exp = torch.mul(torch.exp(score), mask_3dim)
        sumx = torch.sum(score_exp, dim=-1, keepdim=True)
        return score_exp / (sumx + 1e-5)

