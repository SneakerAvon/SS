import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from scipy.fftpack import dct, idct
import torch.optim as optim
from model import *
import util


def filter_negative(input_, thres):
    times = input_[:, 0, 0, 1]

    m = []
    cnt = 0
    c = thres / 288
    for t in times:
        if t < c:
            st = times < 0
            gt = torch.logical_and(times <= (1 + t - c), times >= (t + c))
        elif t > (1 - c):
            st = torch.logical_and(times <= (t - c), times >= (c + t - 1))
            gt = times > 1
        else:
            st = times <= (t - c)
            gt = times >= (t + c)

        res = torch.logical_or(st, gt).view(1, -1)
        res[0, cnt] = True
        cnt += 1
        m.append(res)
    m = torch.cat(m)
    return m

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit,
                 adj_m, method, fn_t, em_t, im_t, ts_t, ism_t, ism_e, tempe, lam):
        self.device = device
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

        self.nor_adj = np.array(adj_m)[0,]
        self.method = method
        self.fn_t = fn_t
        self.em_t = em_t
        self.im_t = im_t
        self.ts_t = ts_t
        self.ism_t = ism_t
        self.ism_e = ism_e
        self.tempe = tempe
        self.lam = lam

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input__ = input.transpose(1, 3)
        bs = input__.shape[0]
        frame = input__.shape[1]
        num_node = input__.shape[2]

        input = nn.functional.pad(input,(1,0,0,0))

        output, rep = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]

        real_val = torch.unsqueeze(real_val,dim=1)
        real = real_val.transpose(1, 3)
        predict = self.scaler.inverse_transform(output)

        s_loss = self.loss(predict, real_val, 0.0)
        #loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        #self.optimizer.step()
        mape = util.masked_mape(predict,real_val,0.0).item()
        rmse = util.masked_rmse(predict,real_val,0.0).item()

        if self.method == 'pure':
            s_loss.backward()
            self.optimizer.step()
            return s_loss.item(), mape, rmse

        elif self.method == 'graph':
            diff = 0
            if self.im_t or self.ts_t or self.ism_t:
                input_ = input__.detach().clone()

                if self.im_t:
                    rand = torch.rand(bs, frame, num_node).to(self.device)
                    input_[:, :, :, 0] = input_[:, :, :, 0] * (rand >= self.im_t)

                if self.ts_t:
                    r = self.scaler.transform(real[:, :, :, 0])
                    s = torch.cat((input_[:, :, :, 0], r), dim=1)[:, :frame + 1, :]
                    rand = (1 - self.ts_t) * torch.rand(bs, 1, 1) + self.ts_t
                    rand = rand.expand(bs, frame + 1, num_node).to(self.device)
                    input_[:, :, :, 0] = (s * rand + torch.roll(s, -1, 1) * (1 - rand))[:, :frame, :]

                if self.ism_t:
                    r = self.scaler.transform(real[:, :, :, 0])
                    s = torch.cat((input_[:, :, :, 0], r), dim=1).cpu()
                    o = []
                    for i in range(bs):
                        t = np.transpose(np.array(s[i]))
                        m1 = np.ones((num_node, self.ism_e))
                        m2 = np.random.uniform(low=self.ism_t, high=1.0, size=(num_node, t.shape[1] - self.ism_e))
                        m1 = np.matmul(self.nor_adj, m1)
                        m2 = np.matmul(self.nor_adj, m2)
                        mall = np.concatenate((m1, m2), axis=1)
                        t = dct(t, norm='ortho')
                        t = np.multiply(t, mall)
                        t = idct(t, norm='ortho')
                        o.append(np.transpose(t))
                    o = np.stack(o)
                    input_[:, :, :, 0] = torch.tensor(o[:, :frame, :]).to(self.device)

                #diff = torch.mean(torch.abs(input_[:, :, :, 0] - input[:, :, :, 0])).item()
                input_ = input_.transpose(1, 3)
                input_ = nn.functional.pad(input_, (1, 0, 0, 0))

                if not self.em_t:
                    _, aug_rep = self.model(input_)
                else:
                    _, aug_rep = self.model(input_, self.em_t)

            else:
                _, aug_rep = self.model(input)

        norm1 = rep.norm(dim=1)
        norm2 = aug_rep.norm(dim=1)
        sim_matrix = torch.mm(rep, torch.transpose(aug_rep, 0, 1)) / torch.mm(norm1.view(-1, 1), norm2.view(1, -1))
        sim_matrix = torch.exp(sim_matrix / self.tempe)

        diag = bs
        pos_sim = sim_matrix[range(diag), range(diag)]

        avg_neg = diag - 1
        if self.fn_t:
            m = filter_negative(input, self.fn_t)
            s = torch.sum(m, dim=1) - 1
            avg_neg = torch.mean(s * 1.0).cpu().item()
            sim_matrix = sim_matrix * m

        max_id = torch.argmax(sim_matrix, dim=1)
        labels = torch.arange(diag).to(self.device)
        corr_num = torch.sum(max_id == labels).item()
        avg_acc = corr_num / diag

        u_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        u_loss = torch.mean(-torch.log(u_loss))

        loss = s_loss + self.lam * u_loss
        loss.backward()
        self.optimizer.step()

        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output, _ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real_val = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real_val, 0.0)
        mape = util.masked_mape(predict,real_val,0.0).item()
        rmse = util.masked_rmse(predict,real_val,0.0).item()
        return loss.item(),mape,rmse
