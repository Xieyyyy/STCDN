import numpy as np
import torch
import torch.optim as op

import utils
from model import Model


class Holder():
    def __init__(self, args):
        self.args = args
        self.model = Model(args).to(self.args.device)
        self.optimizer = op.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.lr_sch = op.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.995, verbose=True)
        self.loss = utils.masked_mae
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.extraction = list(
            np.arange(0, 360,
                      6))
        print(len(self.extraction))
        print('Total:', total_num, 'Trainable:', trainable_num)

    def train(self, inputs, reals):
        self.model.train()
        self.optimizer.zero_grad()
        if self.args.decoder_interval == None:
            decrete_outputs = self.model(inputs)
        else:
            continous_outputs, decrete_outputs = self.model(inputs)
            # print(continous_outputs.shape)
        reals = reals[:, :self.args.seq_out, :, :]
        prediction = self.args.scaler.inv_transform(decrete_outputs)
        loss = self.loss(prediction, reals, 0.0)
        loss.backward(retain_graph=True)
        if self.args.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()
        mape = utils.masked_mape(prediction, reals, 0.0).item()
        rmse = utils.masked_rmse(prediction, reals, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, inputs, reals):
        out_idx = list(np.arange(0, 48, 2))
        self.model.eval()
        with torch.no_grad():
            if self.args.decoder_interval == None:
                decrete_outputs = self.model(inputs)
            else:
                continous_outputs, decrete_outputs = self.model(inputs)
                continous_outputs = continous_outputs[:, out_idx, :, :]
        prediction = self.args.scaler.inv_transform(continous_outputs)
        if self.args.task == "speed":
            maes = [self.loss(prediction[:, 2, :, :], reals[:, 2, :, :], 0.0).item(),
                    self.loss(prediction[:, 5, :, :], reals[:, 5, :, :], 0.0).item(),
                    self.loss(prediction[:, -1, :, :], reals[:, -1, :, :], 0.0).item(),
                    self.loss(prediction, reals, 0.0).item()]
            mapes = [utils.masked_mape(prediction[:, 2, :, :], reals[:, 2, :, :], 0.0).item(),
                     utils.masked_mape(prediction[:, 5, :, :], reals[:, 5, :, :], 0.0).item(),
                     utils.masked_mape(prediction[:, -1, :, :], reals[:, -1, :, :], 0.0).item(),
                     utils.masked_mape(prediction, reals, 0.0).item()]
            rmses = [utils.masked_rmse(prediction[:, 2, :, :], reals[:, 2, :, :], 0.0).item(),
                     utils.masked_rmse(prediction[:, 5, :, :], reals[:, 5, :, :], 0.0).item(),
                     utils.masked_rmse(prediction[:, -1, :, :], reals[:, -1, :, :], 0.0).item(),
                     utils.masked_rmse(prediction, reals, 0.0).item()]
            return maes, mapes, rmses
        else:
            mae = self.loss(prediction, reals, 0.0).item()
            mape = utils.masked_mape(prediction, reals, 0.0).item()
            rmse = utils.masked_rmse(prediction, reals, 0.0).item()
            return mae, mape, rmse
