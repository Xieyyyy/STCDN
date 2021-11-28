import pickle

import numpy as np
import torch
# import util
import utils
# from model import *
# import ASTGCN_r
from scipy.interpolate import interp1d

# param_dict = torch.load("./garage/pemsd4_epoch_32_105.3.pth")

adj_mx, distance_mx = utils.get_adjacency_matrix("../../GraphWaveNet/Graph-WaveNet/data/PEMS-D860/distance_pemsd8.csv",
                                                 170)
dataloader = utils.load_dataset("../../GraphWaveNet/Graph-WaveNet/data/PEMS-D860/", 32, 32, 32)
scaler = dataloader['scaler']

# model = ASTGCN_r.make_model("cuda:1", 2, 1, 3, 64, 64, 1, adj_mx,
#                             12, 12, 170)
# model = gwnet("cuda:1", 307, 0.3, supports=[torch.tensor(i).to("cuda:1") for i in adj_mx], gcn_bool=False,
#               addaptadj=False, aptinit=torch.zeros(307).to("cuda:1"), in_dim=1, out_dim=12, residual_channels=32,
#               dilation_channels=32, skip_channels=32 * 8, end_channels=32 * 16)
# model.load_state_dict(param_dict)

model = torch.load("./PEMS-D860.pkl", map_location="cuda:2")

model.eval()
test_mae = []
test_mape = []
test_rmse = []
x_idx = list(np.arange(0, 12 * 5, 5))
y_idx = list(np.arange(0, 12 * 5, 5))


def lagrangeInterp(x, y, xi):
    '''

    :param x: [0,1,2,3,4,5,6,7,8,9,10,11]
    :param y: values
    :param xi: point
    :return:
    '''
    n = len(x)  # T
    L = np.zeros(n)  # T

    for k in range(0, n):

        L[k] = 1

        for p in range(0, n):

            if p != k:
                L[k] = L[k] * (xi - x[p]) / (x[k] - x[p])

    yi = sum(y * L)

    return yi


def lagrangeInterptensor(x, y, xi):
    '''

    :param x: T
    :param y: [b,n]
    :param xi: point
    :return:
    '''
    n = len(x)  # T
    L = torch.zeros(y.shape).to("cuda:2")  # [0] * 12

    for k in range(0, n):

        L[k] = 1

        for p in range(0, n):

            if p != k:
                L[k] = L[k] * (xi - x[p]) / (x[k] - x[p])

    yi = sum(y * L)

    return yi


def interpolationtensor(y):
    X = np.arange(0, 12, 1)
    Xi = np.arange(0, 12, 0.2)
    y = y.squeeze().transpose(-1, 0)  # [b,T,N]
    res = [lagrangeInterptensor(X, y, xi) for xi in Xi]
    res = torch.stack(res)  # [60,32,307]
    return res.permute(2, 0, 1)


def otherinterpolation(y):
    y = y.transpose(-1, -2).cpu().numpy()
    x = np.linspace(0, 11, num=12, endpoint=True)
    f1 = interp1d(x, y, kind='nearest')
    f2 = interp1d(x, y, kind='linear')
    f3 = interp1d(x, y, kind='quadratic')
    f4 = interp1d(x, y, kind='cubic')
    xnew = np.linspace(0, 11, num=60, endpoint=True)
    return torch.Tensor(f1(xnew)).to("cuda:2"), \
           torch.Tensor(f2(xnew)).to("cuda:2"), \
           torch.Tensor(f3(xnew)).to("cuda:2"), \
           torch.Tensor(f4(xnew)).to("cuda:2")


def main():
    test_mae = []
    test_mape = []
    test_rmse = []
    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testX = torch.Tensor(x).to("cuda:2")[:, x_idx, :, :]
        # testX = testX.transpose(1, 3)
        testy = torch.Tensor(y).to("cuda:2")
        # testX = nn.functional.pad(testX, (1, 0, 0, 0))
        with torch.no_grad():
            outputs = model(testX)  # [32,1,307,13]
        prediction = scaler.inv_transform(outputs[0])
        # with open("./truth.pkl", "wb") as f:
        #     pickle.dump(testy, f)
        # with open("./cstn.pkl", "wb") as f:
        #     pickle.dump(prediction, f)
        # results = interpolationtensor(prediction.transpose(1, 2))  # [B,N,T]
        # results = results.unsqueeze(-1)
        # nearest, linear, quadratic, cubic = otherinterpolation(prediction)  # [B,N,T]
        # nearest, linear, quadratic, cubic = nearest.transpose(1, 2).unsqueeze(-1), linear.transpose(1, 2).unsqueeze(
        #     -1), quadratic.transpose(1, 2).unsqueeze(-1), cubic.transpose(1, 2).unsqueeze(-1)

        # with open("./label.pkl", "wb") as f:
        #     pickle.dump(testy, f)

        mae = utils.masked_mae(prediction, testy, 0.0).item()
        rmse = utils.masked_rmse(prediction, testy, 0.0).item()
        mape = utils.masked_mape(prediction, testy, 0.0).item()
        test_mae.append(mae)
        test_rmse.append(rmse)
        test_mape.append(mape)
        print("Iter:" + str(iter) + " mae:" + str(np.mean(np.asarray(test_mae))) + " rmse:" + str(np.mean(
            np.asarray(test_rmse))) + " mape:" + str(np.mean(np.asarray(test_mape))))

    print(" mae:" + str(np.mean(np.asarray(test_mae))) + " rmse:" + str(np.mean(
        np.asarray(test_rmse))) + " mape:" + str(np.mean(np.asarray(test_mape))))


if __name__ == '__main__':
    main()
