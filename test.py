import argparse

import numpy as np
import torch
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:1", help='device name')
parser.add_argument('--data', type=str, default='./data/PEMS-D360', help='dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument("--seq_in", type=int, default=12, help='historical length')
parser.add_argument("--seq_out", type=int, default=12, help='prediction length')
parser.add_argument("--scale", type=float, default=0.01, help='scaler of T')
parser.add_argument("--interval", type=int, default=5, help="interval of ODE")
args = parser.parse_args()

if args.data == "METR-LA":
    args.data_file = "./data/METR-LA"
    args.adj_data = "./data/sensor_graph/adj_mx.pkl"
    args.num_node = 207
    args.in_dim = 2
    args.task = "speed"

elif args.data == "PEMS-BAY":
    args.data_file = "./data/PEMS-BAY"
    args.adj_data = "./data/sensor_graph/adj_mx_bay.pkl"
    args.num_node = 325
    args.in_dim = 2
    args.task = "speed"

elif args.data == "PEMS-D360":
    args.data_file = "./data/PEMS-D360"
    args.adj_data = "./data/sensor_graph/pems03.csv"
    args.num_node = 358
    args.in_dim = 1
    args.task = "flow"

elif args.data == "PEMS-D4":
    args.data_file = "./data/PEMS-D4"
    args.adj_data = "./data/sensor_graph/distance_pemsd4.csv"
    args.num_node = 307
    args.in_dim = 1
    args.task = "flow"

elif args.data == "PEMS-D460":
    args.data_file = "./data/PEMS-D4-60"
    args.adj_data = "./data/sensor_graph/distance_pemsd4.csv"
    args.num_node = 307
    args.in_dim = 1
    args.task = "flow"

elif args.data == "PEMS-D8":
    args.data_file = "./data/PEMS-D8"
    args.adj_data = "./data/sensor_graph/distance_pemsd8.csv"
    args.num_node = 170
    args.in_dim = 1
    args.task = "flow"

elif args.data == "PEMS-D860":
    args.data_file = "./data/PEMS-D860"
    args.adj_data = "./data/sensor_graph/distance_pemsd8.csv"
    args.num_node = 170
    args.in_dim = 1
    args.task = "flow"

dataloader = utils.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
args.scaler = dataloader['scaler']
args.device = torch.device(args.device)


def test():
    # T = torch.linspace(0., args.seq_out, args.interval * args.seq_out + 1) * args.scale
    # id_train = list(np.arange(args.interval, args.interval * (args.seq_out + 1), args.interval))
    model = torch.load("./PEMS-D360.pkl", map_location=args.device)
    # model.decoderr.T = T
    # model.decoder.id_train = id_train
    # model.decoder.args.seq_out = args.seq_out
    # model.encoder.args.seq_in = args.seq_in
    model.eval()
    test_mae = []
    test_mape = []
    test_rmse = []

    x_idx = list(np.arange(0, 60, 5))
    y_idx = list(np.arange(0, 60, 5))

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        x = x[:, x_idx, :, :]
        testX = torch.Tensor(x).to(args.device)
        testy = torch.Tensor(y).to(args.device)

        with torch.no_grad():
            continous_outputs, decrete_outputs = model(testX[:, -args.seq_in:, :, :])
        testy = testy[:, :, :, 0:1]
        prediction = args.scaler.inv_transform(continous_outputs)
        # with open("./pred.pkl", "wb") as f:
        #     pickle.dump(prediction, f)
        # with open("./lebels.pkl", "wb") as f:
        #     pickle.dump(testy, f)
        mae = utils.masked_mae(prediction, testy, 0.0).item()
        rmse = utils.masked_rmse(prediction, testy, 0.0).item()
        mape = utils.masked_mape(prediction, testy, 0.0).item()
        log = 'Test Loss: {:.4f},  Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(mae, mape, rmse))
        test_mae.append(mae)
        test_mape.append(mape)
        test_rmse.append(rmse)
    mae = np.mean(test_mae)
    mape = np.mean(test_mape)
    rmse = np.mean(test_rmse)
    log = 'Test Loss: {:.4f},  Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(mae, mape, rmse))


if __name__ == '__main__':
    test()
