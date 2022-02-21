import argparse
import time

import numpy as np
import torch
import utils
from holder import Holder
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()

# ---for training----
parser.add_argument("--device", type=str, default="cuda:4")
parser.add_argument('--data', type=str, default='PEMS-D8', help='dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='training epoch')
parser.add_argument("--seed", type=int, default=42, help='random seed')
parser.add_argument("--clip", type=float, default=5., help='gradient clip')
parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
parser.add_argument("--dropout", type=float, default=0.2, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.000001, help='weight decay rate')
parser.add_argument("--comment", type=str, default="D8_no_backlook_ratio0.01",
                    help='whether recording')
parser.add_argument("--recording", type=bool, default=True, help='whether recording')

# python main.py --device cuda:3 --data PEMS-D8 --comment PEMS-D8_multi_input2 --recording True

# ---for model----
parser.add_argument("--num_heads", type=int, default=8, help='heads (GAT)')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--out_dim', type=int, default=1, help='output dimension')
parser.add_argument("--seq_in", type=int, default=12, help='historical length')
parser.add_argument("--seq_out", type=int, default=12, help='prediction length')
parser.add_argument("--graph", type=str, default="adap", help='the type of graph')
parser.add_argument("--retain_ratio", type=float, default=0.01, help="the ratio of retaining edges")

# ---for encoder----
parser.add_argument("--encoder_interval", type=int, default=3, help="interval of ODE")
parser.add_argument("--encoder_integrate_mathod", type=str, default="euler", help='method of ode')
parser.add_argument("--encoder_rtol", type=float, default=.01, help='')
parser.add_argument("--encoder_atol", type=float, default=.001, help='')
parser.add_argument("--encoder_adjoint", type=bool, default=False, help='')
parser.add_argument("--encoder_scale", type=float, default=0.01, help='scaler of T')

# ---for decoder----
parser.add_argument("--decoder_integrate_mathod", type=str, default="euler", help='method of ode')
parser.add_argument("--decoder_interval", type=int, default=3, help="interval of ODE")
parser.add_argument("--decoder_rtol", type=float, default=.01, help='')
parser.add_argument("--decoder_atol", type=float, default=.001, help='')
parser.add_argument("--decoder_adjoint", type=bool, default=False, help='')
parser.add_argument("--decoder_scale", type=float, default=0.01, help='scaler of T')
# parser.add_argument("--back_look", type=int, default=3, help='back look')

args = parser.parse_args()

if args.data == "METR-LA":
    args.data_file = "./data/METR-LA"
    args.adj_data = "./data/sensor_graph/adj_mx.pkl"
    args.num_node = 207
    args.in_dim = 1
    args.task = "speed"

elif args.data == "PEMS-BAY":
    args.data_file = "./data/PEMS-BAY"
    args.adj_data = "./data/sensor_graph/adj_mx_bay.pkl"
    args.num_node = 325
    args.in_dim = 1
    args.task = "speed"


elif args.data == "PEMS-D3":
    args.data_file = "./data/PEMS-D3"
    args.adj_data = "./data/sensor_graph/pems03.csv"
    args.num_node = 358
    args.in_dim = 1
    args.task = "flow"

elif args.data == "PEMS-D360":
    args.data_file = "./data/PEMS-D360"
    args.adj_data = "./data/sensor_graph/pems03.csv"
    args.num_node = 358
    args.in_dim = 1
    args.task = "flow"
    x_idx = list(np.arange(0, 60, 5))
    y_idx = list(np.arange(0, 60, 5))



elif args.data == "PEMS-D4":
    args.data_file = "./data/PEMS-D4"
    args.adj_data = "./data/sensor_graph/distance_pemsd4.csv"
    args.num_node = 307
    args.in_dim = 1
    args.task = "flow"

elif args.data == "PEMS-D460":
    args.data_file = "./data/PEMS-D460"
    args.adj_data = "./data/sensor_graph/distance_pemsd4.csv"
    args.num_node = 307
    args.in_dim = 1
    args.task = "flow"
    x_idx = list(np.arange(0, 60, 5))
    y_idx = list(np.arange(0, 60, 5))

elif args.data == "PEMS-D7":
    args.data_file = "./data/PEMS-D7"
    args.adj_data = "./data/sensor_graph/PEMS07.csv"
    args.num_node = 883
    args.in_dim = 1
    args.task = "flow"

elif args.data == "PEMS-D760":
    args.data_file = "./data/PEMS-D760"
    args.adj_data = "./data/sensor_graph/PEMS07.csv"
    args.num_node = 883
    args.in_dim = 1
    args.task = "flow"
    x_idx = list(np.arange(0, 60, 5))
    y_idx = list(np.arange(0, 60, 5))

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
    x_idx = list(np.arange(0, 60, 5))
    y_idx = list(np.arange(0, 60, 5))



if args.recording:
    utils.record_info(str(args), "./records/" + args.comment)
    utils.record_info("D7, 无backlook,ratio降低为0.01",
                      "./records/" + args.comment)
    sw = SummaryWriter(comment=args.comment)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.device = torch.device(args.device)


def main():
    if args.graph == "geo":
        if args.task == "speed":
            args.adj_mx = torch.Tensor(utils.load_pickle(args.adj_data)[-1])
        else:
            args.adj_mx, _ = torch.Tensor(utils.get_adjacency_matrix(args.adj_data, args.num_node))
    else:
        args.adj_mx = torch.ones(args.num_node, args.num_node)
    dataloader = utils.load_dataset(args.data_file, args.batch_size, args.batch_size, args.batch_size)
    args.scaler = dataloader['scaler']

    print(str(args))
    engine = Holder(args)
    print("start training...")

    his_loss = []
    val_time = []
    train_time = []

    for epoch_num in range(args.epochs + 1):
        averaged_nfe_record_enc = []
        averaged_nfe_record_dec = []
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            # trainX = torch.Tensor(x[:, x_idx, :, :]).to(args.device)
            # trainy = torch.Tensor(y[:, y_idx, :, :]).to(args.device)
            # x = x[:, x_idx, :, :]
            # y = y[:, y_idx, :, :]
            # print(x.shape)
            trainX = torch.Tensor(x).to(args.device)
            trainy = torch.Tensor(y).to(args.device)

            if args.task == "speed":
                metrics = engine.train(trainX[..., 0:1], trainy[..., 0:1])
            elif args.task == "flow":
                metrics = engine.train(trainX[..., 0:1], trainy[..., 0:1])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            averaged_nfe_enc = engine.model.encoder.ode_func.nfe / engine.model.encoder.ode_dynamics.perform_num
            averaged_nfe_dec = engine.model.decoder.ode_func.nfe / (
                    engine.model.decoder.ode_dynamics.perform_num * args.seq_out)
            averaged_nfe_record_enc.append(averaged_nfe_enc)
            averaged_nfe_record_dec.append(averaged_nfe_dec)
            engine.model.encoder.ode_func.reset()
            engine.model.encoder.ode_dynamics.reset()
            engine.model.decoder.ode_func.reset()
            engine.model.decoder.ode_dynamics.reset()
            if iter % 200 == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
                print("Averaged NFE Encoder:" + str(np.mean(np.asarray(averaged_nfe_record_enc))), flush=True)
                print("Averaged NFE Decoder:" + str(np.mean(np.asarray(averaged_nfe_record_dec))), flush=True)
                if args.recording:
                    utils.record_info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),
                                      "./records/" + args.comment)

        t2 = time.time()
        train_time.append(t2 - t1)
        if args.task == "flow":
            valid_loss = []
            valid_mape = []
            valid_rmse = []
        elif args.task == "speed":
            valid_loss = []
            valid_mape = []
            valid_rmse = []

        print("eval...")
        averaged_nfe_record_enc = []
        averaged_nfe_record_dec = []
        s1 = time.time()
        # dataloader['test_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            # valx = torch.Tensor(x[:, x_idx, :, :]).to(args.device)
            # valy = torch.Tensor(y[:, y_idx, :, :]).to(args.device)
            # x = x[:, x_idx, :, :]
            # y = y[:, y_idx, :, :]
            valx = torch.Tensor(x).to(args.device)
            valy = torch.Tensor(y).to(args.device)
            if args.task == "speed":
                metrics = engine.eval(valx[..., 0:1], valy[..., 0:1])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
            elif args.task == "flow":
                metrics = engine.eval(valx[..., 0:1], valy[..., 0:1])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(epoch_num, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        if args.task == "flow":
            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
        elif args.task == "speed":
            valid_loss = np.array(valid_loss)
            valid_mape = np.array(valid_mape)
            valid_rmse = np.array(valid_rmse)
            valid_loss = np.mean(valid_loss, axis=0)
            valid_mape = np.mean(valid_mape, axis=0)
            valid_rmse = np.mean(valid_rmse, axis=0)

        averaged_nfe_enc = engine.model.encoder.ode_func.nfe / engine.model.encoder.ode_dynamics.perform_num
        averaged_nfe_dec = engine.model.decoder.ode_func.nfe / (
                engine.model.decoder.ode_dynamics.perform_num * args.seq_out)
        averaged_nfe_record_enc.append(averaged_nfe_enc)
        averaged_nfe_record_dec.append(averaged_nfe_dec)
        engine.model.encoder.ode_func.reset()
        engine.model.encoder.ode_dynamics.reset()
        engine.model.decoder.ode_func.reset()
        engine.model.decoder.ode_dynamics.reset()

        if args.recording:
            if args.task == "flow":
                sw.add_scalar('Loss/train', mtrain_loss, global_step=epoch_num)
                sw.add_scalar('Loss/valid', mvalid_loss, global_step=epoch_num)
                sw.add_scalar('MAPE/train', mtrain_mape, global_step=epoch_num)
                sw.add_scalar('MAPE/valid', mvalid_mape, global_step=epoch_num)
                sw.add_scalar('RMSE/train', mtrain_rmse, global_step=epoch_num)
                sw.add_scalar('RMSE/valid', mvalid_rmse, global_step=epoch_num)
            elif args.task == "speed":
                sw.add_scalar('Loss/train', mtrain_loss, global_step=epoch_num)
                sw.add_scalar('Loss/valid 3', valid_loss[0], global_step=epoch_num)
                sw.add_scalar('Loss/valid 6', valid_loss[1], global_step=epoch_num)
                sw.add_scalar('Loss/valid 12', valid_loss[2], global_step=epoch_num)
                sw.add_scalar('Loss.Valid', valid_loss[3], global_step=epoch_num)
                sw.add_scalar('MAPE/train', mtrain_mape, global_step=epoch_num)
                sw.add_scalar('MAPE/valid 3', valid_mape[0], global_step=epoch_num)
                sw.add_scalar('MAPE/valid 6', valid_mape[1], global_step=epoch_num)
                sw.add_scalar('MAPE/valid 12', valid_mape[2], global_step=epoch_num)
                sw.add_scalar('MAPE/valid', valid_mape[3], global_step=epoch_num)
                sw.add_scalar('RMSE/train', mtrain_rmse, global_step=epoch_num)
                sw.add_scalar('RMSE/valid 3', valid_rmse[0], global_step=epoch_num)
                sw.add_scalar('RMSE/valid 6', valid_rmse[1], global_step=epoch_num)
                sw.add_scalar('RMSE/valid 12', valid_rmse[2], global_step=epoch_num)
                sw.add_scalar('RMSE/valid', valid_rmse[3], global_step=epoch_num)

        if args.task == "flow":
            log = 'Epoch: {:03d}, Train Loss: {:.4f} ,Train MAPE: {:.4f}, Train RMSE: {:.4f}, ' \
                  'Valid Loss: {:.4f},  Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(epoch_num, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss,
                             mvalid_mape,
                             mvalid_rmse,
                             (t2 - t1)),
                  flush=True)
        elif args.task == "speed":
            log = 'Epoch: {:03d}, Train Loss: {:.4f} ,Train MAPE: {:.4f}, Train RMSE: {:.4f}, ' \
                  'Valid Loss 3: {:.4f},  Valid MAPE 3: {:.4f}, Valid RMSE 3: {:.4f}, ' \
                  'Valid Loss 6: {:.4f},  Valid MAPE 6: {:.4f}, Valid RMSE 6: {:.4f}, ' \
                  'Valid Loss 12: {:.4f},  Valid MAPE 12: {:.4f}, Valid RMSE 12: {:.4f}, ' \
                  'Valid Loss: {:.4f},  Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, ' \
                  'Training Time: {:.4f}/epoch'
            print(log.format(epoch_num, mtrain_loss, mtrain_mape, mtrain_rmse,
                             valid_loss[0],
                             valid_mape[0],
                             valid_rmse[0],
                             valid_loss[1],
                             valid_mape[1],
                             valid_rmse[1],
                             valid_loss[2],
                             valid_mape[2],
                             valid_rmse[2],
                             valid_loss[3],
                             valid_mape[3],
                             valid_rmse[3],
                             (t2 - t1)), )
        print("Averaged NFE Encoder test:" + str(np.mean(np.asarray(averaged_nfe_record_enc))), flush=True)
        print("Averaged NFE Decoder test:" + str(np.mean(np.asarray(averaged_nfe_record_dec))), flush=True)
        if args.recording:
            if args.task == "speed":
                utils.record_info(
                    log.format(epoch_num, mtrain_loss, mtrain_mape, mtrain_rmse,
                               valid_loss[0],
                               valid_mape[0],
                               valid_rmse[0],
                               valid_loss[1],
                               valid_mape[1],
                               valid_rmse[1],
                               valid_loss[2],
                               valid_mape[2],
                               valid_rmse[2],
                               valid_loss[3],
                               valid_mape[3],
                               valid_rmse[3],
                               (t2 - t1)),
                    "./records/" + args.comment)
            elif args.task == "flow":
                utils.record_info(
                    log.format(epoch_num, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss,
                               mvalid_mape,
                               mvalid_rmse,
                               (t2 - t1)),
                    "./records/" + args.comment)
        engine.lr_sch.step()

        # if epoch_num % 50 == 0:
        #     torch.save(engine.model, "./PEMS-D360_lb4.pkl")
        # torch.save(engine.model.state_dict(), './parameter_12.pkl')
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))




if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
