from data.data_loader import Dataset_MTS
from cross_exp.exp_basic import Exp_Basic
from cross_models.cross_former import Crossformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.aligner import ARAligner

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import os
import time
import json
import pickle

import warnings
warnings.filterwarnings('ignore')

class Exp_crossformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_crossformer, self).__init__(args)
    
    def _build_model(self):        
        model = Crossformer(
            self.args.data_dim, 
            self.args.in_len, 
            self.args.out_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model, 
            self.args.d_ff,
            self.args.n_heads, 
            self.args.e_layers,
            self.args.dropout, 
            self.args.baseline,
            self.args.use_revin if hasattr(self.args, 'use_revin') else False,  # 添加 RevIN 参数
            self.device
        ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
            enable_data_cleaning= args.enable_data_cleaning,
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        scale_statistic = {'mean': train_data.scaler.mean, 'std': train_data.scaler.std}
        with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
            pickle.dump(scale_statistic, f)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')
        
        return self.model

    def test(self, setting, save_pred = False, inverse = False):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse = False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        return outputs, batch_y
    
    def eval(self, setting, save_pred = False, inverse = False):
        #evaluate a saved model
        args = self.args
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='test',
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
            scale = True,
            scale_statistic = args.scale_statistic,
            enable_data_cleaning= args.enable_data_cleaning,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(data_loader):
                pred, true = self._process_one_batch(
                    data_set, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                # print('batch_size:', batch_size)
                # print('pred:', pred.shape) 

                # batch_size, out_len, data_dim] => [batch_size, out_len, use_dim] 
                # aligner = ARAligner(ratio=0.1, lags=5)
                # pred = aligner.align(pred, batch_x)
                # use_dim = 9
                # pred = pred[:, :, :use_dim]
                # true = true[:, :, :use_dim]

                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        # print(metrics_all)
        metrics_all = np.stack(metrics_all, axis = 0)
        # print(metrics_all)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)
            
            # 添加可视化代码
            try:
                import matplotlib.pyplot as plt
                
                # 创建可视化文件夹
                vis_folder = os.path.join(folder_path, 'visualization')
                if not os.path.exists(vis_folder):
                    os.makedirs(vis_folder)
                
                # 选择前5个样本和前16个维度进行可视化
                sample_num = min(5, preds.shape[0])
                dim_num = min(16, preds.shape[2])
                
                # 获取输入序列数据
                inputs = []
                with torch.no_grad():
                    for i, (batch_x, _) in enumerate(data_loader):
                        inputs.append(batch_x.detach().cpu().numpy())
                        if len(inputs) * args.batch_size >= sample_num * preds.shape[0] // sample_num + sample_num:
                            break
                
                inputs = np.concatenate(inputs, axis=0)
                
                for ii in range(sample_num):
                    i = ii * preds.shape[0] // sample_num
                    plt.figure(figsize=(25, 30))
                    
                    for j in range(dim_num):
                        plt.subplot(dim_num, 1, j+1)
                        
                        # 绘制输入序列
                        input_seq = inputs[i, :, j]
                        x_input = np.arange(0, len(input_seq))
                        plt.plot(x_input, input_seq, label='input', color='blue', marker='.')
                        
                        # 绘制真实值和预测值
                        x_true = np.arange(len(input_seq), len(input_seq) + len(trues[i, :, j]))
                        plt.plot(x_true, trues[i, :, j], label='true', color='green', marker='o')
                        plt.plot(x_true, preds[i, :, j], label='pred', color='red', marker='*')
                        
                        # 添加垂直线分隔输入和输出
                        plt.axvline(x=len(input_seq)-0.5, color='gray', linestyle='--')
                        
                        plt.legend()
                        plt.title(f'sample {i+1}, dim {j+1}')
                        plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_folder, f'sample_{i+1}_with_input.png'))
                    plt.close()
                
                # 绘制整体性能指标
                plt.figure(figsize=(10, 6))
                metrics_names = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
                metrics_values = [mae, mse, rmse, mape, mspe]
                
                plt.bar(metrics_names, metrics_values)
                plt.title('Overall Performance Metrics')
                plt.savefig(os.path.join(vis_folder, 'metrics.png'))
                plt.close()
                
                print(f"已保存可视化结果到 {vis_folder}")
            except Exception as e:
                print(f"可视化过程出现错误: {e}")
                import traceback
                traceback.print_exc()

        return mae, mse, rmse, mape, mspe
