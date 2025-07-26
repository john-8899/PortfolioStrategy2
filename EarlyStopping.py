# -*- coding:utf-8 -*-
"""
    EarlyStopping是一种在机器学习和深度学习中常用的正则化技术，用于避免模型过拟合。
    其工作原理基于一个观察：随着训练的进行，模型在训练数据上的表现通常会持续提升，
    但在验证数据集上的表现可能在某个点之后开始恶化，这表明模型开始过拟合训练数据，
    即模型学习到了训练数据中的噪声或细节，而这些并不适用于未见过的数据。

    EarlyStopping的基本原理如下：
    (1)初始化：
        设置一个“耐心”参数（patience），它表示在验证集上性能没有显著改进后，模型还会继续训练多少个周期（epochs）。
        设定一个最小改变量（min_delta），用来判断性能是否真的有提升。例如，如果验证损失的减少小于min_delta，那么这次变化将不被视为改进。
    (2)监测验证性能：
        在每个训练周期结束后，计算模型在验证集上的性能指标，通常是损失（loss）或准确率（accuracy）。
        __call__ 参数里面：loss传入正值， accuracy 传入负值
    (3)比较与决策：
        如果验证集上的性能比之前的最佳性能更好（或者差值小于min_delta），则更新最佳性能记录，并保存当前模型的状态（权重）。如果验证集上的性能没有提高，增加一个内部计数器。当内部计数器达到patience设定的值时，训练过程被终止，模型恢复到之前记录的最佳状态。

"""
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_model = None #记录最优模型

    def __call__(self, val_loss, model):
        '''
            val_loss:
                loss 传入正值
                score 传入负值
        '''
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            #print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.best_model = model
        #torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss