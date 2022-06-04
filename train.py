# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from layers.focalLoss import FocalLoss
from sklearn import metrics
from data_utils import ABSADatesetReader
from models import LSTM, ASCNN, ASGCN, ASTCN
from get_position import Vocab_post
import time  # 引入time模块
from MultiFocalLoss import MultiFocalLoss

class Instructor:
    def __init__(self, opt,post_vocab):
        self.opt = opt

        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim,post_vocab = args['post_vocab'])

        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False)

        self.model = opt.model_class(absa_dataset.embedding_matrix,post_vocab,opt).to(opt.device)
        self._print_args()
        self.global_f1 = 0.
        self.global_acc = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer,i):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                
                outputs,penal = self.model(inputs)
                if self.opt.losstype is not None:
                    loss = criterion(outputs, targets) + penal
                else:
                    loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, test_f1 = self._evaluate_acc_f1(i)
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if self.opt.save and test_acc > self.global_acc:
                            self.global_acc = test_acc
                            torch.save(self.model.state_dict(),'state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '_acc'+ '.pkl')
                            print('>>> best model saved.')
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            torch.save(self.model.state_dict(), 'state_dict/'+self.opt.model_name+'_'+self.opt.dataset+'.pkl')
                            print('>>> best model saved.')
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, test_acc, test_f1))
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 5:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0    
        return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self,p):
        # switch model to evaluation mode
        self.model.eval()
        ticks = time.time()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        f_out2 = open('log/' + self.opt.model_name + '_' + self.opt.dataset + str(p)+'_test.txt', 'w',encoding='utf-8')
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_outputs,penal= self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)
                k = t_targets.cpu().numpy().tolist()
                temp = torch.argmax(t_outputs, -1).cpu().numpy().tolist()
                batch_num = 0
                for (i, j) in zip(k, t_sample_batched['xuhao']):
                    line = str(j) + "\t" + str(temp[k.index(i)]) + "\n"
                    f_out2.write(line)
                    # print(t_sample_batched['xuhao'][i],temp[k.index(i)])

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f_out2.close()
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def run(self, repeats=3):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        #criterion = MultiFocalLoss()
        #criterion=FocalLoss(3)
        if not os.path.exists('log/'):
            os.mkdir('log/')
       #'_CLayerNum'+str(self.opt.c_layer_num)
        f_out = open('log/'+self.opt.model_name+'_'+self.opt.dataset+'_val.txt', 'w', encoding='utf-8')

        max_test_acc_avg = 0
        max_test_f1_avg = 0
        max_test_acc_true = 0
        max_test_f1_true = 0
        for i in range(repeats):
            print('repeat: ', (i+1))
            f_out.write('repeat: '+str(i+1))
            self._reset_params()
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            max_test_acc, max_test_f1 = self._train(criterion, optimizer,i)
            print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
            f_out.write('max_test_acc: {0}, max_test_f1: {1}'.format(max_test_acc, max_test_f1))
            if max_test_acc > max_test_acc_true:
                max_test_acc_true = max_test_acc
            if max_test_f1 > max_test_f1_true:
                max_test_f1_true = max_test_f1

            max_test_acc_avg += max_test_acc
            max_test_f1_avg += max_test_f1
            # max_test_acc_avg += max_test_acc
            # max_test_f1_avg += max_test_f1
            print('#' * 100)
        print("max_test_acc_avg:", max_test_acc_avg / repeats)
        print("max_test_f1_avg:", max_test_f1_avg / repeats)
        print("max_test_acc_temp:", max_test_acc_true)
        print("max_test_f1_temp:", max_test_f1_true)
        f_out.write('max_test_acc_avg: {0}, max_test_f1_avg: {1}'.format(max_test_acc_avg / repeats, max_test_f1_avg / repeats))
        f_out.write('max_test_acc_temp: {0}, max_test_f1_temp: {1}'.format(max_test_acc_true, max_test_f1_true))
        #print("max_test_acc_avg:", max_test_acc_avg / repeats)
        #print("max_test_f1_avg:", max_test_f1_avg / repeats)

        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lstm', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, rest14, lap14, rest15, rest16')
    parser.add_argument('--vocab_dir',default='datasets/acl-14-short-data/')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', default=30, type=int,help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--c_layer_num', default=4, type=int)
    # encoder self-attention params
    parser.add_argument('--ffn_dim', default=300, type=int)
    #parser.add_argument('--head_num', default=1, type=int)
    parser.add_argument('--num-layers',type=int,default=2,help='Number of layers in each Transformer stack')
    parser.add_argument('--num_heads',type=int,default=3,help='Number of heads in each Transformer layer for multi-headed attention')
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=29, type=int)
    parser.add_argument('--kernel_size', default='2,3,4', type=str)
    parser.add_argument('--num_kernel', default=200, type=int)
    parser.add_argument('--topK_max_pooling', default=1, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--num_labels', default=3, type=int)
    parser.add_argument('--dropout', default=0.4, type=float)
    parser.add_argument('--textdropout', default=0.1, type=float)
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--linear_size', default=64, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--n_heads', default=1, type=int)
    parser.add_argument('--p_drop', default=0.1, type=float)
    parser.add_argument('--d_ff', default=600, type=int)
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    #parser.add_argument('--gcn_dropout', default=0.45, type=float)
    parser.add_argument('--window_size', default=3, type=int)
    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.4, help='GCN layer dropout rate.')
    parser.add_argument('--head_num', default=3, type=int, help='head_num must be a multiple of 3')
    parser.add_argument('--second_layer', type=str, default='max')

    parser.add_argument('--losstype', default=None, type=str, help="['doubleloss', 'orthogonalloss', 'differentiatedloss']")
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--beta', default=0.25, type=float)
    parser.add_argument('--gamma', default=0.25, type=float)
    opt = parser.parse_args()

    model_classes = {
        'lstm': LSTM,
        'ascnn': ASCNN,
        'asgcn': ASGCN,
        'astcn': ASTCN,
    }
    input_colses = {
        'lstm': ['text_indices'],
        'sedcgcn': ['text_indices', 'aspect_indices', 'left_indices','post_emb','seq_graph','pmi_graph'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    post_vocab = Vocab_post.load_vocab(opt.vocab_dir + 'vocab_post.pkl')
    # position
    args = vars(opt)
    args['post_vocab'] = post_vocab
    args['post_size'] = len(post_vocab)
    print(len(post_vocab))
    ins = Instructor(opt,post_vocab)
    ins.run()
