from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch import optim
from models.testresults import testResult
from models import network 
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import warnings
warnings.filterwarnings('ignore')

#set some random number seed
def setup_seed(seed):
    torch.manual_seed(seed)                                 
    torch.cuda.manual_seed_all(seed)           
    np.random.seed(seed)                       
    torch.backends.cudnn.deterministic = True

#adjust the learning rate at some epoch
def adjust_learning_rate(optimizer, epoch):
    update_list = [15, 17, 20]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

#training process
def train(epoch):
    model.train()
    for batch_idx, data, in enumerate(trainloader):
        if not args.cpu:
            inputs ,labels = data
            data, target = inputs.cuda(), labels.cuda()
        data = torch.tensor(data, dtype=torch.float32)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return

#after training, test the model and get the change detection results
def test():
    global best_kappa
    global best_FA
    global best_MA
    model.eval()
    Label = []
    for i , data in enumerate(testloader,0):
            if not args.cpu: 
                images = data.cuda()
            images = torch.tensor(images, dtype=torch.float32)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted.cpu().numpy()
            predicted.cuda()
            Label.extend(predicted)
    outputLabel = np.array(Label)
    outputImg = outputLabel.reshape((641,613),order = 'F')
    outputImg = Image.fromarray(np.uint8(outputImg*255.0))
    outputImg = outputImg.convert('L')  
    refImg  = plt.imread('/home/rf/wangliang/ref_img/Sendai-B-ref.png')
    testResults = testResult(outputImg,refImg)
    best_FA = 0
    best_MA = 0
    FA, MA = testResults.FA_MA()
    refLabel = refImg.reshape(-1,)
    acc = testResults.Acc(outputLabel,refLabel)
    kappa = testResults.KappaCoef(FA, MA)
    if kappa > best_kappa:
        outputImg.save('./'+num+'-SB-result_'+num1+'.bmp') 
        best_MA = MA
        best_FA = FA
        best_kappa = kappa
        with open('./'+num+'_result-SB_'+num1+'.txt','a') as file_handle:
                file_handle.write("epoch = {}\tFA = {}\tMA = {}\tKappa = {}\n".format(epoch,FA,MA,kappa))
    print('Best FA: {}\tBest MA: {}\tBest kappa: {:.4f}\n'.format(FA,MA,kappa))
    return

#load your train data by using this class
class trainDatasets(Dataset):
    def __init__(self,datasets,label,transforms=None):
        self.datasets = np.load(datasets)
        self.label = np.load(label)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self,idx):
        traindata = self.datasets[idx]
        traindata = np.transpose(traindata,(2,0,1))
        label = self.label[idx]
        data = (traindata,label)
        return data

#load your test data by using this class
class testDatasets(Dataset):
    def __init__(self,datasets,transforms=None):
        self.datasets = np.load(datasets)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self,idx):
        testdata = self.datasets[idx]
        testdata = np.transpose(testdata,(2,0,1))
        return testdata

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',help='set if only CPU is available')
    parser.add_argument('--gpu_id', action='store', default='1',help='gpu_id')
    parser.add_argument('--lr', action='store', default=0.00001,help='the intial learning rate')
    parser.add_argument('--wd', action='store', default=1e-5,help='the intial learning rate')
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--start_epochs', type=int, default=1, metavar='N',help='number of epochs to train_start')
    parser.add_argument('--end_epochs', type=int, default=100, metavar='N',help='number of epochs to train_end')
    # W/A — bits
    parser.add_argument('--Wbits', type=int, default=4)
    parser.add_argument('--Abits', type=int, default=4)
    parser.add_argument('--q_type', type=int, default=0,help='quantization type:0-symmetric,1-asymmetric')
    args = parser.parse_args()
    print('==> Options:',args)
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    setup_seed(1)
    print('==> Preparing data..')
    #load your data here
    train_datasets = trainDatasets(datasets='./'+ num +'YA+YB+SA_train.npy',label='./'+ num +'YA+YB+SA_train_Lab.npy')
    trainloader = DataLoader(train_datasets,shuffle=True,batch_size=args.train_batch_size,num_workers=args.num_workers)
    test_datasets = testDatasets(datasets='./'+ num +'SB_test.npy')
    testloader = DataLoader(test_datasets,shuffle=False,batch_size=args.test_batch_size,num_workers=args.num_workers)
    print('******Initializing model******')
    model = network.Net( abits=args.Abits, wbits=args.Wbits)
    best_kappa = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
    #model to cuda
    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []
    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr, 'weight_decay':args.wd}]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=base_lr, weight_decay=args.wd)
    for epoch in range(args.start_epochs, args.end_epochs):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
    test()