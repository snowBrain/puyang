import os
# Using this code to force the usage of any specific GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import os
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torch.utils.data as data
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from math import log10
import torchvision
import cv2
import skimage
import scipy.io
import scipy.misc
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
#from model import losses
from model.networks import *
from utils.model_storage import save_checkpoint
from data.dataloader import *
from evl import write_log, _normalize, _denormalize

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--lr", default="2.5e-4", type=float, help="The learning rate of our network")
parser.add_argument("--save_freq", default="2", type=float, help="The intervals of our model storage intervals")
parser.add_argument("--iter_freq", default="2", type=float, help="The intervals of our model's evaluation intervals")
parser.add_argument("--result_dir", default="./result/", type=str, help="The path of our result images")
parser.add_argument("--model_path", default="./weights/", type=str, help="The path to store our model")
parser.add_argument("--epochs", default="200", type=int, help="The path to store our model")
parser.add_argument("--start_epoch", default="0", type=int, help="The path to store our model")
parser.add_argument("--batch_size", default="2", type=int, help="The path to store our batch_size")
parser.add_argument("--image_dir", default="./data/CelebA-HQ-img/", type=str, help="The path to store our batch_size")
parser.add_argument("--image_list", default="./data/train_fileList.txt", type=str, help="The path to store our batch_size")


  

global opt,model
opt = parser.parse_args()

start_time = time.time()

demo_dataset = ImageDatasetFromFile(
    opt.image_list,
    opt.image_dir)
train_data_loader = data.DataLoader(dataset=demo_dataset, batch_size=opt.batch_size, num_workers=0, drop_last=True,
                                    pin_memory=True)
 
fsrnet = Generator(input_nc = 3, output_nc = 3)
criterion_MSE = nn.MSELoss()
gradloss1 = nn.CrossEntropyLoss()


if torch.cuda.is_available():
    fsrnet = fsrnet.cuda()
    criterion_MSE = criterion_MSE.cuda()
    

optimizerG = optim.RMSprop(fsrnet.parameters(),lr = opt.lr)

if opt.pretrained:
    if os.path.isfile(opt.pretrained):
        print("=> loading model '{}'".format(opt.pretrained))
        weights = torch.load(opt.pretrained)

        # debug
        pretrained_dict = weights['model'].state_dict()
        model_dict = fsrnet.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        fsrnet.load_state_dict(model_dict)
    else:
        print("=> no model found at '{}'".format(opt.pretrained))

for epoch in range(opt.start_epoch,opt.epochs):
    if epoch % opt.save_freq == 0:
        #model, epoch, model_path, iteration, prefix=""
        save_checkpoint(fsrnet, epoch, opt.model_path,0, prefix='_Gfacenet_')
    
    history_loss =[]

    for iteration, batch in enumerate(train_data_loader):
        input, target, heatmaps, gradmaps = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])#, Variable(batch[4])   #  tensor
        heatmaps = heatmaps.type_as(input)
        
        #print(heatmaps.shape)
        #gradmaps = gradmaps.type_as(input)
        #print(gradmaps.shape)
 
        if torch.cuda.is_available():
            input = input.cuda()
            heatmaps = heatmaps.cuda()
            target = target.cuda()
            gradmaps = gradmaps.cuda()
           
        upscaled,boundaries,reconstructed,regrad,srgrad = fsrnet(input)
        fsrnet.zero_grad()

        loss_us = criterion_MSE(upscaled,target)
        #print(upscaled.shape)
        #print(target.shape)
        loss_hm = criterion_MSE(boundaries,heatmaps)
        loss_final = criterion_MSE(reconstructed,target)
        #print(regrad.shape)

        #print(gradmaps.shape)
        loss_grad = criterion_MSE(regrad, gradmaps)
        loss_srgrad = criterion_MSE(srgrad, gradmaps)
        #smallgradmaps1 = smallgradmaps.expand(-1,32,-1,-1)
        #print(smallgradmaps.shape)
        #loss_b1 = criterion_MSE(B2, smallgradmaps1)

        #loss_b2 = criterion_MSE(B3, smallgradmaps1)
        #loss_b3 = criterion_MSE(B4, smallgradmaps1)
        g_loss = loss_us + loss_hm + loss_final + loss_grad + loss_srgrad #+ loss_b1 + loss_b2 + loss_b3
        history_loss.append(g_loss)
        g_loss.backward()
        optimizerG.step()


        info = "===> Epoch[{}]({}/{}): time: {:4.4f}:\n".format(epoch, iteration, len(demo_dataset) // 16,
                                                              time.time() - start_time)
        info += "Total_loss: {:.4f}, Basic Upscale Loss:{:.4f}, Prior Estimation Loss:{:.4f}, Final Reconstruction Loss: {:.4f}\n".format(
            g_loss.float(), loss_us.float(), loss_hm.float(), loss_final.float())
        
        print(info)




        if epoch % opt.iter_freq == 0:
           # model, epoch, model_path, iteration, prefix=""
            if not os.path.isdir(opt.result_dir + '%04d_Grad_SR_network' % epoch):
                os.makedirs(opt.result_dir + '%04d_Grad_SR_network' % epoch)
            if not os.path.isdir(opt.result_dir + '%04d_Final_SR_reconstruction' % epoch):
                os.makedirs(opt.result_dir + '%04d_Final_SR_reconstruction' % epoch)
           # if not os.path.isdir(opt.result_dir + '%04d_FINAL_SR_Grad' % epoch):
             #   os.makedirs(opt.result_dir + '%04d_FINAL_SR_Grad' % epoch)


            final_output = reconstructed.permute(0,2,3,1).detach().cpu().numpy()
            final_output_0 = final_output[0,:,:,:]


            grad_output = regrad.permute(0,2,3,1).detach().cpu().numpy()
            grad_output_0 = grad_output[0,:,:,0]

            sr_grad = srgrad.permute(0,2,3,1).detach().cpu().numpy()
            sr_grad_0 = sr_grad[0,:,:,0]
            scipy.misc.toimage(grad_output_0 *
                               255, high=255, low=0, cmin=0, cmax=255).save(
                    opt.result_dir + '%04d_Grad_SR_network/%d.png' % (epoch, iteration))
            scipy.misc.toimage(final_output_0 * 255, high=255, low=0, cmin=0, cmax=255).save(
                    opt.result_dir + '%04d_Final_SR_reconstruction/%d.png' % (epoch, iteration))
            #scipy.misc.toimage(sr_grad_0 * 255, high=255, low=0, cmin=0, cmax=255).save(
                   # opt.result_dir + '%04d_FINAL_SR_Grad/%d.png' % (epoch, iteration))

plt.subplot(1,1,1)
plt.suptitle('G-Facenet')
plt.xlabel('opt.epochs')
plt.ylabel('loss')
# plt.plot(history_train_loss,label='Train Loss')
plt.plot(history_loss,label='Loss')
plt.legend(loc='lower right')
plt.show()
plt.savefig('./Loss.png')
plt.close()