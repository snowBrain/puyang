# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
from os import listdir
import os
from os.path import join
import torch.nn.functional as F
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)
        

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            #print(x_i.shape)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)
        #print(x.shape)
        return x

def loadFromFile(path, datasize):
    if path is None:
        return None, None

    # print("Load from file %s" % path)
    f = open(path)
    data = []
    for idx in range(0, datasize):
        line = f.readline()
        line = line[:-1]
        data.append(line)
    f.close()
    return data


def load_lr_hr_prior(file_path, input_height=256, input_width=256, output_height=256, output_width=256, is_mirror=False,
                     is_gray=True, scale=8.0, is_scale_back=True, is_parsing_map=True):
    if input_width is None:
        input_width = input_height
    if output_width is None:
        output_width = output_height

    # print(file_path)

    img = cv2.imread(file_path)
    # img = Image.open(file_path)

    if is_gray is False:
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
    if is_gray is True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img.shape)
    if is_mirror and random.randint(0, 1) is 0:
        img = ImageOps.mirror(img)

    if input_height is not None:
        img = cv2.resize(img, (input_width, input_height), interpolation=cv2.INTER_CUBIC)
    #print(img.shape)
    if is_parsing_map:
        str = ['skin.png','lbrow.png','rbrow.png','leye.png','reye.png','lear.png','rear.png','nose.png','mouth.png','ulip.png','llip.png']

        hms = np.zeros((128, 128, len(str)))

        for i in range(len(str)):
            (onlyfilePath, img_name) = os.path.split(file_path)
            full_name = onlyfilePath + "/Parsing_Maps/" + img_name[:-4] + "_"+ str[i]
            hm = cv2.imread(full_name, cv2.IMREAD_GRAYSCALE)
            hm_resized = cv2.resize(hm, (128, 128), interpolation=cv2.INTER_CUBIC) / 255.0
            hms[:, :, i] = hm_resized

    img = cv2.resize(img, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
    #print(img.shape)
    
    #img_grad = Grad(img)
    #print(img_grad.shape)
    img_lr = cv2.resize(img, (int(output_width / scale), int(output_height / scale)), interpolation=cv2.INTER_CUBIC)

    if is_scale_back:
        img_lr = cv2.resize(img_lr, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
        return img_lr, img, hms  #, img_grad
    else:
        return img_lr, img, hms  #, img_grad


class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, img_path, input_height=256, input_width=256, output_height=256, output_width=256,
                 is_mirror=False, is_gray=False, upscale=8.0, is_scale_back=True, is_parsing_map=True):
        super(ImageDatasetFromFile, self).__init__()

        self.image_filenames = image_list
        self.upscale = upscale
        self.is_mirror = is_mirror
        self.img_path = img_path
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.is_scale_back = is_scale_back
        self.is_gray = is_gray
        self.is_parsing_map = is_parsing_map
        self.Get_gradient_nopadding = Get_gradient_nopadding()
        self.input_transform = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, idx):

        if self.is_mirror:
            is_mirror = random.randint(0, 1) is 0
        else:
            is_mirror = False

        image_filenames = loadFromFile(self.image_filenames, len(open(self.image_filenames, 'r').readlines()))
        fullpath = join(self.img_path, image_filenames[idx])

        lr, hr, pm = load_lr_hr_prior(fullpath,
                                      self.input_height, self.input_width, self.output_height, self.output_width,
                                      self.is_mirror, self.is_gray, self.upscale, self.is_scale_back,
                                      self.is_parsing_map)

        input = self.input_transform(lr)
        #print(input.shape)
        #+smallimg = cv2.resize(hr, (64, 64), interpolation=cv2.INTER_CUBIC)
       # smallgrad = self.input_transform(smallimg)
        target = self.input_transform(hr)
        #print(target.shape) #([3,128,128])
        parsing_map = self.input_transform(pm)
        #print(parsing_map.shape)
        grad_img = torch.unsqueeze(target, 0)
      #  grad_small = torch.unsqueeze(smallgrad, 0)
        #print(grad_img.shape)
        grad_img = self.Get_gradient_nopadding(grad_img)
      #  grad_small = self.Get_gradient_nopadding(grad_small)
        #print(grad_img.shape)
       # grad_small = torch.squeeze(grad_small)
        grad_img  = torch.squeeze(grad_img)
        #print(grad_img.shape) 
        return input, target, parsing_map, grad_img#, grad_small

    def __len__(self):
        return len(open(self.image_filenames, 'rU').readlines())


#demo_dataset = ImageDatasetFromFile("/home/cydia/文档/毕业设计/make_Face_boundary/81_landmarks/fileList.txt",
                                    #"/home/cydia/图片/sample/")

#train_data_loader = data.DataLoader(dataset=demo_dataset, batch_size=1, num_workers=0)

if __name__ == '__main__':
    for titer, batch in enumerate(train_data_loader):
        input, target, heatmaps, grad_img = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])

        Input = input.permute(0, 2, 3, 1).cpu().data.numpy()
        Target = target.permute(0, 2, 3, 1).cpu().data.numpy()
        Parsing_maps = heatmaps.permute(0, 2, 3, 1).cpu().data.numpy()
        Grad_img = grad_img.permute(0, 2, 3, 1).cpu().data.numpy()
        
        plt.figure("Input Image")
        plt.imshow(Input[0, :, :, :])
        plt.axis('on')
        plt.title('image')
        plt.show()
        

        plt.figure("Grad Image")
        plt.imshow(Grad_img[0, :, :, 0])
        plt.axis('on')
        plt.title('grad')
        plt.show()

        plt.figure("Target Image")
        plt.imshow(Target[0, :, :, :])
        plt.axis('on')
        plt.title('Target')
        plt.show()

        plt.figure("HMS")
        plt.imshow(Parsing_maps[0, :, :, 0])
        plt.axis('on')
        plt.title('OMS')
        plt.show()
