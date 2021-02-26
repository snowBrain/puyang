import sys
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from data.dataloader import Get_gradient_nopadding
###############################################################################
# Functions
###############################################################################


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0)
	elif classname.find('BatchNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer

def define_coarse_SR_Encoder_1(norm_layer):
	#nn.ReflectionPad2d类似于一种 镜像填充 百度即可
	coarse_SR_Encoder_1 = [nn.ReflectionPad2d(1),
							nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=False),
							norm_layer(32),#大小没有改变 
							nn.ReLU(True)]#大小没有改变
	coarse_SR_Encoder_1 = nn.Sequential(*coarse_SR_Encoder_1)
	return coarse_SR_Encoder_1 #will be catted with the output of B1

def define_coarse_SR_Encoder_2(norm_layer):
	for i in range(3):
		coarse_SR_Encoder_2 = [ResnetBlock(64, 'reflect', norm_layer, False, False)]
	coarse_SR_Encoder_2 += [nn.ReflectionPad2d(1),
							nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0, bias=False),
							nn.Tanh()]
	coarse_SR_Encoder_2 = nn.Sequential(*coarse_SR_Encoder_2)
	return coarse_SR_Encoder_2   #[3,128,128]

def define_fine_SR_Encoder_1(norm_layer):
	fine_SR_Encoder_1 = [nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
						norm_layer(32),
						nn.ReLU(True)]
	fine_SR_Encoder_1 = nn.Sequential(*fine_SR_Encoder_1)
	return fine_SR_Encoder_1 #will be catted with the output of B2 which then sent to next layer   [32,64,64]

def define_fine_SR_Encoder_2(norm_layer):
	for i in range(12):
		fine_SR_Encoder_2 = [ResnetBlock(64, 'reflect', norm_layer, False, False)]
	fine_SR_Encoder_2 += [nn.ReflectionPad2d(1),
						nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
						nn.Tanh()]
	fine_SR_Encoder_2 = nn.Sequential(*fine_SR_Encoder_2)
	return fine_SR_Encoder_2 # [64,64,64] cattde with b3  [96,64,64]

def define_fine_SR_Encoder_3(norm_layer):
	fine_SR_Encoder_3 = [nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
						norm_layer(64),
						nn.ReLU(True)]
	for i in range(12):
		fine_SR_Encoder_3 += [ResnetBlock(64, 'reflect', norm_layer, False, False)]
	fine_SR_Encoder_3 += [nn.ReflectionPad2d(1),
						nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
						nn.Tanh()]
	fine_SR_Encoder_3 = nn.Sequential(*fine_SR_Encoder_3)
	return fine_SR_Encoder_3
def define_prior_Estimation_Network(norm_layer):
	prior_Estimation_Network = [nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
								norm_layer(64),
								nn.ReLU(True)]
	prior_Estimation_Network += [Residual(64, 128)]
	for i in range(3):
		prior_Estimation_Network += [ResnetBlock(128, 'reflect', norm_layer, False, False)]
	for i in range(2):
		prior_Estimation_Network += [HourGlassBlock(128, 3, norm_layer)]
	prior_Estimation_Network += [nn.Conv2d(128, 11, kernel_size=1, stride=1, padding=0, bias=False)]	
	prior_Estimation_Network = nn.Sequential(*prior_Estimation_Network)
	return prior_Estimation_Network

def define_fine_SR_Decoder_1(norm_layer):
	fine_SR_Decoder_1 = [nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=False),
						norm_layer(64),
						nn.ReLU(True)]
	fine_SR_Decoder_1 += [nn.UpsamplingNearest2d(scale_factor=2),
						nn.ReflectionPad2d(1),
						nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
						norm_layer(64),
						nn.ReLU(True)] # 可能是这一步导致了棋盘伪影
	for i in range(3):
		fine_SR_Decoder_1 += [ResnetBlock(64, 'reflect', norm_layer, False, False)]
	fine_SR_Decoder_1 += [nn.ReflectionPad2d(1),
						nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0, bias=False),
						nn.Tanh()]
	fine_SR_Decoder_1 = nn.Sequential(*fine_SR_Decoder_1) #B3 and fine1 will be sent into this network
	return fine_SR_Decoder_1
def define_fine_SR_Decoder_2(norm_layer):
	fine_SR_Decoder_2 = [nn.Conv2d(107, 64, kernel_size=3, stride=1, padding=1, bias=False),
						norm_layer(64),
						nn.ReLU(True)]
	fine_SR_Decoder_2 += [nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
						nn.ReflectionPad2d(1),
						nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
						norm_layer(64),
						nn.ReLU(True)]  # 可能是这一步导致了棋盘伪影
	for i in range(3):
		fine_SR_Decoder_2 += [ResnetBlock(64, 'reflect', norm_layer, False, False)]
	fine_SR_Decoder_2 += [nn.ReflectionPad2d(1),
						nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0, bias=False),
						nn.Tanh()]
	fine_SR_Decoder_2 = nn.Sequential(*fine_SR_Decoder_2)
	return fine_SR_Decoder_2

def Reconstruct(norm_layer):
	Reconstruct = [nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
						norm_layer(64),
						nn.ReLU(True)]
	Reconstruct += [nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
						norm_layer(64),
						nn.ReLU(True)]
	for i in range(3):
		Reconstruct += [ResnetBlock(64, 'reflect', norm_layer, False, False)]
	Reconstruct += [nn.ReflectionPad2d(1),
						nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0, bias=False),#可能需要改成1x1的卷积核
						nn.Tanh()]
	Reconstruct = nn.Sequential(*Reconstruct)
	return Reconstruct
def rgd(norm_layer):
	rgd = [nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=False),
				norm_layer(3),
				nn.Tanh()]
	rgd = nn.Sequential(*rgd)
	return rgd


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], use_parallel=True, learn_residual=False):
	netG = None
	use_gpu = len(gpu_ids) > 0
	norm_layer = get_norm_layer(norm_type=norm)

	if use_gpu:
		assert(torch.cuda.is_available())

	netG = Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
	
	if len(gpu_ids) > 0:
		netG.cuda(gpu_ids[0])
	netG.apply(weights_init)
	return netG


def define_D(input_nc, ndf, which_model_netD,
			 n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], use_parallel = True):
	netD = None
	use_gpu = len(gpu_ids) > 0
	norm_layer = get_norm_layer(norm_type=norm)

	if use_gpu:
		assert(torch.cuda.is_available())
	if which_model_netD == 'basic':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, use_parallel=use_parallel)
	elif which_model_netD == 'n_layers':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, use_parallel=use_parallel)
	else:
		raise NotImplementedError('Discriminator model name [%s] is not recognized' %
								  which_model_netD)
	if use_gpu:
		netD.cuda(gpu_ids[0])
	netD.apply(weights_init)
	return netD




def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class Generator(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[0], use_parallel=True, learn_residual=False, padding_type='reflect'):
		assert(n_blocks >= 0)
		super(Generator, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		self.gpu_ids = gpu_ids
		self.use_parallel = use_parallel
		self.learn_residual = learn_residual
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		self._1x1_conv = _1x1_conv()
		self.Detailnet_7x7 = Detailnet_7x7() #B1
		self.Detailnet_5x5 = Detailnet_5x5() #B2
		self.Detailnet_3x3 = Detailnet_3x3() #B3
		self.Detailnet_3x3_1 = Detailnet_3x3_1() #B4
		self.define_coarse_SR_Encoder_1 = define_coarse_SR_Encoder_1(norm_layer)
		self.define_coarse_SR_Encoder_2 = define_coarse_SR_Encoder_2(norm_layer)
		self.prior_Estimation_Network = define_prior_Estimation_Network(norm_layer)
		self.define_fine_SR_Encoder_1 =  define_fine_SR_Encoder_1(norm_layer)
		self.define_fine_SR_Encoder_2 =  define_fine_SR_Encoder_2(norm_layer)
		self.define_fine_SR_Encoder_3 =  define_fine_SR_Encoder_3(norm_layer)
		self.define_fine_SR_Decoder_1 = define_fine_SR_Decoder_1(norm_layer)
		self.define_fine_SR_Decoder_2 = define_fine_SR_Decoder_2(norm_layer)
		self.Get_gradient_nopadding = Get_gradient_nopadding()
		self.Reconstruct = Reconstruct(norm_layer) # grad branch reconstruct
	def forward(self, input, is_hr=False):
		if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
			if is_hr == True:
				heatmaps = nn.parallel.data_parallel(self.prior_Estimation_Network, input, self.gpu_ids)
				return heatmaps
			else:
				#grad branch
				input_grad = self.Get_gradient_nopadding(input)
				print(input_grad.shape)
				B_0 = nn.parallel.data_parallel(self._1x1_conv, input_grad, self.gpu_ids)
				B_1 = nn.parallel.data_parallel(self.Detailnet_7x7, B_0, self.gpu_ids)
				B_2 = nn.parallel.data_parallel(self.Detailnet_5x5, B_1, self.gpu_ids)
				B_3 = nn.parallel.data_parallel(self.Detailnet_3x3, B_2, self.gpu_ids)
				B_4 = nn.parallel.data_parallel(self.Detailnet_3x3_1, B_3, self.gpu_ids)
				grad = nn.parallel.data_parallel(self.Reconstruct, B_4, self.gpu_ids)
				#main branch
				A_0 = nn.parallel.data_parallel(self.define_coarse_SR_Encoder_1, input, self.gpu_ids)
				A_1 = torch.cat((A_0, B_1), 1)
				coarse_HR = nn.parallel.data_parallel(self.define_coarse_SR_Encoder_2, A_1, self.gpu_ids)
				A_3 = nn.parallel.data_parallel(self.define_fine_SR_Encoder_1, coarse_HR, self.gpu_ids)
				A_4 = torch.cat((A_3, B_2), 1)
				A_5 = nn.parallel.data_parallel(self.define_fine_SR_Encoder_2, A_4, self.gpu_ids)
				A_6 = torch.cat((A_5, B_3), 1)
				A_7 = nn.parallel.data_parallel(self.define_fine_SR_Decoder_1, A_6, self.gpu_ids)
				A_8 = nn.parallel.data_parallel(self.define_fine_SR_Encoder_3, A_7, self.gpu_ids)
				heatmaps = nn.parallel.data_parallel(self.prior_Estimation_Network, A_7, self.gpu_ids)
				A_10 = torch.cat((A_8, heatmaps), 1)
				A_11 = torch.cat((A_10, B_4), 1)
				output = nn.parallel.data_parallel(self.define_fine_SR_Decoder_2, A_11, self.gpu_ids)
				grad_sr = self.Get_gradient_nopadding(output)
				
				#print(grad_sr.shape)
				#grad_sr = nn.parallel.data_parallel(self.rcons, output_sr, self.gpu_ids)

		else:
			if is_hr == True:
				heatmaps = self.prior_Estimation_Network(input)
				return heatmaps
			#不执行下面这个部分
			else:
				input_grad = self.get_g_nopadding(input)
				#print(input_grad.shape)
				input_grad_1 = self._1x1_conv(input_grad) #[32,128,128]
				coarse_HR = nn.parallel.data_parallel(self.coarse_SR_Encoder_1, input, self.gpu_ids)
				parsing_hr= nn.parallel.data_parallel(self.fine_SR_Encoder, coarse_HR, self.gpu_ids)
				cat_1 = nn.parallel.data_parallel(self.fine_SR_Decoder_1, parsing_hr, self.gpu_ids)
				output_hr = nn.parallel.data_parallel(self.fine_SR_Decoder_2, cat_1, self.gpu_ids)
				parsing = nn.parallel.data_parallel(self.fine_SR_Encoder,output_hr, self.gpu_ids)
				heatmaps = nn.parallel.data_parallel(self.prior_Estimation_Network, output_hr, self.gpu_ids)
				concatenation = torch.cat((parsing, heatmaps), 1)
				output_1 = nn.parallel.data_parallel(self.fine_SR_Decoder_3, concatenation, self.gpu_ids)
				#output = nn.parallel.data_parallel(self.fine_SR_Decoder_4, output_1, self.gpu_ids)
				#branch
				
				#print(B_0.shape)
				B_1 = nn.parallel.data_parallel(self.Detailnet_7x7, input_grad_1, self.gpu_ids)
				B_2 = torch.cat((B_1, cat_1), 1)
				B_3 = nn.parallel.data_parallel(self.Detailnet_5x5, B_2, self.gpu_ids)
				B_4 = torch.cat((B_3, output_hr), 1)
				B_5 = nn.parallel.data_parallel(self.Detailnet_3x3, B_4, self.gpu_ids)
				B_6 = nn.parallel.data_parallel(self.Detailnet_3x3_1, B_5, self.gpu_ids)
				output_2 = torch.cat((B_6, output_1), 1)
				output = nn.parallel.data_parallel(self.fine_SR_Decoder_4, output_2, self.gpu_ids)
				grad = nn.parallel.data_parallel(self.Reconstruct, B_6, self.gpu_ids) 

		if self.learn_residual:
			output = input + output
			output = torch.clamp(output, min = -1, max = 1)
		return coarse_HR, heatmaps, output, grad, grad_sr

#Define a hourglass block
class HourGlassBlock(nn.Module):
	def __init__(self, dim, n, norm_layer):
		super(HourGlassBlock, self).__init__()
		self._dim = dim
		self._n = n
		self._norm_layer = norm_layer
		self._init_layers(self._dim, self._n, self._norm_layer)

	def _init_layers(self, dim, n, norm_layer):
		setattr(self, 'res'+str(n)+'_1', Residual(dim, dim))
		setattr(self, 'pool'+str(n)+'_1', nn.MaxPool2d(2,2))
		setattr(self, 'res'+str(n)+'_2', Residual(dim, dim))
		if n > 1:
			self._init_layers(dim, n-1, norm_layer)
		else:
			self.res_center = Residual(dim, dim)
		setattr(self,'res'+str(n)+'_3', Residual(dim, dim))
		setattr(self,'unsample'+str(n), nn.Upsample(scale_factor=2))

	def _forward(self, x, dim, n):
		up1 = x
		up1 = eval('self.res'+str(n)+'_1')(up1)
		low1 = eval('self.pool'+str(n)+'_1')(x)
		low1 = eval('self.res'+str(n)+'_2')(low1)
		if n > 1:
			low2 = self._forward(low1, dim, n-1)
		else:
			low2 = self.res_center(low1)
		low3 = low2
		low3 = eval('self.'+'res'+str(n)+'_3')(low3)
		up2 = eval('self.'+'unsample'+str(n)).forward(low3)
		out = up1 + up2
		return out

	def forward(self, x):
		return self._forward(x, self._dim, self._n)
class _1x1_conv(nn.Module):
	def __init__(self, ins_c = 3, outs_c = 32):
		super(_1x1_conv, self).__init__()
		self._1x1_conv = nn.Conv2d(ins_c, outs_c,1)
	def forward(self, x):
		x = self._1x1_conv(x)
		return x	

class Residual(nn.Module):
    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins,outs//2,1),
            nn.BatchNorm2d(outs//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs//2,outs//2,3,1,1),
            nn.BatchNorm2d(outs//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs//2,outs,1)
        )
        if ins != outs:
            self.skipConv = nn.Conv2d(ins,outs,1)
        self.ins = ins
        self.outs = outs
    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x

class Detailnet_7x7(nn.Module):
	def __init__(self, ins=32, outs=32):
		super(Detailnet_7x7, self).__init__()
		self.Block7x7 = nn.Sequential(
			nn.Conv2d(ins,outs,3,1,1,bias=False),
			nn.Conv2d(ins,outs,3,1,1,bias=False),
			nn.Conv2d(ins,outs,3,1,1,bias=False),
			nn.BatchNorm2d(outs),
			nn.ReLU(inplace=True),
			nn.Conv2d(outs,outs,3,1,1,bias=False),
			nn.Conv2d(outs,outs,3,1,1,bias=False),
			nn.BatchNorm2d(outs)
		)
		self.conv1 = nn.Conv2d(ins, outs, 1)
	def forward(self, x):
		residual = self.conv1(x)
		x = self.Block7x7(x)
		x += residual
		nn.ReLU(inplace=True)
		x = x
		return x
class Detailnet_5x5(nn.Module):
	def __init__(self, ins=32, outs=32):
		super(Detailnet_5x5, self).__init__()
		self.Block5x5 = nn.Sequential(
			nn.Conv2d(ins,outs,3,1,1,bias=False),
			nn.Conv2d(ins,outs,3,1,1,bias=False),
			nn.BatchNorm2d(outs),
			nn.ReLU(inplace=True),
			nn.Conv2d(outs,outs,3,2,1,bias=False),
			nn.BatchNorm2d(outs)
		)
		self.conv1 = nn.Conv2d(ins,outs,3,2,1)
	def forward(self, x):
		residual = self.conv1(x)
		x = self.Block5x5(x)
		x += residual
		nn.ReLU(inplace=True)
		x = x
		return x
class Detailnet_3x3(nn.Module):
	def __init__(self, ins=32, outs=32):
		super(Detailnet_3x3, self).__init__()
		self.Block3x3 = nn.Sequential(
			nn.Conv2d(ins,outs,3,1,1,bias=False),
			nn.BatchNorm2d(outs),
			nn.ReLU(inplace=True),
			nn.Conv2d(outs,outs,3,1,1,bias=False),
			nn.BatchNorm2d(outs)
		)
	def forward(self, x):
		residual = x
		x = self.Block3x3(x)
		x += residual
		nn.ReLU(inplace=True)
		x = x
		return x
class Detailnet_3x3_1(nn.Module):
	def __init__(self, ins=32, outs=32):
		super(Detailnet_3x3_1, self).__init__()
		self.Block3x1 = nn.Sequential(
			nn.Conv2d(ins,outs,3,1,1),
			nn.BatchNorm2d(outs),
			nn.ReLU(inplace=True),
			nn.Conv2d(outs,outs,3,1,1),
			nn.BatchNorm2d(outs)
		)
	def forward(self, x):
		residual = x
		x = self.Block3x1(x)
		x += residual
		nn.ReLU(inplace=True)
		x = x
		return x

class GD(nn.Module):
	def __init__(self):
		super(GD, self).__init__()
		self.Get_gradient_nopad = Get_gradient_nopadding()
		self.block = nn.Sequential(
			nn.Tanh()
		)
	def forward(self, x):
		x =self.Get_gradient_nopad(x)
		x =self.block(x)
		return x	
# Define a resnet block
class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, updimension=False):
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, updimension)

	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, updimension):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		in_chan = dim
		if updimension == True:
			out_chan = in_chan * 2
		else:
			out_chan = dim
		conv_block += [nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=p, bias=use_bias),
						norm_layer(dim),
						nn.ReLU(True)]

		if use_dropout:
			conv_block += [nn.Dropout(0.5)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=p, bias=use_bias),
					   norm_layer(dim)]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out

class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], use_parallel = True):
		super(NLayerDiscriminator, self).__init__()
		self.gpu_ids = gpu_ids
		self.use_parallel = use_parallel
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		kw = 4
		padw = int(np.ceil((kw-1)/2))
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
			nn.LeakyReLU(0.2, True)
		]

		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mult = min(2**n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
						  kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]

		nf_mult_prev = nf_mult
		nf_mult = min(2**n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
					  kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]

		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

		if use_sigmoid:
			sequence += [nn.Sigmoid()]

		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
			return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			return self.model(input)
