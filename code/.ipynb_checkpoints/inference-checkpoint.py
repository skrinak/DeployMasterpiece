import boto3
import sys
import os
import numpy as np
import cv2
import torch
from torchvision import transforms as T
import torch.nn as nn
from torch import optim
import time
from PIL import Image
from torchvision import models
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import tarfile
import json
import logging

BATCH_SIZE=64
NUM_WORKERS=8
SIZE=[480,640]
IMGPATH_FILE='./imagenetsplitpaths.txt'
SOFT_TARGET_PATH='./resnet152_results.npy'
TEMPERATURE=3
EPOCHS=10
SAVE_PATH='./resnet_{}.pt'
EVAL_INTERVAL=1000
LR=1e-4

LOSS_NET_PATH='./models/resnet_9.pt'
STYLE_TARGET='./manulogo.png'    

style_model = "style_7" # actual filename has .pth

bucket = "stepin2mpiece"
model_dir = "models/"
local_model = style_model + ".pth"
tmp_local_model = "/tmp/"  + local_model
zipped_file = style_model + ".tar.gz"
zipped_local_model = "/tmp/" + style_model + ".tar.gz"
zipped_model_key = model_dir + style_model + ".tar.gz"
model_path = "s3://" + bucket + "/" +zipped_model_key

logger = logging.getLogger("sagemaker-inference")
logger.setLevel(logging.DEBUG)

def input_fn(input_data, content_type):
    """Parse input data payload
    """
    if content_type == "application/json":
        data = json.loads(input_data)        
    return data
        
def model_fn(model_dir):
    
    logger.debug("Inside model_fn", model_dir )
    s3 = boto3.resource('s3')
    key = zipped_model_key
    s3.Bucket(bucket).download_file(key, zipped_file)
    with tarfile.open(zipped_file, "r:gz") as tar:
        tar.extractall(".")
    os.remove(zipped_file)       
    logger.debug("zipped_file: ", zipped_file)
    
    with open(os.path.join(".", local_model), 'rb') as f:
        model_info = torch.load(f)
    logger.debug(model_info)

    net=StyleNetwork(local_model)
    for p in net.parameters():
        p.requires_grad=False
        
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net=net.eval().to(device) #use eval just for safety
    return net

def predict_fn(input_data, model):    
    logger.debug("Inside predict_fn", input_data, model )

    for p in model.parameters():
        p.requires_grad=False
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model=model.eval().to(device) #use eval just for safety
    
    preprocess=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 

    # Replace bucket name with "input_data" 
    s3 = boto3.resource('s3')
    key = 'capture/921212210452_m-test_1668035952560_498dda94-1dbd-4542-9866-67d936c3c9a3.jpg'
    s3.Bucket(bucket).download_file(key, 'tmp.jpg')
    file_object = 'tmp.jpg'
    
    frame = cv2.imread(file_object)
    frame=cv2.resize(frame,(1280,720))
    cv2.imwrite("ImageStart.jpg", frame)

    frame=(frame[:,:,::-1]/255.0).astype(np.float32) #convert BGR to RGB, convert to 0-1 range and cast to float32
    frame_tensor=torch.unsqueeze(torch.from_numpy(frame),0).permute(0,3,1,2)

    tensor_in = preprocess(frame_tensor) #normalize
    tensor_in=tensor_in.to(device) #send to GPU

    tensor_out = model(tensor_in) #stylized tensor
    
    tensor_out=torch.squeeze(tensor_out).permute(1,2,0) #remove batch dimension and convert to HWC (opencv format)
    stylized_frame=(255*(tensor_out.to('cpu').detach().numpy())).astype(np.uint8) #convert to 0-255 range and cast as uint8
    stylized_frame=cv2.resize(stylized_frame, (1280, 720))
    result = cv2.imwrite("ImageEnd.jpg", stylized_frame)
    
    return {"result": result}

def output_fn(prediction_output, accept):
    logger.debug("Inside output_fn", prediction_output, accept )
    if accept == "application/json":
        return json.dumps(prediction_output)
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)



class ImageNetData(Dataset):
	def __init__(self, image_paths, labels=None, size=[320, 240]):
		"""
		image_paths: a list of N paths for images in training set
		labels: soft targets for images as numpy array of shape (N, 1000)
		"""
		super(ImageNetData, self).__init__()
		self.image_paths=image_paths
		self.labels=labels
		self.inputsize=size
		self.transforms=self.random_transforms()
		if self.labels is not None:
			assert len(self.image_paths)==self.labels.shape[0]
			#number of images and soft targets should be the same

	def random_transforms(self):
		normalize_transform=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		#define normalization transform with which the torchvision models
		#were trained

		affine=T.RandomAffine(degrees=5, translate=(0.05, 0.05))
		hflip =T.RandomHorizontalFlip(p=0.7)
		#webcam output often has horizontal flips, we would like our network
		#to be resilient to horizontal flips
		blur=T.GaussianBlur(5) #kernel size 5x5

		rt1=T.Compose([T.Resize(self.inputsize), affine, T.ToTensor(), normalize_transform])
		rt2=T.Compose([T.Resize(self.inputsize), hflip, T.ToTensor(), normalize_transform])
		rt3=T.Compose([T.Resize(self.inputsize), blur, T.ToTensor(), normalize_transform])

		return [rt1, rt2, rt3]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		imgpath=self.image_paths[index]
		img=Image.open(imgpath).convert('RGB') 
		#some images are grayscale and need to be converted into RGB
	
		img_tensor=self.transforms[torch.randint(0,3,[1,1]).item()](img)

		if self.labels is None:
			return img_tensor
		else:
			label_tensor=torch.tensor(self.labels[index,:])
			return img_tensor, label_tensor

class DataManager(object):
	def __init__(self,imgpathfile, labelpath=None, size=[320, 240], use_test_data=False):
		"""
		imgpathfile: a text file containing paths of all images in the dataset
		stored as a list containting three lists for train, valid, test splits
		ex: [[p1,p2,p6...],[p3,p4...],[p5...]]

		labelpath (optional): path of .npy file which has a numpy array 
		of size (N, 1000) containing pre-computed soft targets
		The order of soft targets in the numpy array should correspond to
		the order of images in imgpathfile 

		size (2-list): [width, height] to which all images will be resized
		use_test_data (bool): whether or not to use test data (generally test data is used
		only once after you have verified model architecture and hyperparameters on validation dataset)
		"""

		self.imgpathfile=imgpathfile
		self.labelpath=labelpath
		self.imgsize=size

		assert os.path.exists(self.imgpathfile), 'File {} does not exist'.format(self.imgpathfile)

		self.dataloaders=self.get_data_loaders(use_test_data)

	def get_data_loaders(self, test=False):
		"""
		test (bool): whether or not to get test data loader
		"""

		with open(self.imgpathfile,'r') as f:
			train_paths, valid_paths, test_paths= eval(f.read())

		if self.labelpath is not None:
			all_labels=np.load(self.labelpath)

			assert all_labels.shape[0]== (len(train_paths)+len(valid_paths)+len(test_paths))

			train_labels=all_labels[:len(train_paths),:]
			valid_labels=all_labels[len(train_paths):len(train_paths)+len(valid_paths),:]
			test_labels=all_labels[-len(test_paths):,:]

		else:
			train_labels=None
			valid_labels=None
			test_labels=None

		train_data=ImageNetData(train_paths, train_labels, self.imgsize)
		valid_data=ImageNetData(valid_paths, valid_labels, self.imgsize)
		
		train_loader=DataLoader(train_data, cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
		valid_loader=DataLoader(valid_data, cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
		#evaluation of network (validation) does not require storing gradients, so GPU memory is freed up
		#therefore, validation can be performed at roughly twice the batch size of training for most
		#networks and GPUs. This reduces training time by doubling the throughput of validation

		if test:
			test_data=ImageNetData(test_paths, test_labels, self.imgsize)
			test_loader=DataLoader(test_data, cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)

			return train_loader, valid_loader, test_loader

		return train_loader, valid_loader

class StyleNetwork(nn.Module):
	def __init__(self, loadpath=None):
		super(StyleNetwork, self).__init__()
		self.loadpath=loadpath

		self.layer1 = self.get_conv_module(inc=3, outc=16, ksize=9)

		self.layer2 = self.get_conv_module(inc=16, outc=32)

		self.layer3 = self.get_conv_module(inc=32, outc=64)

		self.layer4 = self.get_conv_module(inc=64, outc=128)

		self.connector1=self.get_depthwise_separable_module(128, 128)

		self.connector2=self.get_depthwise_separable_module(64, 64)

		self.connector3=self.get_depthwise_separable_module(32, 32)

		self.layer5 = self.get_deconv_module(256, 64)

		self.layer6 = self.get_deconv_module(128, 32)

		self.layer7 = self.get_deconv_module(64, 16)

		self.layer8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

		self.activation=nn.Sigmoid()

		if self.loadpath:
			self.load_state_dict(torch.load(self.loadpath, map_location=torch.device('cpu')), strict=False)

	def get_conv_module(self, inc, outc, ksize=3):
		padding=(ksize-1)//2
		conv=nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=ksize, stride=2, padding=padding)
		bn=nn.BatchNorm2d(outc)
		relu=nn.LeakyReLU(0.1)

		return nn.Sequential(conv, bn, relu)

	def get_deconv_module(self, inc, outc, ksize=3):
		padding=(ksize-1)//2
		tconv=nn.ConvTranspose2d(inc, outc, kernel_size=ksize, stride=2, padding=padding, output_padding=padding)
		bn=nn.BatchNorm2d(outc)
		relu=nn.LeakyReLU(0.1)

		return nn.Sequential(tconv, bn, relu)


	def get_depthwise_separable_module(self, inc, outc):
		"""
		inc(int): number of input channels
		outc(int): number of output channels

		Implements a depthwise separable convolution layer
		along with batch norm and activation.
		Intended to be used with inc=outc in the current architecture
		"""
		depthwise=nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1, groups=inc)
		pointwise=nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0, groups=1)
		bn_layer=nn.BatchNorm2d(outc)
		activation=nn.LeakyReLU(0.1)

		return nn.Sequential(depthwise, pointwise, bn_layer, activation)

	def forward(self, x):

		x=self.layer1(x)

		x2=self.layer2(x)

		x3=self.layer3(x2)

		x4=self.layer4(x3)

		xs4=self.connector1(x4)
		xs3=self.connector2(x3)
		xs2=self.connector3(x2)

		c1=torch.cat([x4, xs4], dim=1)

		x5=self.layer5(c1)

		c2=torch.cat([x5, xs3], dim=1)

		x6=self.layer6(c2)

		c3=torch.cat([x6, xs2], dim=1)

		x7=self.layer7(c3)

		out=self.layer8(x7)

		out=self.activation(out)

		return out

class StyleLoss(nn.Module):
	def __init__(self):
		super(StyleLoss, self).__init__()
		pass

	def forward(self, target_features, output_features):

		loss=0

		for target_f,out_f in zip(target_features, output_features):
			#target is batch size 1
			t_bs,t_ch,t_w,t_h=target_f.shape
			assert t_bs ==1, 'Network should be trained for only one target image'

			target_f=target_f.reshape(t_ch, t_w*t_h)
			
			target_gram_matrix=torch.matmul(target_f,target_f.T)/(t_ch*t_w*t_h) #t_ch x t_ch matrix

			i_bs, i_ch, i_w, i_h = out_f.shape

			assert t_ch == i_ch, 'Bug'

			for img_f in out_f: #contains features for batch of images
				img_f=img_f.reshape(i_ch, i_w*i_h)

				img_gram_matrix=torch.matmul(img_f, img_f.T)/(i_ch*i_w*i_h)

				loss+= torch.square(target_gram_matrix - img_gram_matrix).mean()

		return loss

class ContentLoss(nn.Module):
	def __init__(self):
		super(ContentLoss, self).__init__()

	def forward(self, style_features, content_features):
		loss=0
		for sf,cf in zip(style_features, content_features):
			a,b,c,d=sf.shape
			loss+=(torch.square(sf-cf)/(a*b*c*d)).mean()

		return loss

class TotalVariationLoss(nn.Module):
	def __init__(self):
		super(TotalVariationLoss, self).__init__()

	def forward(self, x):
		horizontal_loss=torch.pow(x[...,1:,:]-x[...,:-1,:],2).sum()

		vertical_loss=torch.pow(x[...,1:]-x[...,:-1],2).sum()

		return (horizontal_loss+vertical_loss)/x.numel()



class StyleTrainer(object):
	def __init__(self, student_network, loss_network, style_target_path, data_manager,feature_loss, style_loss, savepath=None):
		self.student_network=student_network
		self.loss_network=loss_network
		
		assert os.path.exists(style_target_path), 'Style target does not exist'
		image=Image.open(style_target_path).convert('RGB').resize(cfg.SIZE[::-1])
		preprocess=T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

		self.style_target=torch.unsqueeze(preprocess(image),0)

		self.manager=data_manager

		self.feature_loss=feature_loss

		self.style_loss=style_loss

		self.total_variation = TotalVariationLoss()

		self.savepath=savepath

        # 		self.writer=SummaryWriter()

		self.optimizer=optim.Adam(self.student_network.parameters(), lr=cfg.LR)

	def save(self, epoch):
		if self.savepath:
			path=self.savepath.format(epoch)
			torch.save(self.student_network.state_dict(), path)
			logger.debug(f'Saved model to {path}')

def resnet_forward(net, x):
	layers_used=['layer1', 'layer2', 'layer3', 'layer4']
	output=[]
	#logger.debug(net._modules.keys())
	for name, module in net._modules.items():
		if name=='fc':
			continue #dont run fc layer since _modules does not include flatten

		x=module(x)
		if name in layers_used:
			output.append(x)
	#logger.debug('Resnet forward method called')
	#[logger.debug(q.shape) for q in output]
	return output

