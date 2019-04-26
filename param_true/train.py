import torch
import torch.nn as nn
import torchvision.models as models
from scipy import io
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import IMAGE_Dataset
from pathlib import Path
import copy
from torch.autograd import Variable
import os
os.environ['CUDA_VISIBLE_DEVICES']='1';
torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
TRAINSET_ROOT = '../trainset/'
label_location='./devkit/cars_train_annos.mat'
CUDA_DEVICES = 0
training_acc=0.0
training_loss=0.0

result_train_loss=[]
result_train_acc=[]

def train():
	
	data_transform=transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	train_set=IMAGE_Dataset(label_location,TRAINSET_ROOT,8144,data_transform);
	data_loader=DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)
	
	resnet101=models.resnet101(pretrained=True)
	fc_features=resnet101.fc.in_features
	resnet101.fc=nn.Linear(fc_features,196)
	resnet101=resnet101.cuda(CUDA_DEVICES)
	resnet101.train()

	best_model_params=copy.deepcopy(resnet101.state_dict())
	best_acc=0.0
	num_epochs=30
	criterion = nn.CrossEntropyLoss()
	optimizer=torch.optim.SGD(params=resnet101.parameters(), lr=0.01, momentum=0.9)

		
	for epoch in range(num_epochs):
		print(f'Epoch: {epoch+1}/{num_epochs}')
		print('-'*len(f'Epoch: (epoch+1)/{num_epochs}'))
		
		training_loss=0.0
		training_corrects=0.0
		
		for i, (inputs, labels) in enumerate(data_loader):
			inputs=Variable(inputs.cuda(CUDA_DEVICES))
			labels=Variable(labels.cuda(CUDA_DEVICES))
			
			optimizer.zero_grad()
			outputs=resnet101(inputs)
			_, preds=torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)
			
			loss.backward()
			optimizer.step()
			training_loss+=loss.item()*inputs.size(0)
			training_corrects+=torch.sum(preds==labels.data)
			
		training_loss=training_loss/len(train_set)
		training_acc=training_corrects.double() / len(train_set)

		result_train_loss.append(str(training_loss))     #train_data
		result_train_acc.append(str(training_acc))
		
		if (epoch%2)==0:
			torch.save(resnet101, f'model-{epoch}.pth')
		
		print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')
		if training_acc>best_acc:
			best_acc = training_acc
			best_model_params=copy.deepcopy(resnet101.state_dict())
	resnet101.load_state_dict(best_model_params)
	torch.save(resnet101, f'best_model.pth')
	
if __name__=='__main__':
	train()
	train_loss_file=open('train_loss_file.txt', 'w')
	train_acc_file=open('train_acc_file.txt', 'w')
	for i in result_train_loss:
		train_loss_file.write(i)
		train_loss_file.write('\n')
	for i in result_train_acc:
		train_acc_file.write(i)
		train_acc_file.write('\n')
	train_loss_file.close()
	train_acc_file.close()
	
	
