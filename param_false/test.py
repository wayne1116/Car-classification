import torch 
import torch.nn as nn
from torchvision import transforms
from dataset import IMAGE_Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn as nn
from torch.autograd import Variable
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

CUDA_DEVICES=0
TESTSET_ROOT='../testset/'
PATH_TO_MODEL='./best_model.pth'
label_location='./devkit/cars_test_annos_withlabels.mat'
classes1=[]

def test():
	data_transform=transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	test_set=IMAGE_Dataset(label_location, TESTSET_ROOT, 8041, data_transform)
	data_loader=DataLoader(dataset=test_set, batch_size=32, shuffle=True, num_workers=1)
	classes=[i for i in range(196)]
	
	model=torch.load(PATH_TO_MODEL)
	model=model.cuda(CUDA_DEVICES)
	model.eval()
	
	total_correct=0
	total=0
	class_correct=list(0. for i in enumerate(classes))
	class_total=list(0. for i in enumerate(classes))
	
	criterion=nn.CrossEntropyLoss()
	test_loss=0
	
	with torch.no_grad():
		for inputs, labels in data_loader:
			inputs=Variable(inputs.cuda(CUDA_DEVICES))
			labels=Variable(labels.cuda(CUDA_DEVICES))
			outputs=model(inputs)
			_, predicted=torch.max(outputs.data,1)
			
			loss=criterion(outputs, labels)
			test_loss+=loss.item()*labels.size(0)
			
			total+=labels.size(0)
			#total_correct+=(predicted==labels).sum().item()
			total_correct+=torch.sum(predicted==labels.data)
			c=(predicted==labels).squeeze()
			
			for i in range(labels.size(0)):
				label=labels[i]
				class_correct[label]+=c[i].item()
				class_total[label]+=1
	total_acc=total_correct.double()/total;
	test_loss=test_loss/total
	print('total: %d' % total)
	print('Accuracy on the ALL test images : %d %%' %(100*total_correct/total))
	for i, c in enumerate(classes):
		print('Accuracy of class %3d : %2d %%' % (c, 100*class_correct[i]/class_total[i]))
		classes1.append(100*class_correct[i]/class_total[i])
	print(f'test_loss: {test_loss:.4f}\n')
	print(f'test_acc: {total_acc:.4f}\n')
	#return test_loss, total_acc
	

if __name__=='__main__':
	"""loss_list=[]
	acc_list=[]
	for i in range(0,30,2):
		filepath='./model-'+str(i)+'.pth'	
		loss,acc=test(filepath)
		loss_list.append(loss)
		acc_list.append(acc)
	test_loss_file=open('test_loss_file.txt', 'w')
	test_acc_file=open('test_acc_file.txt', 'w')
	for i in loss_list:
		test_loss_file.write(str(i))
		test_loss_file.write('\n')
	for i in acc_list:
		test_acc_file.write(str(i))
		test_acc_file.write('\n')
	test_loss_file.close()
	test_acc_file.close()"""
	test()
	classes_file=open('class_correct.txt', 'w')
	for i in classes1:
		classes_file.write(str(i))
		classes_file.write('\n')
	classes_file.close()
