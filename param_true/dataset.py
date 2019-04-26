from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from scipy import io

class IMAGE_Dataset(Dataset):
	def __init__(self, label_location, data_location, number, transform=None):
		mat=io.loadmat(label_location)
		#self.root_dir = Path(root_dir)
		self.x = []
		self.y= []
		self.transform = transform
		self.num_classes = 196
        #print(self.root_dir.name)
		for i in range(number):
			path=data_location+mat['annotations'][0][i][5][0];
			self.x.append(path)
			self.y.append(int(mat['annotations'][0][i][4][0][0])-1)
			#print(type(mat['annotations'][0][i][4][0][0]))
			#print(int(mat['annotations'][0][i][4][0][0]))

	def __len__(self):
		return len(self.x)
	def __getitem__(self, index):
		image = Image.open(self.x[index]).convert('RGB')
		if self.transform:
			image = self.transform(image)

		return image, self.y[index]
