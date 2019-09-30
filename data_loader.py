import torch
import xarray
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


train_transformer = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize((32,128)),
	transforms.ToTensor(),
	#transforms.Normalize((0.0), (1.0))
	])


def random_x_translation(img):
	translation = int(torch.randint(0, img.size(-1), (1,)))
	out = torch.cat([img[..., translation:], img[..., :translation]], dim=-1)
	return out


def data_normalize(tensor):
	"""
	normalize a tensor to [-1,1]
	"""
	center = (torch.max(tensor) + torch.min(tensor))/2.0
	span = (torch.max(tensor) - torch.min(tensor))/2.0
	normalized_tensor = (tensor - center)/span
	 
	return normalized_tensor, center, span


class MetaDataset(Dataset):
	def __init__(self, data_dir, transfrom):
		training_dataset = xarray.open_dataset(data_dir).load()
		self.wavelengths, self.wc, self.wspan = data_normalize(
											torch.from_numpy(training_dataset.wavelength.values))
		self.angles, self.ac, self.aspan = data_normalize(
											torch.from_numpy(training_dataset.angle.values))
		self.patterns = torch.from_numpy(training_dataset.pattern.values).unsqueeze(1)
		self.transfrom = transfrom


	def __len__(self):
		return len(self.wavelengths)


	def __getitem__(self, idx):
		#if torch.cuda.is_available():
		#	label = torch.cuda.FloatTensor([self.wavelengths[idx], self.angles[idx]])
		#else:
		#	label = torch.FloatTensor([self.wavelengths[idx], self.angles[idx]])
		label = torch.FloatTensor([self.wavelengths[idx], self.angles[idx]])			
		img = random_x_translation(self.patterns[idx])
		img = (self.transfrom(img)-0.5)/0.5

		return img, label


def fetch_dataloader(data_dir, params):
	
	dataloader = DataLoader(MetaDataset(data_dir, train_transformer),
							batch_size=params.batch_size,
							shuffle=True)

	# record normalization parameters
	params.wc = dataloader.dataset.wc
	params.ac = dataloader.dataset.ac
	params.wspan = dataloader.dataset.wspan
	params.aspan = dataloader.dataset.aspan

	return dataloader


