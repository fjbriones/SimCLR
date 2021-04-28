from torchvision import datasets
from torchvision.transforms import transforms
from exceptions.exceptions import InvalidDatasetSelection

class HeadDataset:
	def __init__(self, root_folder):
		self.root_folder = root_folder

	def get_dataset(self, name, train=True, split='train'):
		valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=train, transform=transforms.ToTensor(), download=True),
							'stl10': lambda: datasets.STL10(self.root_folder, split=split, transform=transforms.ToTensor(), download=True),
							'mnist': lambda: datasets.MNIST(self.root_folder, train=train, transform=transforms.ToTensor(), download=True)}

		try:
			dataset_fn= valid_datasets[name]
		except KeyError:
			raise InvalidDatasetSelection()
		else:
			return dataset_fn()

	def get_num_classes(self, name):
		valid_datasets = {'cifar10': 10,
							'stl10': 10,
							'mnist': 10}

		try:
			dataset_classes = valid_datasets[name]
		except KeyError:
			raise InvalidDatasetSelection()
		else:
			return dataset_classes