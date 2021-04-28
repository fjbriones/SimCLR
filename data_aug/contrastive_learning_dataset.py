from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection

from imgaug import augmenters as iaa

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1, channels=3):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        #Original
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        # data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
        #                                       # transforms.RandomHorizontalFlip(),
        #                                       transforms.RandomApply([color_jitter], p=0.8),
        #                                       transforms.RandomGrayscale(p=0.2),
        #                                       GaussianBlur(kernel_size=int(0.1 * size), channels=channels),
        #                                       transforms.ToTensor()])

        data_transforms = transforms.Compose([iaa.Sequential([iaa.SomeOf((1, 5), 
                                              [iaa.LinearContrast((0.5, 1.0)),
                                              iaa.GaussianBlur((0.5, 1.5)),
                                              iaa.Crop(percent=((0, 0.4),(0, 0),(0, 0.4),(0, 0.0)), keep_size=True),
                                              iaa.Crop(percent=((0, 0.0),(0, 0.02),(0, 0),(0, 0.02)), keep_size=True),
                                              iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
                                              iaa.PiecewiseAffine(scale=(0.02, 0.03), mode='edge'),
                                              iaa.PerspectiveTransform(scale=(0.01, 0.02))],
                                              random_order=True)]).augment_image,
                                              transforms.ToTensor()])

        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),

                          'mnist': lambda: datasets.MNIST(self.root_folder, train=True,
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(28, channels=1),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
