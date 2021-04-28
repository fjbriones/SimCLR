import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, in_channels):
        super(ResNetSimCLR, self).__init__()
        # self.resnet_dict = {"resnet18": ResNetChannel(18, in_channels=in_channels, num_classes=out_dim),
        #                     "resnet50": ResNetChannel(50, in_channels=in_channels, num_classes=out_dim)}

        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.in_channels = in_channels

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            if self.in_channels == 1:
                model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)

class ResNetChannel(models.resnet.ResNet):

    def __init__(self, num_layers, num_classes, pretrained=False, in_channels=3):

        if num_layers==18:
            block = models.resnet.BasicBlock
            layers = [2,2,2,2]
        elif num_layers==50:
            block = models.resnet.Bottleneck
            layers = [3,4,6,3]

        if in_channels != 3:
            assert not pretrained


        super(ResNetChannel, self).__init__(block, layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # super(ResNetChannel, self).conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # print(self.inplanes)
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = self._norm_layer(self.inplanes)
