import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, h, w, outputs, *args, **kwargs):
        super(DQN, self).__init__(*args, **kwargs)
        self._add_layers_to_attrs()
        
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def _add_layers_to_attrs(self):
        self._conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self._bn1 = nn.BatchNorm2d(16)
        self._conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self._bn2 = nn.BatchNorm2d(32)
        self._conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self._bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self._bn1(self._conv1(x)))
        x = F.relu(self._bn2(self._conv2(x)))
        x = F.relu(self._bn3(self._conv3(x)))
        return self.head(x.view(x.size(0), -1))

