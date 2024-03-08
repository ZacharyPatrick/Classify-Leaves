from torch import nn


class VGG_block(nn.Module):

    def __init__(self, num_convs, in_channel, out_channel):
        super(VGG_block, self).__init__()
        self.block = []
        self.in_channel = in_channel
        self.out_channel = out_channel
        for _ in range(num_convs):
            self.block.append(nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1))
            self.block.append(nn.ReLU())
            self.in_channel = self.out_channel
        self.block.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return nn.Sequential(*self.block)(x)


# conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]


class VGG(nn.Module):

    def __init__(self, conv_arch, num_classes):
        super(VGG, self).__init__()
        self.conv_arch = conv_arch
        self.num_classes = num_classes
        self.in_channel = 3
        self.vgg_blocks = []
        for block in self.conv_arch:
            num_convs, out_channel = block[0], block[1]
            vgg_block = VGG_block(num_convs, in_channel=self.in_channel, out_channel=out_channel)
            self.vgg_blocks.append(vgg_block)
            self.in_channel = out_channel
        self.sub_net = nn.Sequential(nn.Flatten(), nn.Linear(conv_arch[-1][-1] * 7 * 7, 4096), nn.ReLU(),
                                     nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                                     nn.Linear(4096, self.num_classes))

    def forward(self, x):
        out = nn.Sequential(*self.vgg_blocks)(x)
        out = self.sub_net(out)
        return out