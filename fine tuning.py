import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os

import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
我们首先将压缩后的数据集下载到路径data_dir之下，然后在该路径将下载好的数据集解压，得到两个文件夹hotdog/train和hotdog/test。
这两个文件夹下面均有hotdog和not-hotdog两个类别文件夹，每个类别文件夹里面是图像文件。
'''
data_dir = './Datasets'
os.listdir(os.path.join(data_dir, "hotdog"))  # ['train', 'test']

'''
我们创建两个ImageFolder实例来分别读取训练数据集和测试数据集中的所有图像文件。
'''
train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))

# print(train_imgs.classes)        # ['hotdog', 'not-hotdog']
# print(train_imgs.class_to_idx)   # {'hotdog': 0, 'not-hotdog': 1}
# print(train_imgs.imgs)
# [('./Datasets\\hotdog/train\\hotdog\\0.png', 0), ('./Datasets\\hotdog/train\\hotdog\\1.png', 0),...]
#
# print(train_imgs[0])     # (<PIL.Image.Image image mode=RGB size=122x144 at 0x2BD39DA7408>, 0)
# print(train_imgs[0][0])  # <PIL.Image.Image image mode=RGB size=122x144 at 0x2BD39DA7408>
# print(train_imgs[0][1])  # 得到的是类别0，即hotdog
'''
下面画出前8张正类图像和最后8张负类图像。可以看到，它们的大小和高宽比各不相同。
'''
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
# d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

'''
在训练时，我们先从图像中裁剪出随机大小和随机高宽比的一块随机区域，然后将该区域缩放为高和宽均为224像素的输入。
测试时，我们将图像的高和宽均缩放为256像素，然后从中裁剪出高和宽均为224像素的中心区域作为输入。
此外，我们对RGB（红、绿、蓝）三个颜色通道的数值做标准化：每个数值减去该通道所有数值的平均值，再除以该通道所有数值的标准差作为输出。
'''
# 指定RGB三个通道的均值和方差来将图像通道归一化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

'''
定义和初始化模型:
我们使用在ImageNet数据集上预训练的ResNet-18作为源模型。这里指定pretrained=True来自动下载并加载预训练的模型参数。
在第一次使用时需要联网下载模型参数。
'''
pretrained_net = models.resnet18(pretrained=True)
# win系统预训练模型保存在 C:\用户名\.cache\torch\.checkpoints下
# ubuntu系统预训练模型保存在 /home/ubuntu/.cache/torch/checkpoints/
'''
下面打印源模型的成员变量fc。作为一个全连接层，它将ResNet最终的全局平均池化层输出变换成ImageNet数据集上1000类的输出。
'''
# print(pretrained_net.fc)  # Linear(in_features=512, out_features=1000, bias=True)

'''
可见此时pretrained_net最后的输出个数等于目标数据集的类别数1000。所以我们应该将最后的fc成修改我们需要的输出类别数:
'''
pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)  # Linear(in_features=512, out_features=2, bias=True)

'''
此时，pretrained_net的fc层就被随机初始化了，但是其他层依然保存着预训练得到的参数。由于是在很大的ImageNet数据集上预训练的，
所以参数已经足够好，因此一般只需使用较小的学习率来微调这些参数，而fc中的随机初始化参数一般需要更大的学习率从头训练。
PyTorch可以方便的对模型的不同部分设置不同的学习参数，我们在下面代码中将fc的学习率设为已经预训练过的部分的10倍。
'''
# ## 获取指定层参数id
output_params = list(map(id, pretrained_net.fc.parameters()))
# id([object])
# 参数说明：object -- 对象。
# 返回值:返回对象的内存地址。
# id() 函数返回对象的唯一标识符，标识符是一个整数。
#
# map(function, iterable, ...)
# map() 会根据提供的函数对指定序列做映射。
# 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
print(output_params)  # fc层 w,b的id
# ## 获取非指定层的参数id
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
# Python内建的filter()函数用于过滤序列。
# 和map()类似，filter()也接收一个函数和一个序列。和map()不同的时，filter()把传入的函数依次作用于每个元素，
# 然后根据返回值是True还是False决定保留还是丢弃该元素。

lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001  # weight_decay梯度惩罚，默认L2
                      )

'''
微调模型:
我们先定义一个使用微调的训练函数train_fine_tuning以便多次调用。
'''
def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

train_fine_tuning(pretrained_net, optimizer)
'''
迁移学习将从源数据集学到的知识迁移到目标数据集上。微调是迁移学习的一种常用技术。
目标模型复制了源模型上除了输出层外的所有模型设计及其参数，并基于目标数据集微调这些参数。而目标模型的输出层需要从头训练。
一般来说，微调参数会使用较小的学习率，而从头训练输出层可以使用较大的学习率。
'''