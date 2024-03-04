from tqdm import tqdm
import torch
import torchvision as tv
from torch.utils.data import DataLoader
import torch.nn as nn


class Config(object):
    """
    定义一个配置类
    """
    # 0.参数调整
    data_path = 'D:\pythonProject9\GAN_image_data'
    num_workers = 4  # 多线程
    img_size = 96  # 剪切图片的像素大小
    batch_size = 1  # 批处理数量
    max_epoch = 1000  # 最大轮次
    lr1 = 2e-4  # 生成器学习率
    lr2 = 2e-4  # 判别器学习率
    beta1 = 0.5  # 正则化系数，Adam优化器参数
    gpu = True  # 是否使用GPU运算（建议使用）
    nz = 100  # 噪声维度
    ngf = 64  # 生成器的卷积核个数
    ndf = 64  # 判别器的卷积核个数
    # 1.模型保存路径
    save_path = 'D:\pythonProject9\Img10'  # opt.netg_path生成图片的保存路径
    # 判别模型的更新频率要高于生成模型
    d_every = 1  # 每一个batch 训练一次判别器
    g_every = 5  # 每1个batch训练一次生成模型
    save_every = 5  # 每save_every次保存一次模型
    netd_path = None
    netg_path = None
    # 选择保存的照片
    # 一次生成保存64张图片
    gen_num = 1
    gen_search_num = 35
    gen_mean = 0  # 生成模型的噪声均值
    gen_std = 1  # 噪声方差


# 实例化Config类，设定超参数，并设置为全局参数
opt = Config()


class NetG(nn.Module):
    # 构建初始化函数，传入opt类
    def __init__(self, opt):
        super(NetG, self).__init__()
        self.ngf = opt.ngf
        self.Gene = nn.Sequential(
            nn.ConvTranspose2d(in_channels=opt.nz, out_channels=self.ngf * 8, kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.ngf * 8, out_channels=self.ngf * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.ngf * 4, out_channels=self.ngf * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.ngf * 2, out_channels=self.ngf, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.Gene(x)


# 构建Discriminator判别器
class NetD(nn.Module):
    def __init__(self, opt):
        super(NetD, self).__init__()
        self.ndf = opt.ndf
        self.Discrim = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ndf, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.ndf * 4, out_channels=self.ndf * 8, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 展平后返回
        return self.Discrim(x).view(-1)


def train(**kwargs):
    # 配置属性
    # 如果函数无字典输入则使用opt中设定好的默认超参数
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    # device(设备)，分配设备
    if opt.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.img_size),
        tv.transforms.CenterCrop(opt.img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(root=opt.data_path, transform=transforms)

    dataloader = DataLoader(
        dataset,  # 数据加载
        batch_size=opt.batch_size,  # 批处理大小设置
        shuffle=True,  # 是否进行洗牌操作
        drop_last=True  # 为True时，如果数据集大小不能被批处理大小整除，则设置为删除最后一个不完整的批处理。
    )

    # 初始化网络
    netg, netd = NetG(opt), NetD(opt)
    # 判断网络是否有权重数值
    map_location = lambda storage, loc: storage
    if opt.netg_path:
        netg.load_state_dict(torch.load(f=opt.netg_path, map_location=map_location))
    if opt.netd_path:
        netd.load_state_dict(torch.load(f=opt.netd_path, map_location=map_location))
    netd.to(device)
    netg.to(device)
    optimize_g = torch.optim.Adam(netg.parameters(), lr=opt.lr1, betas=(opt.beta1, 0.999))
    optimize_d = torch.optim.Adam(netd.parameters(), lr=opt.lr2, betas=(opt.beta1, 0.999))
    criterions = nn.BCELoss().to(device)
    true_labels = torch.ones(opt.batch_size).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)
    noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    fix_noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    for epoch in range(opt.max_epoch):
        for ii_, (img, _) in tqdm((enumerate(dataloader))):
            real_img = img.to(device)
            if ii_ % opt.d_every == 0:
                optimize_d.zero_grad()
                output = netd(real_img)
                error_d_real = criterions(output, true_labels)
                error_d_real.backward()
                noises = noises.detach()
                fake_image = netg(noises).detach()
                output = netd(fake_image)
                error_d_fake = criterions(output, fake_labels)
                error_d_fake.backward()
                optimize_d.step()
            if ii_ % opt.g_every == 0:
                optimize_g.zero_grad()
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_image = netg(noises)
                output = netd(fake_image)
                error_g = criterions(output, true_labels)
                error_g.backward()
                optimize_g.step()

        if (epoch + 1) % opt.save_every == 0:
            fix_fake_image = netg(fix_noises)
            tv.utils.save_image(fix_fake_image.data[:64], "%s/%s.png" % (opt.save_path, epoch), normalize=True)

            torch.save(netd.state_dict(), 'D:\pythonProject9\Img10' + 'netd_{0}.pth'.format(epoch))
            torch.save(netg.state_dict(), 'D:\pythonProject9\Img10' + 'netg_{0}.pth'.format(epoch))


@torch.no_grad()
def generate(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = torch.device("cuda") if opt.gpu else torch.device("cpu")

    netg, netd = NetG(opt).eval(), NetD(opt).eval()

    map_location = lambda storage, loc: storage

    netd.load_state_dict(torch.load('D:\pythonProject9\Img10\\netd_999.pth', map_location=map_location), False)
    netg.load_state_dict(torch.load('D:\pythonProject9\Img10\\netg_999.pth', map_location=map_location), False)
    netd.to(device)
    netg.to(device)

    noise = torch.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std).to(device)
    fake_image = netg(noise)
    score = netd(fake_image).detach()

    indexs = score.topk(opt.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_image.data[ii])
    tv.utils.save_image(torch.stack(result), opt.gen_img, normalize=True, range=(-1, 1))


def main():
    # 训练模型
    train()
    # 生成图片
    generate()


if __name__ == '__main__':
    main()
