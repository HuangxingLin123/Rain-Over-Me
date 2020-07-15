import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler



###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = derainnet(feature_num=16)



    # netG = UNet(n_channels=3, n_classes=3)

    return init_net(net, init_type, init_gain, gpu_ids)




class ms_module0(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch,dila=3):
        super(ms_module0, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=dila, dilation=dila)
            # nn.ReLU(inplace=True)


        self.nolinear = nn.ReLU(inplace=True)

    def forward(self, x):
        la = self.conv1(x)
        res=x-la
        output=self.dilated_conv(res)+la
        output = self.nolinear(output)
        return output


class aggre_module(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch):
        super(aggre_module, self).__init__()
        self.conv1=ms_module0(in_ch=in_ch)
        self.conv2=ms_module0(in_ch=in_ch)
        self.merge=nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        a1=self.conv1(x)
        a2=self.conv2(a1)
        m1=torch.cat((a1, a2), 1)
        output=self.merge(m1)

        return output

class derainnet(nn.Module):
    def __init__(self, feature_num=16):
        super(derainnet, self).__init__()
        self.inc=nn.Sequential(
            nn.Conv2d(3, feature_num, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block1 = aggre_module(feature_num)
        self.block2 = aggre_module(feature_num)
        self.block3 = aggre_module(feature_num)

        self.merge11 = nn.Sequential(
            nn.Conv2d(feature_num * 2, feature_num, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.merge12 = nn.Sequential(
            nn.Conv2d(feature_num * 2, feature_num, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )

        self.outc=nn.Sequential(
            nn.Conv2d(feature_num, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        ##  detail enhancement block
        self.out2=nn.Sequential(
            nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )



    def forward(self, input1):

        fi=self.inc(input1)
        f0 = self.block1(fi)
        f1 = self.block2(f0)
        m01=torch.cat((f0, f1), 1)
        f01 = self.merge11(m01)

        f3=self.block3(f01)
        m23=torch.cat((f01, f3), 1)
        f33=self.merge12(m23)

        res=self.outc(f33)
        output=input1-res

        res2 = self.out2(res)


        output2=input1 - res2

        return res,output,res2,output2





