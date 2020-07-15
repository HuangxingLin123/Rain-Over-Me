import torch
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import pytorch_ssim
import numpy as np



class DerainModel(BaseModel):
    def name(self):
        return 'DerainModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):


        parser.set_defaults(norm='batch', netG='unet_256')
        parser.set_defaults(dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser



    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        if self.isTrain:

            self.visual_names = [ 'Y_hat','Y_tilde','Y_b','X','C','rain_D','C_hat','C_tilde','res_map','res2_map']
        else:
            self.visual_names = ['Y_hat']

        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,id=2)
        self.netG = networks.define_G(opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            # define loss functions

            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.ssim_loss = pytorch_ssim.SSIM()

            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)


    def set_input(self, input,epoch,iteration):
        if self.isTrain:
            self.X = input['X'].to(self.device)
            self.Y_b = input['Y_b'].to(self.device)

            self.epoch = epoch
            self.iteration = iteration

            self.C = input['C'].to(self.device)

            self.image_paths = input['X_paths']
        else:
            self.X = input['X'].to(self.device)
            self.image_paths = input['X_paths']


    def forward(self):
        if self.isTrain:
            self.res, self.Y_hat, self.res2, self.Y_tilde = self.netG(self.X)
            self.res_map = self.res - torch.min(self.res)
            self.res_map = self.res_map / torch.max(self.res_map)

            self.res2_map = self.res2 - torch.min(self.res2)
            self.res2_map = self.res2_map / torch.max(self.res2_map)

            self.res3 = self.res2.detach()

            self.rain_D = self.C + self.res3
            self.rain_D[self.rain_D > 1.0] = 1.0
            self.rain_D[self.rain_D < 0] = 0

            res3, self.C_hat, res4, self.C_tilde = self.netG(self.rain_D)
        else:
            self.res, self.Y_hat, self.res2, self.Y_tilde = self.netG(self.X)



    def backward_G2(self):
        self.loss_soft_SSIM = (1 - self.ssim_loss(self.Y_tilde, self.Y_b))*(1000/(self.iteration+1000))

        p = (self.iteration/1000)
        if p > 10:
            p = 10

        self.loss_hard_SSIM = (1 - self.ssim_loss(self.C_hat, self.C)) * p

        self.loss_G2 = self.loss_soft_SSIM  + self.loss_hard_SSIM



        self.loss_G2.backward()




    def optimize_parameters(self):
        self.forward()
        # update D

        self.optimizer_G.zero_grad()
        self.backward_G2()
        self.optimizer_G.step()