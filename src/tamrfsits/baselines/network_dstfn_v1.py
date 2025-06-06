# This is a file imported from an external repo, so no quality checks here. It  is what it is.
# pylint: skip-file

import torch
import torch.nn.init as init
from torch import nn

"""
File import from the dstfn repository:
https://github.com/andywu456/dstfn/blob/main/DSTFN-pytorch-submitted-no-testdata-20220331.rar
"""


class make_dense(nn.Module):
    def __init__(self, nFeat, growthRate):
        super().__init__()
        self.conv_dense = nn.Sequential(
            nn.Conv2d(nFeat, growthRate, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        out1 = self.conv_dense(x)
        out = torch.cat((x, out1), 1)
        return out


class RDB(nn.Module):
    def __init__(self, nFeat, nDense, growthRate):
        super().__init__()
        nFeat_ = nFeat
        modules = []
        for _ in range(nDense):
            modules.append(make_dense(nFeat_, growthRate))
            nFeat_ += growthRate
            self.dense_layers = nn.Sequential(*modules)
            self.conv_1x1 = nn.Conv2d(
                nFeat_, nFeat, kernel_size=1, padding=0, bias=False
            )

    def forward(self, x):
        out1 = self.conv_1x1(self.dense_layers(x))
        out = torch.add(x, out1)
        return out


class CALayer(nn.Module):
    def __init__(self, nFeat, ratio=16):
        super().__init__()
        self.cal_conv1 = nn.Sequential(
            nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.cal_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cal_max_pool = nn.AdaptiveMaxPool2d(1)
        self.cal_fc1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat // ratio, 1, padding=0, bias=False), nn.ReLU()
        )
        self.cal_fc2 = nn.Sequential(
            nn.Conv2d(nFeat // ratio, nFeat, 1, padding=0, bias=False), nn.Sigmoid()
        )
        self.cal_fc3 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat // ratio, 1, padding=0, bias=False), nn.ReLU()
        )
        self.cal_fc4 = nn.Sequential(
            nn.Conv2d(nFeat // ratio, nFeat, 1, padding=0, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        cal_weight_avg = self.cal_fc2(self.cal_fc1(self.cal_avg_pool(x)))
        cal_weight_max = self.cal_fc4(self.cal_fc3(self.cal_max_pool(x)))
        out_cal_avg = x * cal_weight_avg
        out_cal_max = x * cal_weight_max
        out = self.cal_conv1(torch.cat((out_cal_avg, out_cal_max), 1))
        return out


class SALayer(nn.Module):
    def __init__(self, nFeat):
        super().__init__()
        self.sal_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, groups=nFeat, bias=False),
            nn.ReLU(),
        )
        self.sal_conv1x1 = nn.Sequential(
            nn.Conv2d(nFeat, 1, kernel_size=1, padding=0, bias=False), nn.Sigmoid()
        )
        self.sal_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        sal_weight = self.sal_conv1x1(self.sal_conv1(x))
        out = self.sal_conv2(x * sal_weight)
        return out


class CSA(nn.Module):
    def __init__(self, nFeat):
        super().__init__()
        self.csa_cal = CALayer(nFeat)
        self.csa_sal = SALayer(nFeat)
        self.csa_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        out1 = self.csa_cal(x)
        out2 = self.csa_sal(x)
        out = self.csa_conv1(torch.add(out1, out2))
        return out


class NET1(nn.Module):
    def __init__(self, args):
        super().__init__()
        ncha_s2_hr = args.ncha_s2_hr
        ncha_s2_lr = args.ncha_s2_lr
        nDense = args.nDense
        nFeat = args.nFeat
        growthRate = args.growthRate
        self.conv1_net1 = nn.Sequential(
            nn.Conv2d(ncha_s2_hr, nFeat, kernel_size=3, padding=1, bias=True), nn.ReLU()
        )
        self.conv2_net1 = nn.Sequential(
            nn.Conv2d(ncha_s2_lr, nFeat, kernel_size=3, padding=1, bias=True), nn.ReLU()
        )
        self.conv3_net1 = nn.Sequential(
            nn.Conv2d(nFeat, ncha_s2_lr, kernel_size=3, padding=1, bias=True)
        )
        self.CSA1_net1 = CSA(nFeat)
        self.CSA2_net1 = CSA(nFeat)
        self.CSA3_net1 = CSA(nFeat)
        self.CSA4_net1 = CSA(nFeat)
        self.RDB1_net1 = RDB(nFeat, nDense, growthRate)
        self.RDB2_net1 = RDB(nFeat, nDense, growthRate)
        self.RDB3_net1 = RDB(nFeat, nDense, growthRate)
        self.RDB4_net1 = RDB(nFeat, nDense, growthRate)
        self.conv1_1x1_net1 = nn.Conv2d(
            nFeat * 2, nFeat, kernel_size=1, padding=0, bias=True
        )
        self.conv2_1x1_net1 = nn.Conv2d(
            nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True
        )
        self.upsample_net1 = torch.nn.Upsample(scale_factor=2, mode="bicubic")
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, s2_hr, s2_lr):
        out1 = self.conv1_net1(s2_hr)
        out2 = self.upsample_net1(
            self.RDB1_net1(self.CSA1_net1(self.conv2_net1(s2_lr)))
        )
        out3 = self.conv1_1x1_net1(torch.cat((out1, out2), 1))
        out4 = self.RDB2_net1(self.CSA2_net1(out3))
        out5 = self.RDB3_net1(self.CSA3_net1(out4))
        out6 = self.RDB4_net1(self.CSA4_net1(out5))
        out7 = self.conv2_1x1_net1(torch.cat((out4, out5, out6), 1))
        out8 = torch.add(out7, out2)
        out = self.conv3_net1(out8)
        return out


class SLFNT1(nn.Module):
    def __init__(self, args):
        super().__init__()
        ncha_s2 = args.ncha_s2
        ncha_l8 = args.ncha_l8
        nDense = args.nDense
        nFeat = args.nFeat
        growthRate = args.growthRate
        self.conv1_slfnt1 = nn.Sequential(
            nn.Conv2d(ncha_s2, nFeat, kernel_size=3, padding=1, bias=True), nn.ReLU()
        )
        self.conv2_slfnt1 = nn.Sequential(
            nn.Conv2d(ncha_l8, nFeat, kernel_size=3, padding=1, bias=True), nn.ReLU()
        )
        self.conv3_slfnt1 = nn.Sequential(
            nn.Conv2d(nFeat, ncha_l8, kernel_size=3, padding=1, bias=True)
        )
        self.CSA1_slfnt1 = CSA(nFeat)
        self.CSA2_slfnt1 = CSA(nFeat)
        self.CSA3_slfnt1 = CSA(nFeat)
        self.CSA4_slfnt1 = CSA(nFeat)
        self.RDB1_slfnt1 = RDB(nFeat, nDense, growthRate)
        self.RDB2_slfnt1 = RDB(nFeat, nDense, growthRate)
        self.RDB3_slfnt1 = RDB(nFeat, nDense, growthRate)
        self.RDB4_slfnt1 = RDB(nFeat, nDense, growthRate)
        self.conv1_1x1_slfnt1 = nn.Conv2d(
            nFeat * 2, nFeat, kernel_size=1, padding=0, bias=True
        )
        self.conv2_1x1_slfnt1 = nn.Conv2d(
            nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True
        )
        self.upsample1_slfnt1 = torch.nn.Upsample(scale_factor=3, mode="bicubic")
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, s2, l8):
        s2_fea = self.conv1_slfnt1(s2)
        l8_fea = self.upsample1_slfnt1(
            self.RDB1_slfnt1(self.CSA1_slfnt1(self.conv2_slfnt1(l8)))
        )
        out1 = self.conv1_1x1_slfnt1(torch.cat((s2_fea, l8_fea), 1))
        out2 = self.RDB2_slfnt1(self.CSA2_slfnt1(out1))
        out3 = self.RDB3_slfnt1(self.CSA3_slfnt1(out2))
        out4 = self.RDB4_slfnt1(self.CSA4_slfnt1(out3))
        out5 = self.conv2_1x1_slfnt1(torch.cat((out2, out3, out4), 1))
        out6 = torch.add(out5, l8_fea)
        out = self.conv3_slfnt1(out6)
        return out


class NET2(nn.Module):
    def __init__(self, args):
        super().__init__()
        ncha_s2 = args.ncha_s2
        ncha_l8 = args.ncha_l8
        ncha_l8_pan = args.ncha_l8_pan
        nDense = args.nDense
        nFeat = args.nFeat
        growthRate = args.growthRate
        self.conv1_net2 = nn.Sequential(
            nn.Conv2d(ncha_s2, nFeat, kernel_size=3, padding=1, bias=True), nn.ReLU()
        )
        self.conv2_net2 = nn.Sequential(
            nn.Conv2d(ncha_l8, nFeat, kernel_size=3, padding=1, bias=True), nn.ReLU()
        )
        self.conv3_net2 = nn.Sequential(
            nn.Conv2d(ncha_l8_pan, nFeat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )
        self.conv4_net2 = nn.Sequential(
            nn.Conv2d(nFeat, ncha_l8, kernel_size=3, padding=1, bias=True)
        )
        self.CSA1_net2 = CSA(nFeat)
        self.CSA2_net2 = CSA(nFeat)
        self.CSA3_net2 = CSA(nFeat)
        self.CSA4_net2 = CSA(nFeat)
        self.RDB1_net2 = RDB(nFeat, nDense, growthRate)
        self.RDB2_net2 = RDB(nFeat, nDense, growthRate)
        self.RDB3_net2 = RDB(nFeat, nDense, growthRate)
        self.RDB4_net2 = RDB(nFeat, nDense, growthRate)
        self.conv1_1x1_net2 = nn.Conv2d(
            nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True
        )
        self.conv2_1x1_net2 = nn.Conv2d(
            nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True
        )
        self.upsample1_net2 = torch.nn.Upsample(scale_factor=3, mode="bicubic")
        self.upsample2_net2 = torch.nn.Upsample(scale_factor=1.5, mode="bicubic")
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, s2, l8, l8_pan):
        s2_fea = self.conv1_net2(s2)
        l8_fea = self.upsample1_net2(
            self.RDB1_net2(self.CSA1_net2(self.conv2_net2(l8)))
        )
        l8_pan_fea = self.upsample2_net2(self.conv3_net2(l8_pan))
        out1 = self.conv1_1x1_net2(torch.cat((s2_fea, l8_fea, l8_pan_fea), 1))
        out2 = self.RDB2_net2(self.CSA2_net2(out1))
        out3 = self.RDB3_net2(self.CSA3_net2(out2))
        out4 = self.RDB4_net2(self.CSA4_net2(out3))
        out5 = self.conv2_1x1_net2(torch.cat((out2, out3, out4), 1))
        out6 = torch.add(out5, l8_fea)
        out = self.conv4_net2(out6)
        return out
