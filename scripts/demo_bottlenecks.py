import torch
import sys
sys.path.append("/Users/teguhsatya/lab/MobileNet/paperplay-mobilenetv2")
from models import bottleneck


def main():
    bn = bottleneck.BottleneckBlock(in_c=32, out_c=16, exp_factor=1, nstrides=1)
    img = torch.rand(1, 32, 224, 224)
    print(bn(img).shape)


if __name__ == '__main__':
    main()