#this routine will replace all the calculation for current training.
import torch
import torchvision.transforms.functional as F
import numpy as np
class ToTensor9(object):
    """Convert ndarrays in sample to Tensors of 8 channel."""

    def __call__(self, sample):
        #image, landmarks = sample['image'], sample['landmarks']
        image = sample
        # refer the git/lfastai/tcat.py for increase the dimension

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # we need this routine 
        tensorfromimg = F.to_tensor(image)
        return torch.cat((tensorfromimg, tensorfromimg, tensorfromimg), 0)
        # image9 = np.concatenate((image, image, image), 2)
        # image9 = image9.transpose((2, 0, 1))
        # return torch.from_numpy(image9)
        # return {'image': torch.from_numpy(image),
        #         'landmarks': torch.from_numpy(landmarks)}
