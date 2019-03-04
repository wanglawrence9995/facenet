#this routine will replace all the calculation for current training.
import torch
import torchvision.transforms.functional as F
import numpy as np
from skimage import io, transform
class ToTensor9(object):
    """Convert ndarrays in sample to Tensors of 9 channel."""

    def __call__(self, sample):
        #image, landmarks = sample['image'], sample['landmarks']
        image = sample
        # refer the git/lfastai/tcat.py for increase the dimension

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # we need this routine 
        tensorfromimg = F.to_tensor(image)
        return torch.cat((tensorfromimg, tensorfromimg, tensorfromimg), 0)  # 0 means channels dimention 9 is back to the first one
        # image9 = np.concatenate((image, image, image), 2)
        # image9 = image9.transpose((2, 0, 1))
        # return torch.from_numpy(image9)
        # return {'image': torch.from_numpy(image),
        #         'landmarks': torch.from_numpy(landmarks)}

class FaceDataToTensor(object):
    """ facedata, already 9 channel, so still use F.to_tensor three time."""
    def __call__(self, sample):
        # the image is rescaled already to desired 182 in our case
        # t1 = transform.resize(sample[...,0:3], (182, 182))
        # t1.astype(np.uint8)
        t1 =F.to_tensor(sample[...,0:3])
        
        # t2 = transform.resize(sample[...,3:6], (182, 182))
        # t2.astype(np.uint8)
        t2 =F.to_tensor(sample[...,3:6])

        # t3 = transform.resize(sample[..., 6:], (182, 182))
        # t3.astype(np.uint8)
        t3 =F.to_tensor(sample[..., 6:])
        return torch.cat((t1,t2,t3),0)
