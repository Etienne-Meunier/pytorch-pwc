%load_ext autoreload
%autoreload 2

import torch
import numpy as np
import PIL
import matplotlib.pyplot as plt
from utils import estimate
from Network import Network


arguments_strFirst = 'images/first.png'
arguments_strSecond = 'images/second.png'


tenFirst = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
tenSecond = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))


plt.imshow(np.transpose(tenSecond, [1,2,0]))

netNetwork = None
if netNetwork is None :
    print('test')
tenOutput = estimate(tenFirst, tenSecond)

objOutput = open(arguments_strOut, 'wb')

np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objOutput)
np.array([ tenOutput.shape[2], tenOutput.shape[1] ], np.int32).tofile(objOutput)
np.array(tenOutput.np().transpose(1, 2, 0), np.float32).tofile(objOutput)

objOutput.close()
