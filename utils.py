#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
from Network import Network


def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	if str(tenFlow.shape) not in backwarp_tenPartial:
		backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
	tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)

	tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

	tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

	return tenOutput[:, :-1, :, :] * tenMask


def estimate(tenFirst, tenSecond):
	#netNetwork = None

	#if netNetwork is None:
	netNetwork = Network().cuda().eval()
	# end

	assert(tenFirst.shape[1] == tenSecond.shape[1])
	assert(tenFirst.shape[2] == tenSecond.shape[2])

	intWidth = tenFirst.shape[2]
	intHeight = tenFirst.shape[1]

	assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
	tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

	tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tenFlow = 20.0 * torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tenFlow[0, :, :, :].cpu()
