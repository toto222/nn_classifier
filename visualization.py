# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:53:32 2024

@author: TWP
"""

import matplotlib.pyplot as plt
from NN import Model

filename = 'save/lr_0.001_bsz_128_act_tanh_sche_cos_hidden-layer_256'
model = Model()
model = model.load(filename)
weights = (model.fc1)


weights_image = weights.reshape(256, 28, 28)

# 创建一个 16x16 的子图网格，每个子图显示一个神经元的权重图像
plt.figure(figsize=(16, 16))
for i in range(256):
    plt.subplot(16, 16, i+1)
    plt.imshow(weights_image[i], cmap='gray')
    plt.axis('off')

# plt.suptitle('Visualization of First Layer Weights')
plt.tight_layout()
plt.savefig('feature.pdf',format='pdf')
plt.show()