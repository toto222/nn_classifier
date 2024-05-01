# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:18:40 2024

@author: TWP
"""

from subprocess import run

# for act sigmoid 0.1 0.05 0.001
# for act relu 0.001 0.0005 0.0001
# for act tanh relu 0.005 0.001 0.0005

# hidden  256 128 64
# L2 0.1 0.01 0.001 0
rec_sig=dict()
for lr in [0.1, 0.05, 0.001]:
    for hidden in [256, 128, 64]:
        for L2 in [0.1, 0.01, 0.001, 0]:
            command = f'python main_nn.py --epoch 20\
                --learning_rate {lr}\
                --hidden {hidden}\
                --act_func sigmoid\
                --L2 {L2}'
            
            output = run(command,capture_output=True).stdout
            normal_string = output.decode()            
            # 分割字符串为多行
            lines = normal_string.split('\r\n')            
            # 找到最后一行包含的 val loss 和 accuracy
            last_line = lines[-2]  # 获取倒数第二行
            values = last_line.split()  # 根据空格分割
            val_loss = float(values[-2][5:])  # 获取 val loss
            accuracy = float(values[-1][9:])  # 获取 accuracy
            rec_sig[f'lr_{lr}_{hidden}_{L2}']=val_loss,accuracy
            
rec_relu=dict()
for lr in [0.001, 0.0005, 0.0001]:
    for hidden in [256, 128, 64]:
        for L2 in [0.1, 0.01, 0.001, 0]:
            command = f'python main_nn.py --epoch 20\
                --learning_rate {lr}\
                --hidden {hidden}\
                --act_func relu\
                --L2 {L2}'
            
            output = run(command,capture_output=True).stdout
            normal_string = output.decode()            
            # 分割字符串为多行
            lines = normal_string.split('\r\n')            
            # 找到最后一行包含的 val loss 和 accuracy
            last_line = lines[-2]  # 获取倒数第二行
            values = last_line.split()  # 根据空格分割
            val_loss = float(values[-2][5:])  # 获取 val loss
            accuracy = float(values[-1][9:])  # 获取 accuracy
            rec_relu[f'lr_{lr}_{hidden}_{L2}']=val_loss,accuracy
            
rec_tanh=dict()
for lr in [0.005, 0.001, 0.0005]:
    for hidden in [256, 128, 64]:
        for L2 in [0.1, 0.01, 0.001, 0]:
            command = f'python main_nn.py --epoch 20\
                --learning_rate {lr}\
                --hidden {hidden}\
                --act_func tanh\
                --L2 {L2}'
            
            output = run(command,capture_output=True).stdout
            normal_string = output.decode()            
            # 分割字符串为多行
            lines = normal_string.split('\r\n')            
            # 找到最后一行包含的 val loss 和 accuracy
            last_line = lines[-2]  # 获取倒数第二行
            values = last_line.split()  # 根据空格分割
            val_loss = float(values[-2][5:])  # 获取 val loss
            accuracy = float(values[-1][9:])  # 获取 accuracy
            rec_tanh[f'lr_{lr}_{hidden}_{L2}']=val_loss,accuracy                       
            

            