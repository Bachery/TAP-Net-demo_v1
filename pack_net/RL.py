# -*- coding: utf-8 -*-
import math
import random
import sys
import os
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy

import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# sys.path.append('../')
# import tools


def is_stable_2d(support, obj_left, obj_width):
    '''
    check if the obj is stable
    ---
    params:
    ---
        support: obj_width x 1 array / list / tnesor, the container data under obj
        obj_left: a float number, left index of obj's position (x of obj.position)
        obj_width: a float number, width of obj
    return:
    ---
        is_stable: bool, stable or not
    '''
    object_center = obj_left + obj_width/2
    left_index = obj_left
    right_index = obj_left + obj_width
    for left in support:
        if left <= 0:
            left_index += 1
        else:
            break
    for right in reversed(support):
        if right <= 0:
            right_index -= 1
        else:
            break
    if left_index + 1 == right_index and obj_width == 1:
        return True
    # 如果物体中心线不在范围内，不稳定
    if object_center <= left_index or object_center >= right_index:
        return False
    return True

class PackEngine:
    def __init__(self, container_width, container_height, max_blocks_num, is_train=True):
        self.container_width = container_width
        self.container_height = container_height
        self.height_map = np.zeros(container_width)

        self.container = np.zeros((container_width, container_height))

        # actions are triggered by letters
        self.value_action_map = [ i for i in range(container_width) ]
        self.nb_actions = container_width

        # for running the engine
        self.score = 0
        self.time = 0
        self.num_blocks = 0
        self.max_blocks_num = max_blocks_num
        # clear after initializing
        self.positions = []
        self.blocks = []
        self.stable = [False for i in range(max_blocks_num)]
        self.clear()
        self.is_train = is_train
        self.reward = 0


    def step(self, action, block, draw_img=False):
        '''
        Parameters:
        ---
            action: 1D int, place position (x)
            block: 2D float, the block (w, h)
        Returns:
        ---
            state, reward, done, pos
        '''
        # if the place position can place the block, punish it
        block = block.int()

        # if self.is_train == False:
        while action + block[0] > self.container_width:
            action -= 1
        
        if action + block[0] > self.container_width:
            reward = -1
            pos = None
        else:

            self.container, self.height_map, self.stable, \
                box_size, valid_size, empty_size, pos, self.num_blocks = add_block(block.cpu().numpy(), action, self.container, self.height_map, self.stable, self.num_blocks)
            
            # # continus space
            # max_height = int(self.height_map.max())
            # min_height = int(self.height_map.min())

            # total_box = self.container_width * (max_height - min_height)

            # # for w in range(self.container_width):
            # each_column = self.container[:, min_height:max_height]
            # total_empty = (each_column==0).sum()

            if box_size == 0:
                C = 0
            else:
                C = valid_size / box_size
            
            if valid_size + empty_size == 0:
                P = 0
            else:
                P = valid_size / (valid_size + empty_size)
            
            if self.num_blocks == 0:
                S = 0
            else:
                S = np.sum(self.stable) / self.num_blocks
            
            reward = (C+P+S) / 3
            
            # if reward < 0.7:
            # reward -= 1
            # if total_box == 0:
            #     reward = 1
            # else:
            #     reward = (total_box - total_empty) / total_box

            if block[0] != 0:
                self.blocks.append( block.cpu().numpy() )
                self.positions.append( pos )

            # tools.draw_container_2d(self.blocks, self.positions, [self.container_width, 15], 
            #         # save_title='C: %.3f P: %.3f S: %.3f' % (C, P, S),
            #         save_title='box: %d empty: %d' % (total_box, total_empty),
            #         save_name='p_%s' % (note) )
            # input('Enter')                    

        # Update time and reward
        self.time += 1

        done = False
        if self.time == self.max_blocks_num:
            # if draw_img:
                # tools.draw_container_2d(self.blocks, self.positions, [self.container_width, 15], 
                #     # save_title='C: %.3f P: %.3f S: %.3f' % (C, P, S),
                #     save_name='p_%s' % (note) )
                
            self.clear()
            done = True
            
        state = np.copy(self.height_map)

        return state, reward, done, pos



    def clear(self):
        self.time = 0
        self.score = 0
        self.num_blocks = 0
        # self.reward = 0
        self.blocks = []
        self.positions = []
        self.height_map = np.zeros_like(self.height_map)
        self.container = np.zeros((self.container_width, self.container_height))
        self.stable = [False for i in range(self.max_blocks_num)]
        
        return self.height_map

class PackDataset(Dataset):
    def __init__(self, data_file, blocks_num, data_size):
        '''
        Data initialization
        ----
        params
        ----
            
        '''
        super(PackDataset, self).__init__()

        gt_blocks = np.loadtxt(data_file + 'gt_blocks.txt').astype('float32')
        gt_positions = np.loadtxt(data_file + 'gt_pos.txt').astype('float32')

        gt_positions = torch.from_numpy(gt_positions)
        gt_positions = gt_positions.view(data_size, -1, blocks_num)

        gt_blocks = torch.from_numpy(gt_blocks)
        gt_blocks = gt_blocks.view(data_size, -1, blocks_num)
        
        order = [blocks_num - i-1 for i in range(blocks_num)  ]

        # np.random.shuffle(order)

        # inverse order
        gt_blocks = gt_blocks[:,:,order]
        gt_positions = gt_positions[:,:,order]

        # gt_blocks = gt_blocks.transpose(2, 1)
        # gt_positions = gt_positions.transpose(2, 1)
        
        self.gt_blocks = gt_blocks
        self.gt_positions = gt_positions
        
        self.data_size = data_size

        print(gt_blocks.shape)
        print(gt_positions.shape)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return (self.gt_blocks[idx], self.gt_positions[idx] )


class PackDataset_rand(Dataset):
    def __init__(self, data_file, blocks_num, data_size):
        '''
        Data initialization
        ----
        params
        ----
            
        '''
        super(PackDataset_rand, self).__init__()

        gt_positions = np.loadtxt(data_file + 'pos.txt').astype('float32')
        gt_positions = torch.from_numpy(gt_positions)
        gt_positions = gt_positions.view(data_size, -1, blocks_num)

        all_blocks = np.loadtxt( data_file + 'blocks.txt').astype('float32')
        block_dim = 2
        rotate_types = np.math.factorial(block_dim)

        data_size = int(len(all_blocks) / rotate_types)
        all_blocks = all_blocks.reshape( data_size, -1, block_dim, blocks_num)
        all_blocks = all_blocks.transpose(0, 1, 3, 2)
        all_blocks = all_blocks.reshape( data_size, -1, block_dim )
        all_blocks = all_blocks.transpose(0,2,1)
        all_blocks = torch.from_numpy(all_blocks)


        gt_blocks = all_blocks[:,:,:blocks_num]
        
        self.gt_blocks = gt_blocks
        self.gt_positions = gt_positions
        
        self.data_size = data_size

        print(gt_blocks.shape)
        print(gt_positions.shape)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return (self.gt_blocks[idx], self.gt_positions[idx] )


class PackDataset_mix(Dataset):
    def __init__(self, data_file_1, data_file_2, blocks_num, data_size):
        '''
        Data initialization
        ----
        params
        ----
            
        '''
        super(PackDataset_mix, self).__init__()

        self.data_size = data_size

        gt_positions = np.loadtxt(data_file_1 + 'pos.txt').astype('float32')
        gt_positions = torch.from_numpy(gt_positions)
        gt_positions = gt_positions.view(data_size, -1, blocks_num)

        if True:
            all_blocks = np.loadtxt( data_file_1 + 'blocks.txt').astype('float32')
            block_dim = 2
            rotate_types = np.math.factorial(block_dim)

            data_size = int(len(all_blocks) / rotate_types)
            all_blocks = all_blocks.reshape( data_size, -1, block_dim, blocks_num)
            all_blocks = all_blocks.transpose(0, 1, 3, 2)
            all_blocks = all_blocks.reshape( data_size, -1, block_dim )
            all_blocks = all_blocks.transpose(0,2,1)
            all_blocks = torch.from_numpy(all_blocks)

            gt_blocks = all_blocks[:,:,:blocks_num]

            # 2
            all_blocks = np.loadtxt( data_file_2 + 'blocks.txt').astype('float32')
            data_size = int(len(all_blocks) / rotate_types)
            all_blocks = all_blocks.reshape( data_size, -1, block_dim, blocks_num)
            all_blocks = all_blocks.transpose(0, 1, 3, 2)
            all_blocks = all_blocks.reshape( data_size, -1, block_dim )
            all_blocks = all_blocks.transpose(0,2,1)
            all_blocks = torch.from_numpy(all_blocks)
        
            gt_blocks[:int(data_size/4),:,:] = all_blocks[:int(data_size/4),:,:blocks_num]


        self.gt_blocks = gt_blocks
        self.gt_positions = gt_positions
        
        print('Mix')
        print(gt_blocks.shape)
        print(gt_positions.shape)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return (self.gt_blocks[idx], self.gt_positions[idx] )


def load_data(data_file, blocks_num, num_samples):
    '''
    Data initialization
    ----
    params
    ----
        xxx
    Returns
    ---
        gt_blocks: num-samples x block-dim x blocks-num
        gt_position: num-samples x block-dim x blocks-num
    '''
    
    gt_blocks = np.loadtxt(data_file + 'gt_blocks.txt').astype('float32')
    gt_positions = np.loadtxt(data_file + 'gt_pos.txt').astype('float32')

    gt_positions = torch.from_numpy(gt_positions)
    gt_positions = gt_positions.view(num_samples, -1, blocks_num)

    gt_blocks = torch.from_numpy(gt_blocks)
    gt_blocks = gt_blocks.view(num_samples, -1, blocks_num)
    
    order = [blocks_num - i-1 for i in range(blocks_num)  ]

    # inverse order
    gt_blocks = gt_blocks[:,:,order]
    gt_positions = gt_positions[:,:,order]

    if use_cuda:
        gt_positions = gt_positions.cuda().detach()
        gt_blocks = gt_blocks.cuda().detach()

    return gt_blocks, gt_positions


def load_data_rand(data_file, blocks_num, num_samples):
    '''
    Data initialization
    ----
    params
    ----
        xxx
    Returns
    ---
        gt_blocks: num-samples x block-dim x blocks-num
        gt_position: num-samples x block-dim x blocks-num
    '''
    
    gt_positions = np.loadtxt(data_file + 'pos.txt').astype('float32')

    gt_positions = torch.from_numpy(gt_positions)
    gt_positions = gt_positions.view(num_samples, -1, blocks_num)

    all_blocks = np.loadtxt( data_file + 'blocks.txt').astype('int')

    block_dim = 2
    rotate_types = np.math.factorial(block_dim)

    data_size = int(len(all_blocks) / rotate_types)
    all_blocks = all_blocks.reshape( data_size, -1, block_dim, blocks_num)
    all_blocks = all_blocks.transpose(0, 1, 3, 2)
    all_blocks = all_blocks.reshape( data_size, -1, block_dim )

    gt_blocks = all_blocks[:,:blocks_num]

    gt_blocks = torch.from_numpy( gt_blocks.astype('float32') ).transpose(2,1).cuda().detach()
    print(gt_blocks.shape)
    return gt_blocks, gt_positions


class DQN(nn.Module):

    def __init__(self, output_size, is_diff_height):
        super(DQN, self).__init__()      
        if is_diff_height:  
            self.conv_height_map = nn.Conv1d(1, 128, kernel_size=1)
            self.conv_block = nn.Conv1d(1, 128, kernel_size=1)
            self.bn1 = nn.BatchNorm1d(128)
            self.conv2 = nn.Conv1d(128, 256, kernel_size=1)
            self.bn2 = nn.BatchNorm1d(256)
            self.conv3 = nn.Conv1d(256, 256, kernel_size=1)
            self.bn3 = nn.BatchNorm1d(256)
            self.lin1 = nn.Linear(1536, 512)
            self.head = nn.Linear(512, output_size)
        else:
            self.conv_height_map = nn.Conv1d(1, 128, kernel_size=1)
            self.conv_block = nn.Conv1d(1, 128, kernel_size=1)
            self.bn1 = nn.BatchNorm1d(128)
            self.conv2 = nn.Conv1d(128, 256, kernel_size=1)
            self.bn2 = nn.BatchNorm1d(256)
            self.conv3 = nn.Conv1d(256, 256, kernel_size=1)
            self.bn3 = nn.BatchNorm1d(256)
            self.lin1 = nn.Linear(1792, 512)
            self.head = nn.Linear(512, output_size)
        
        self.is_diff_height = is_diff_height

    def forward(self, height_map, block):
        '''
        height_map: batch_size x 1 x container_width
        block: batch_size x 1 x 2
        '''
        # if self.is_zero_height:
        #     container_width = height_map.shape[-1]
        #     height_map -= height_map.min(-1)[0].unsqueeze(-1).repeat(1, 1, container_width)
        if self.is_diff_height:
            height_map = height_map[:,:,:-1]
            # height_map = height_map.transpose(2, 1)
        encode_height_map = self.conv_height_map(height_map)
        encode_block = self.conv_block(block)

        output = F.relu(self.bn1( torch.cat((encode_height_map, encode_block), dim=-1) ))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.lin1(output.view(output.size(0), -1)))
        # return self.head(output.view(output.size(0), -1))
        return F.softmax(self.head(output.view(output.size(0), -1)), dim=1)


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, block_input_size, block_hidden_size, container_width, container_height, heightmap_type, max_blocks_num):
        super(Critic, self).__init__()        

        self.encoder = nn.GRU(block_hidden_size, block_hidden_size, batch_first=True)
        
        self.block_conv = nn.Conv1d(block_input_size, block_hidden_size, kernel_size=1)
        

        self.fc = nn.Sequential(
            nn.Linear(block_hidden_size, block_hidden_size),
            nn.ReLU(),
            nn.Linear(block_hidden_size, 1),
        )
        self.dropout = nn.Dropout(p=0.1)

        self.container_width = container_width
        self.container_height = container_height
        self.max_blocks_num = max_blocks_num

        self.heightmap_type = heightmap_type 
        
    def forward(self, blocks):
        '''
        block: batch_size x block-dim x blocks-num
        '''
        use_cuda = blocks.is_cuda
        batch_size = blocks.shape[0]
        
        block_vec = self.block_conv( blocks )
        encode_rnn_out, encoder_last_hh = self.encoder( block_vec.transpose(2, 1), None )
        encoder_last_hh = self.dropout(encoder_last_hh)

        encoder_last_hh = encoder_last_hh[0]

        output = self.fc(encoder_last_hh)

        return output

class PackNet(object):
    def __init__(self, container_width, container_height, heightmap_type, max_blocks_num=10):
        super(PackNet, self).__init__()        

        self.container_width = container_width
        self.container_height = container_height
        self.max_blocks_num = max_blocks_num
        self.heightmap_type = heightmap_type
        
        self.net = DQN(container_width, heightmap_type == 'diff')
        # self.net = DQN(container_width, False)

    def forward(self, blocks, blocks_num):
        '''
        blocks: batch_size x block-dim x blocks-num
        '''
        use_cuda = blocks.is_cuda
        batch_size = blocks.shape[0]
        engines = [ PackEngine(self.container_width, self.container_height, self.max_blocks_num, False) for i in range(batch_size) ]

        height_map = np.zeros((batch_size, 1, self.container_width)).astype('float32')

        blocks = blocks.transpose(2, 1)

        positions = np.zeros( blocks.shape )

        height_map = np.zeros((batch_size, self.container_width, 1)).astype('float32')
        
        rw = np.zeros( batch_size )
        hit_porb_log = []

        last_hh = None

        for block_index in range(blocks_num):
            if self.heightmap_type == 'diff':
                tmp = copy.deepcopy(height_map)
                tmp[:, :, :-1] = tmp[:,:, 1:]
                tmp[:, :, -1] = height_map[:,:, -1]
                tmp = tmp - height_map
                # tmp = tmp[:,:, :-1]
              
                hit_map = self.net( torch.from_numpy(tmp).cuda().transpose(2, 1), blocks[:, block_index:block_index+1] )

            else:
                hit_map = self.net( torch.from_numpy(height_map).cuda().transpose(2, 1), blocks[:, block_index:block_index+1] )
                
            # if self.net.training:
            #     m = torch.distributions.Categorical(hit_map)
            #     ptr = m.sample()
            #     # while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
            #     # ptr = m.sample()
            #     logp = m.log_prob(ptr)
            # else:
            #     prob, ptr = torch.max(hit_map, 1)  # Greedy
            #     logp = prob.log()

            # ptr = ptr.float()

            hit_porb_log.append( torch.log(hit_map.max(1)[0]).unsqueeze(1) )
            # hit_porb_log.append( logp.unsqueeze(1) )


            for batch_index in range(batch_size):
                block = blocks[batch_index][block_index]
                
                action = hit_map[batch_index].argmax()
                # action = ptr[batch_index]

                _, rw[batch_index], _, pos = engines[batch_index].step( action, block )

                positions[batch_index][block_index] = pos

                # for index in range(self.container_width):
                hm = copy.deepcopy(engines[batch_index].height_map)
                if self.heightmap_type == 'zero':
                    hm = hm - np.min(hm)
                # height_map  batch x width x 1
                height_map[batch_index][:,0] = hm

        reward = torch.from_numpy(rw.astype('float32'))
        if use_cuda:
            reward = reward.cuda()

        # hit_porb_log = torch.from_numpy(hit_porb_log.astype('float32')).cuda()
        hit_porb_log = torch.cat(hit_porb_log, dim=1)
        return positions, hit_porb_log, -reward



def add_block(block, pos_x, container, height_map, stable, current_blocks_num, is_train=True):
    '''
    Parameters:
    ---
        block: (w, h)
        pos-x: int
        container: 2d array
        height-map: 1d array (container-w)
        stable: 1d array, store the stable state  
        current_blocks_num
        is_train
    Returns:
    ---
        container
        height-map
        stable
        box-size, valid-size, empty-size,
        (pos-x, pos-z)
        current_blocks_num
    '''
    
    container_width, container_height = container.shape
    block_width, block_height = block
    
    block_width = int(block_width)
    block_height = int(block_height)
    pos_x = int(pos_x)
    
    if is_train == False:
        while pos_x + block_width > container_width:
            pos_x -= 1

    pos_z = int(height_map[pos_x:pos_x+block_width].max())
    

    block_id = current_blocks_num + 1
    block_id = int(block_id)


    if pos_x + block_width <= container_width:
        support = container[pos_x:pos_x+block_width, pos_z-1]
        if pos_z == 0:
            stable[block_id - 1] = True
        else:
            stable[block_id - 1] = is_stable_2d(support, pos_x, block_width)
        
        container[ pos_x:pos_x+block_width, pos_z:pos_z+block_height ] = block_id

        under_block = container[pos_x:pos_x+block_width, :pos_z]
        container[pos_x:pos_x+block_width, :pos_z][ under_block == 0 ] = -1

        height_map[pos_x:pos_x+block_width] = height_map[pos_x:pos_x+block_width].max() + block_height

    else:
        pos_x = container_width
        stable[block_id - 1] = False

    if block_width != 0:
        current_blocks_num += 1

    # C P S
    box_size = (height_map.max() * container_width)
    valid_size = (container >= 1).sum()
    py_size = (container != 0).sum()
    empty_size = py_size - valid_size

    return container, height_map, stable, box_size, valid_size, empty_size, (pos_x, pos_z), current_blocks_num


def calc_positions(net, blocks, container_size, is_train=True):
    """
    Parameters:
    ---
        net: net
        blocks: batch-size x block-dim x blocks-num
        container-size: 2d array/list
    """
    # if gpu is to be used
    use_cuda = blocks.is_cuda


    batch_size = blocks.shape[0]
    blocks_num = blocks.shape[-1]

    net.net.eval()
    net_positions, _, rewards = net.forward(blocks, blocks_num)
    
    container_width, container_height = container_size

    # batch_size x blocks_num x block_dim
    blocks = blocks.transpose(2, 1)

    containers = [np.zeros(container_size) for i in range(batch_size)]
    height_maps = [np.zeros(container_width) for i in range(batch_size)]
    stables = [np.zeros(blocks_num) for i in range(batch_size)]
    current_blocks_nums = np.zeros(batch_size)
    positions = np.zeros((batch_size, blocks_num, 2))
    
    box_size = np.zeros(batch_size)
    empty_size = np.zeros(batch_size)
    packing_height = np.zeros(batch_size)
    ratio = np.zeros(batch_size)
    stable_num = np.zeros(batch_size)
    valid_size = np.zeros(batch_size)

    for block_index in range(blocks_num):

        block = blocks[:, block_index, :]

        action = net_positions[:, block_index,0]

        for batch_index in range(batch_size):
            containers[batch_index], height_maps[batch_index], stables[batch_index], \
                box_size[batch_index], valid_size[batch_index], empty_size[batch_index], \
                positions[batch_index][block_index], current_blocks_nums[batch_index] = \
                    add_block(blocks[batch_index][block_index], action[batch_index], \
                        containers[batch_index], height_maps[batch_index], \
                            stables[batch_index], current_blocks_nums[batch_index], False )


    for batch_index in range(batch_size):
        packing_height[batch_index] = height_maps[batch_index].max()
        stable_num[batch_index] = stables[batch_index].sum()
        
        
        # tools.draw_container_2d( blocks[batch_index], positions[batch_index], container_size, save_name='gt')

        # tools.draw_container_2d( blocks[batch_index], positions[batch_index], container_size, \
        #     save_title='C %.3f   S %.3f   P %.3f' % (C[batch_index], S[batch_index], P[batch_index]), \
        #         save_name='./img/gt_%d' % batch_index)
        # break
    C = valid_size / box_size
    P = valid_size / (valid_size + empty_size)
    S = stable_num / blocks_num


    for batch_index in range(batch_size):
        if batch_index > 3:
            break
        # tools.draw_container_2d( blocks[batch_index], positions[batch_index], container_size, \
        #     save_title='C %.3f   S %.3f   P %.3f' % (C[batch_index], S[batch_index], P[batch_index]), \
        #         save_name='./img/gt_%d' % batch_index)


    # Accumulate reward
    ratio = (C+P+S) / 3

    # import IPython
    # IPython.embed()


    return ratio, positions, box_size, valid_size, box_size, empty_size, stable_num, packing_height, containers, stables 


def train(actor, critic, train_size, valid_size, blocks_num, batch_size, epoch_num, learning_rate, save_dir, use_cuda, net_type, is_train=False):
    
    
    valid_data_file = './data/gt_2d/pack-valid-%d-%d-5-1-5/' % (blocks_num, valid_size)
    # valid_data_file = '../data/rand_2d/pack-valid-%d-%d-7-1-5/' % (blocks_num, valid_size)


    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        os.makedirs(save_dir + '/img')

    actor_optim = optim.Adam(actor.net.parameters(), lr=learning_rate)
    critic_optim = optim.Adam(critic.parameters(), lr=learning_rate)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(actor_optim, 'min',factor=0.8, patience=1000, verbose=True, min_lr=1e-6)
    
    if net_type == 'RL_mix':
        train_data_file_1 = '../data/rand_2d/pack-train-%d-%d-7-1-5/' % (blocks_num, train_size)
        train_data_file_2 = '../data/gt_2d/pack-train-%d-128000-7-1-5/' % (blocks_num)
        train_data = PackDataset_mix( train_data_file_1, train_data_file_2, blocks_num, train_size )
    elif net_type == 'RL_gt':
        train_data_file = './data/gt_2d/pack-train-%d-%d-5-1-5/' % (blocks_num, train_size)
        train_data = PackDataset( train_data_file, blocks_num, train_size )
    else:
        train_data_file = '../data/rand_2d/pack-train-%d-%d-7-1-5/' % (blocks_num, train_size)
        train_data = PackDataset_rand( train_data_file, blocks_num, train_size )

    valid_data = PackDataset( valid_data_file, blocks_num, valid_size )

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, valid_size, shuffle=False, num_workers=0)

    log_step = int(tarin_size / batch_size)
    if log_step > 100:
        log_step = int(100)
    if log_step == 0:
        log_step = int(1)
    
    my_losses = []
    my_rewards = []
    my_critic_losses = []
    my_critic_rewards = []

    times = []
    best_valid = 0


    for epoch in range(epoch_num):
        epoch_start = time.time()
        start = epoch_start

        losses, rewards, critic_rewards, critic_losses = [], [], [], []
        
        for batch_idx, batch in enumerate(train_loader):
            if not is_train:
                break

            # blocks, positions, height_maps, gt_pos_maps = batch
            blocks, _ = batch
            
            if use_cuda:
                blocks = blocks.cuda()#.detach()
            
            positions, hit_porb_log, reward = actor.forward(blocks, blocks_num)

            critic_est = critic(blocks).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * hit_porb_log.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
            # scheduler.step(actor_loss)

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            critic_losses.append(torch.mean(critic_loss.detach()).item())

            if (batch_idx + 1) % log_step == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_reward = np.mean(rewards[-log_step:])
                mean_loss = np.mean(losses[-log_step:])
                mean_critic_reward = np.mean(critic_rewards[-log_step:])
                mean_critic_loss = np.mean(critic_losses[-log_step:])
                
                my_rewards.append(mean_reward)
                my_losses.append(mean_loss)
                my_critic_rewards.append(mean_critic_reward)
                my_critic_losses.append(mean_critic_loss)

                print('    Epoch %d  Batch %d/%d, reward: %2.4f, actor loss: %2.4f, critic loss: %2.4f, took: %2.4fs' %
                      (epoch, batch_idx, len(train_loader), mean_reward, mean_loss, mean_critic_loss, times[-1] ))

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.net.state_dict(), save_path)
        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        mean_loss = np.mean(my_losses)
        # valid network
        with torch.no_grad():
            for batch in valid_loader:
                blocks, _ = batch
                if use_cuda:
                    blocks = blocks.cuda()#.detach()

            positions, _, valid_reward = actor.forward(blocks, blocks_num)

            blks = blocks[0].cpu().numpy()
            blks = blks.transpose()
            positions = positions[0]

            # tools.draw_container_2d(blks, positions, container_size, save_name=save_dir + '/pack')

            mean_valid = torch.mean(reward.detach()).item()

        if mean_valid > best_valid:
            best_valid = mean_valid
            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.net.state_dict(), save_path)
            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        print('Epoch %d,  mean epoch valid: %2.4f  | loss: %2.4f, took: %2.4fs '\
              '(%2.4fs / %d batches)' % \
              (epoch, mean_valid, mean_loss, time.time() - epoch_start,
              np.mean(times), log_step  ))

        plt.close('all')
        plt.title('Reward')
        plt.plot(range(len(my_rewards)), np.abs(my_rewards), '-')
        plt.savefig(save_dir + '/img/rewards.png' , bbox_inches='tight', dpi=400)

        plt.close('all')
        plt.title('Critic Reward')
        plt.plot(range(len(my_critic_rewards)), np.abs(my_critic_rewards), '-')
        plt.savefig(save_dir + '/img/critic_rewards.png' , bbox_inches='tight', dpi=400)


    plt.close('all')
    plt.title('Actor loss')
    plt.plot(range(len(my_losses)), my_losses, '-')
    plt.savefig(save_dir + '/img/actor.png' , bbox_inches='tight', dpi=400)

    plt.close('all')
    plt.title('Critic Loss')
    plt.plot(range(len(my_critic_losses)), my_losses, '-')
    plt.savefig(save_dir + '/img/actor.png' , bbox_inches='tight', dpi=400)


    np.savetxt(save_dir + '/acotr_reward.txt', my_rewards)
    np.savetxt(save_dir + '/acotr_loss.txt', my_losses)
    np.savetxt(save_dir + '/critic_loss.txt', my_critic_rewards)
    np.savetxt(save_dir + '/critic_loss.txt', my_critic_losses)
    


def valid( net, num_samples, blocks_num, save_dir):
    
    # blocks, positions = load_data('./data/gt_2d/pack-valid-%d-%d-5-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)
    blocks, _ = load_data_rand('../data/rand_2d/pack-valid-%d-%d-7-1-5/' % (blocks_num, num_samples), blocks_num, num_samples)
    
    ratio, positions, box_size, valid_size, box_size, empty_size, stable_num, packing_height, _, _ \
        = calc_positions(net, blocks, container_size, is_train=False)
    
    # import IPython
    # IPython.embed()

    if not os.path.exists(save_dir + '/valid'):
        os.mkdir(save_dir + '/valid')

    np.savetxt( save_dir + '/valid/box_size.txt', box_size)
    np.savetxt( save_dir + '/valid/empty_size.txt', empty_size)
    np.savetxt( save_dir + '/valid/packing_height.txt', packing_height)
    np.savetxt( save_dir + '/valid/ratio.txt', ratio)
    np.savetxt( save_dir + '/valid/stable_num.txt', stable_num)
    np.savetxt( save_dir + '/valid/valid_size.txt', valid_size)

    C = np.mean(valid_size / (box_size))
    P = np.mean(valid_size / (valid_size + empty_size))
    S = np.mean(stable_num / 10 )

    print("C: %.3f   P: %.3f  S: %.3f   R: %.3f" % ( C, P, S, np.mean(ratio) )  )


    # print(np.mean(ratio))


if __name__ == '__main__':
    
    # if gpu is to be used
    use_cuda = True
    if use_cuda: 
        print("....Using Gpu....")
        os.environ["CUDA_VISIBLE_DEVICES"] = '2'


    batch_size = 128
    epoch_num = 150
    # learning_rate = 1e-3
    # learning_rate = 5e-4
    learning_rate = 5e-4

    is_train = False

    note = 'RL_rand'

    heightmap_type = 'diff'

    save_dir = './%s_%s_DL_lb' % (note, heightmap_type)
    print(save_dir, learning_rate)

    container_width, container_height = 5, 100 # container_size
    container_size = (container_width, container_height)
    blocks_num = 10

    block_hidden_size = 128
    height_hidden_size = 128

    # checkpoints = './%s/checkpoints/88' % note
    # checkpoints = './RL_RNN_gt/checkpoints/149'
    # checkpoints = './RL_RNN_mix/checkpoints/99'
    if is_train == False:
        checkpoints = save_dir + '/checkpoints/97'
    else:
        checkpoints = None

    # checkpoints = './DL_rand_lb_diff_True/checkpoints/199'
    actor = PackNet(container_width, container_height, heightmap_type, blocks_num)
    critic = Critic(2, block_hidden_size, container_width, container_height, heightmap_type, blocks_num)

    # print(actor.net)
    # actor[1.5]
    if use_cuda:
        actor.net = actor.net.cuda()
        critic = critic.cuda()

    if checkpoints is not None:
        path = os.path.join(checkpoints, 'actor.pt')
        actor.net.load_state_dict(torch.load(path))

        path = os.path.join(checkpoints, 'critic.pt')
        critic.load_state_dict(torch.load(path))

        print('Loading pre-train model', path)

    # path = os.path.join(checkpoints, 'DL.pt')
    # actor.net.load_state_dict(torch.load(path))
    # print(path)

    if is_train:
        
        tarin_size = 64000
        valid_size = 100
        train(actor, critic, tarin_size, valid_size, blocks_num, batch_size, epoch_num, learning_rate, save_dir, use_cuda, note, is_train)
    else:
        print('valid')
        valid_size = 10000
        valid(actor, valid_size, blocks_num, save_dir )
    