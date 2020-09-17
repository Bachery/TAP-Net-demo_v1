import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import pack
import tools
from model import DRL, DRL_RNN, DRL_L, Encoder


def get_params():

	kwargs =  {}

	# Task settings
	kwargs['task'] 			= 'test'   # train, test, generate
	kwargs['note'] 			= 'debug'
	kwargs['use_cuda'] 		= True
	kwargs['cuda'] 			= '0'
	kwargs['seed'] 			= 12345

	# Training/testing settings
	kwargs['train_size'] = 10
	kwargs['valid_size'] = 1
	kwargs['epoch_num']  = 1
	kwargs['batch_size'] = 128

	# Data settings
	kwargs['obj_dim'] 		= 2
	kwargs['nodes'] 		= 10
	kwargs['num_nodes']		= 10
	kwargs['total_obj_num'] = 10			# if more, do Rolling @TODO
	kwargs['dataset'] 		= 'PPSG'		# RAND, PPSG, MIX
	# sizes of blocks and containers
	kwargs['unit'] = 1.0
	kwargs['arm_size'] 	= 1 # size of robotic arm to pass and rotate a block
	kwargs['min_size'] 	= 1
	kwargs['max_size'] 	= 5
	kwargs['container_width'] 	= 5
	kwargs['container_length'] 	= 5  # for 3D
	kwargs['container_height'] 	= 50
	kwargs['initial_container_width'] 	= 7
	kwargs['initial_container_length'] 	= 7  # for 3D
	kwargs['initial_container_height'] 	= 50

	# Packing settings
	kwargs['packing_strategy'] 	= 'LB_GREEDY'
	kwargs['reward_type'] 		= 'C+P+S-lb-soft'

	# Network settings
	# ---- TODO: network reward
	kwargs['input_type'] 			= 'bot'
	kwargs['allow_rot'] 			= True
	kwargs['decoder_input_type'] 	= 'shape_heightmap' # shape_heightmap, shape_only, heightmap_only
	kwargs['heightmap_type'] 		= 'diff'     # full, zero, diff

	# Network parameters
	kwargs['dropout'] 			= 0.1
	kwargs['actor_lr'] 			= 5e-4
	kwargs['critic_lr'] 		= 5e-4
	kwargs['max_grad_norm'] 	= 2.
	kwargs['n_process_blocks'] 	= 3
	kwargs['layers']     		= 1
	kwargs['num_layers'] 		= 1
	kwargs['encoder_hidden']      = 128
	kwargs['encoder_hidden_size'] = 128
	kwargs['decoder_hidden']      = 256
	kwargs['decoder_hidden_size'] = 256

	if kwargs['dataset'] == 'PPSG':
		kwargs['checkpoint'] = './pack/2d-bot-C+P+S-lb-soft-width-5-note-sh-G-newdiff_resume2-2020-05-09-14-40/checkpoints/299/'
	elif kwargs['dataset'] == 'RAND':
		kwargs['checkpoint'] = './pack/2d-bot-C+P+S-lb-soft-width-5-note-sh-R-newdiff_resume2-2020-05-09-14-39/checkpoints/299/'

	return kwargs


class PACKDataset(Dataset):
	def __init__(self, data_file, test_index, blocks_num, num_samples, seed, input_type, heightmap_type, allow_rot, container_width, mix_data_file=None, unit=1):
		'''
		Data initialization
		----
		params
		----
			input_type: str, the type of input data
				'simple':   [idx][w,(l),h]     [0,1,1,0,...][   ]

				'rot':      [idx][w,(l),h]     [0,1,1,0,...][   ]
				'rot-old':  [idx][w,(l),h]     [0,1,1,0,...][0]

				'bot':      [idx][w,(l),h]     [0,1,1,0,...][0,0,1,1,...][0,1,0,1,...]
				'bot-rot':  [idx][w,(l),h]     [0,1,1,0,...][0,0,0,0,0..][0,0,0,0,0..]

				'mul':      [idx][w,(l),h][  ] [0,1,1,0,...] # old
				'mul-with': [idx][w,(l),h][id] [0,1,1,0,...] # old

				'mul':      [idx][w,(l),h][  ] [0,1,1,0,...][0,0,1,1,...][0,1,0,1,...]
				'mul-with': [idx][w,(l),h][id] [0,1,1,0,...][0,0,1,1,...][0,1,0,1,...]

			allow_rot: bool, allow to rotate
			False:  the final dim of input will be blocks_num * 1
			True:   the final dim of input will be blocks_num * rotate_types
		'''
		super(PACKDataset, self).__init__()
		# if seed is None:
		# 	seed = np.random.randint(123456)
		# np.random.seed(seed)
		# torch.manual_seed(seed)


		if mix_data_file is None:
			deps_move = np.loadtxt(data_file + 'dep_move.txt').astype('float32')
			rotate_deps_small = np.loadtxt(data_file + 'dep_small.txt').astype('float32')
			rotate_deps_large = np.loadtxt(data_file + 'dep_large.txt').astype('float32')

			blocks = np.loadtxt(data_file + 'blocks.txt').astype('float32')
			positions = np.loadtxt(data_file + 'pos.txt').astype('float32')
			container_index = np.loadtxt(data_file + 'container.txt').astype('float32')

		else:
			num_mid = int(num_samples / 2)
			print('Mixing... %d' % num_mid)
			deps_move = np.loadtxt(data_file + 'dep_move.txt').astype('float32')[:num_mid]
			positions = np.loadtxt(data_file + 'pos.txt').astype('float32')[:num_mid]
			container_index = np.loadtxt(data_file + 'container.txt').astype('float32')[:num_mid]

			rotate_deps_small = np.loadtxt(data_file + 'dep_small.txt').astype('float32')
			rotate_deps_large = np.loadtxt(data_file + 'dep_large.txt').astype('float32')
			blocks = np.loadtxt(data_file + 'blocks.txt').astype('float32')

			rot_num_mid = int( len(blocks) / 2)
			rotate_deps_small = np.loadtxt(data_file + 'dep_small.txt').astype('float32')[:rot_num_mid]
			rotate_deps_large = np.loadtxt(data_file + 'dep_large.txt').astype('float32')[:rot_num_mid]
			blocks = np.loadtxt(data_file + 'blocks.txt').astype('float32')[:rot_num_mid]

			mix_deps_move = np.loadtxt(mix_data_file + 'dep_move.txt').astype('float32')[:num_mid]
			mix_positions = np.loadtxt(mix_data_file + 'pos.txt').astype('float32')[:num_mid]
			mix_container_index = np.loadtxt(mix_data_file + 'container.txt').astype('float32')[:num_mid]

			mix_rotate_deps_small = np.loadtxt(mix_data_file + 'dep_small.txt').astype('float32')[:rot_num_mid]
			mix_rotate_deps_large = np.loadtxt(mix_data_file + 'dep_large.txt').astype('float32')[:rot_num_mid]
			mix_blocks = np.loadtxt(mix_data_file + 'blocks.txt').astype('float32')[:rot_num_mid]


			deps_move = np.vstack( (deps_move, mix_deps_move) )
			rotate_deps_small = np.vstack( (rotate_deps_small, mix_rotate_deps_small) )
			rotate_deps_large = np.vstack( (rotate_deps_large, mix_rotate_deps_large) )
			blocks = np.vstack( (blocks, mix_blocks) )
			positions = np.vstack( (positions, mix_positions) )
			container_index = np.vstack( (container_index, mix_container_index) )


		positions = torch.from_numpy(positions)
		positions = positions.view(num_samples, -1, blocks_num)

		deps_move = torch.from_numpy(deps_move)
		deps_move = deps_move.view(num_samples, -1, blocks_num)
		deps_move = deps_move.transpose(2, 1)

		block_dim = positions.shape[1]
		rotate_types = np.math.factorial(block_dim)

		# data_size = int(len(blocks) / rotate_types)
		# num_samples x rotate_types x block_dim x blocks_num
		blocks = blocks.reshape( num_samples, -1, block_dim, blocks_num)
		# num_samples x rotate_types x blocks_num x block_dim
		blocks = blocks.transpose(0, 1, 3, 2)
		# num_samples x (rotate_types * blocks_num) x block_dim
		blocks = blocks.reshape( num_samples, -1, block_dim )
		# num_samples x block_dim x (blocks_num * rotate_types)
		blocks = blocks.transpose(0,2,1)
		blocks = torch.from_numpy(blocks)

		# resolution
		blocks = blocks * unit
		# if unit<1:
		blocks = blocks.ceil()#.int()

		rotate_deps_small = rotate_deps_small.reshape( num_samples, -1, blocks_num, blocks_num )
		rotate_deps_large = rotate_deps_large.reshape( num_samples, -1, blocks_num, blocks_num )
		rotate_deps_small = rotate_deps_small.transpose(0,1,3,2)
		rotate_deps_large = rotate_deps_large.transpose(0,1,3,2)
		rotate_deps_small = rotate_deps_small.reshape( num_samples, blocks_num*rotate_types, blocks_num )
		rotate_deps_large = rotate_deps_large.reshape( num_samples, blocks_num*rotate_types, blocks_num )
		rotate_deps_small = rotate_deps_small.transpose(0,2,1)
		rotate_deps_large = rotate_deps_large.transpose(0,2,1)
		rotate_deps_small = torch.from_numpy(rotate_deps_small)
		rotate_deps_large = torch.from_numpy(rotate_deps_large)

		# check rotate type:
		if allow_rot == False:
			blocks = blocks[:,:,:blocks_num]
			rotate_types = 1

		blocks_index = torch.arange(blocks_num)
		blocks_index = blocks_index.unsqueeze(0).unsqueeze(0)
		# num_samples x 1 x (blocks_num * rotate_types)
		blocks_index = blocks_index.repeat(num_samples, 1, rotate_types).float()

		# import IPython

		container_index = torch.from_numpy(container_index)
		container_index = container_index.unsqueeze(1)
		container_index = container_index.repeat(1, 1, rotate_types).float()
		# num_samples x block_dim x (blocks_num * rotate_types)
		positions = positions.repeat(1,1,rotate_types)
		# num_samples x blocks_num x (blocks_num * rotate_types)
		deps_move = deps_move.repeat(1,1,rotate_types)

		# print('blocks:', blocks.shape)
		# print('blocks_index: ', blocks_index.shape)
		# print('positions:', positions.shape)
		# print('container:', container_index.shape)
		# print('deps_move:', deps_move.shape)
		# print('rotate_deps_small:', rotate_deps_small.shape)
		# print('rotate_deps_large:', rotate_deps_large.shape)
		# print()

		# take only one case
		blocks = blocks[test_index:test_index+1]
		blocks_index = blocks_index[test_index:test_index+1]
		positions = positions[test_index:test_index+1]
		container_index = container_index[test_index:test_index+1]
		deps_move = deps_move[test_index:test_index+1]
		rotate_deps_small = rotate_deps_small[test_index:test_index+1]
		rotate_deps_large = rotate_deps_large[test_index:test_index+1]
		num_samples = 1

		# print('blocks:', blocks.shape)
		# print('blocks_index: ', blocks_index.shape)
		# print('positions:', positions.shape)
		# print('container:', container_index.shape)
		# print('deps_move:', deps_move.shape)
		# print('rotate_deps_small:', rotate_deps_small.shape)
		# print('rotate_deps_large:', rotate_deps_large.shape)
		# print()

		# # # random shuffle
		# order = [ o for o in range(blocks_num) ]
		# np.random.shuffle(order)
		# order = order * rotate_types
		# order = np.array(order)
		# for r in range(1, rotate_types):
		#     order[r*blocks_num: ] += blocks_num
		# blocks = blocks[:,:,order]
		# blocks_index = blocks_index[:,:,order]
		# deps_move = deps_move[:,:,order]
		# rotate_deps_small = rotate_deps_small[:,:,order]
		# rotate_deps_large = rotate_deps_large[:,:,order]
		# container_index = container_index[:,:,order]
		# positions = positions[:,:,order]

		# conbine the data into our final input
		if input_type == 'simple':
			# num_samples x (1 + block_dim) x (blocks_num * 1)
			self.static = torch.cat( (blocks_index, blocks), 1 )
			# num_samples x (blocks_num) x (blocks_num * 1)
			self.dynamic = deps_move
		elif input_type == 'rot':
			# num_samples x (1 + block_dim) x (blocks_num * rotate_types)
			self.static = torch.cat( (blocks_index, blocks), 1 )
			# num_samples x (blocks_num) x (blocks_num * rotate_types)
			self.dynamic = deps_move
		elif input_type == 'bot':
			# num_samples x (1 + block_dim) x (blocks_num * rotate_types)
			self.static = torch.cat( (blocks_index, blocks), 1 )
			# num_samples x (blocks_num * 3) x (blocks_num * rotate_types)
			self.dynamic = torch.cat( (deps_move, rotate_deps_small, rotate_deps_large), 1 )
		elif input_type == 'bot-rot':
			# num_samples x (1 + block_dim) x (blocks_num * rotate_types)
			self.static = torch.cat( (blocks_index, blocks), 1 )
			# num_samples x (blocks_num * 3) x (blocks_num * rotate_types)
			rotate_deps_small = torch.zeros_like(rotate_deps_small)
			rotate_deps_large = torch.zeros_like(rotate_deps_large)
			self.dynamic = torch.cat( (deps_move, rotate_deps_small, rotate_deps_large), 1 )

		elif input_type == 'use-static' or input_type == 'use-pnet':
			# num_samples x (1 + block_dim) x (blocks_num * rotate_types)
			self.static = torch.cat( (blocks_index, blocks), 1 )
			# num_samples x (blocks_num * 3) x (blocks_num * rotate_types)
			rotate_deps_small = torch.zeros_like(rotate_deps_small)
			rotate_deps_large = torch.zeros_like(rotate_deps_large)
			self.dynamic = torch.cat( (deps_move, rotate_deps_small, rotate_deps_large), 1 )

		elif input_type == 'mul' or input_type == 'mul-with':
			# num_samples x (1 + block_dim + 1) x (blocks_num * rotate_types)
			self.static = torch.cat( (blocks_index, blocks, container_index), 1 )
			# num_samples x (blocks_num * 3) x (blocks_num * rotate_types)
			self.dynamic = torch.cat( (deps_move, rotate_deps_small, rotate_deps_large), 1 )

		elif input_type == 'rot-old':
			rotate_state = torch.zeros_like(blocks_index)
			# num_samples x (1 + block_dim) x (blocks_num * rotate_types)
			self.static = torch.cat( (blocks_index, blocks), 1 )
			# num_samples x (blocks_num + 1) x (blocks_num * rotate_types)
			self.dynamic = torch.cat( (deps_move, rotate_state), 1 )

		else:
			print('Dataset OHHHHH')
        
		print('    Static shape:  ', self.static.shape)
		print('    Dynamic shape: ', self.dynamic.shape)

		static_dim = block_dim
		heightmap_num = 1

		if heightmap_type == 'diff':
			if block_dim == 2:
				heightmap_width = container_width * unit - 1
			elif block_dim == 3:
				heightmap_num = 2
				heightmap_width = container_width * unit
				heightmap_length = container_width * unit
		else:
			heightmap_width = container_width * unit
			heightmap_length = container_width * unit

		# if unit < 1:
		heightmap_width = np.ceil(heightmap_width).astype(int)
		if block_dim==3: heightmap_length = np.ceil(heightmap_length).astype(int)


		if input_type == 'mul' or input_type == 'mul-with':
			if block_dim == 2:
				heightmap_width = heightmap_width * 2
			else:
				heightmap_num = heightmap_num * 2

		if input_type == 'mul-with':
			static_dim = static_dim + 1

		if block_dim == 2:
			self.decoder_static = torch.zeros(num_samples, static_dim, 1, requires_grad=True)
			self.decoder_dynamic = torch.zeros(num_samples, heightmap_width, 1, requires_grad=True)
		elif block_dim == 3:
			self.decoder_static = torch.zeros(num_samples, static_dim, 1, requires_grad=True)
			self.decoder_dynamic = torch.zeros(num_samples, heightmap_num, heightmap_width, heightmap_length, requires_grad=True)

		self.num_samples = num_samples

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		# (static, dynamic, start_loc)
		return (self.static[idx], self.dynamic[idx], self.decoder_static[idx], self.decoder_dynamic[idx])


def render(static, tour_indices, **kwargs):

	if kwargs['allow_rot'] == False: rotate_types = 1
	else: rotate_types = np.math.factorial(kwargs['obj_dim'])

	blocks_num = int( static.shape[2] / rotate_types )

	all_blocks = static.data[:,1:1+kwargs['obj_dim'],:].cpu().numpy()
	all_blocks = all_blocks.transpose(0, 2, 1).astype('int')

	container_width  = kwargs['container_width']  * kwargs['unit']
	container_height = kwargs['container_height'] * kwargs['unit']
	initial_container_width  = kwargs['initial_container_width']  * kwargs['unit']
	initial_container_height = kwargs['initial_container_height'] * kwargs['unit']
	container_width  = np.ceil(container_width).astype(int)
	container_height = np.ceil(container_height).astype(int)
	initial_container_width  = np.ceil(initial_container_width).astype(int)
	initial_container_height = np.ceil(initial_container_height).astype(int)

	if kwargs['obj_dim'] == 3:
		container_size = [container_width, container_width, container_height]
		initial_container_size = [initial_container_width, initial_container_width, initial_container_height]
	else:
		container_size = [container_width, container_height]
		initial_container_size = [initial_container_width, initial_container_height]
	if kwargs['input_type'] == 'mul' or kwargs['input_type'] == 'mul-with':
		if kwargs['obj_dim'] == 3:
			container_size_a = [container_width, container_width, initial_container_height]
			container_size_b = container_size_a
		else:
			container_size_a = [container_width, container_height]
			container_size_b = container_size_a

	if kwargs['packing_strategy'] == 'MACS' or kwargs['packing_strategy'] == 'MUL':
		calc_position_fn = tools.calc_positions_mcs
	elif kwargs['reward_type'] == 'C+P+S-SL-soft' or kwargs['reward_type'] == 'C+P+S-RL-soft' or \
		 kwargs['reward_type'] == 'C+P+S-G-soft'  or kwargs['reward_type'] == 'C+P+S-LG-soft':
		calc_position_fn = tools.calc_positions_net
	else:
		calc_position_fn = tools.calc_positions_lb_greedy

	# for the case
	if kwargs['dataset'] == 'PPSG':
		valid_file = './data/ppsg_2d/pack-valid-10-10000-7-1-5/'
	elif kwargs['dataset'] == 'RAND':
		valid_file = './data/rand_2d/pack-valid-10-10000-7-1-5/'
	all_init_positions 	= np.loadtxt(valid_file + 'pos.txt').astype('float32')
	all_init_positions 	= all_init_positions.reshape( len(all_init_positions), kwargs['obj_dim'], -1 )
	all_init_positions 	= all_init_positions.transpose(0,2,1)
	init_positions = all_init_positions[kwargs['test_index']]
	i = 0
	order = tour_indices[i].long().cpu().numpy()
	blocks = all_blocks[i][order]
	init_blocks = all_blocks[i][:blocks_num]
	init_heights = init_positions[:, 1] + init_blocks[:, 1]
	init_height = np.max(init_heights).astype('int')

	if kwargs['input_type'] == 'mul' or kwargs['input_type'] == 'mul-with':
		print('not done yet')
	else:
		positions, _, _, _, scores = calc_position_fn(blocks, container_size, kwargs['reward_type'])
		valid_size, box_size, empty_num, stable_num, packing_height = scores
	
	C = valid_size / box_size
	P = valid_size / (valid_size + empty_num)
	S = stable_num / blocks_num

	# draw both initial and target container step by step
	real_order, rotate_state = tools.calc_real_order(order, blocks_num)

	for step in range(blocks_num + 1):
		init_order = real_order[step:]
		tools.draw_container_2d(init_blocks[init_order], init_positions[init_order], initial_container_size, 
								order=init_order, packing_height=init_height, 
								save_name=kwargs['save_dir'] + '/' + 'init_' + str(step) )
		tools.draw_container_2d(blocks[0:step], positions[0:step], container_size, 
								order=real_order[0:step], rotate_state=rotate_state[0:step], 
								packing_height=packing_height, 
								save_name=kwargs['save_dir'] + '/' + 'target_' + str(step) )
	print('Process saved step by step')


def test(actor, **kwargs):
	
	test_loader = DataLoader(kwargs['valid_data'], len(kwargs['valid_data']), shuffle=False, num_workers=0)

	for batch_idx, batch in enumerate(test_loader):
		encoder_static, encoder_dynamic, decoder_static, decoder_dynamic = batch

		if kwargs['use_cuda']:
			encoder_static = encoder_static.cuda()
			encoder_dynamic = encoder_dynamic.cuda()
			decoder_static = decoder_static.cuda()
			decoder_dynamic = decoder_dynamic.cuda()

		# Full forward pass through the dataset
		if kwargs['decoder_input_type'] == 'shape_only':
			decoder_input = decoder_static
		else:
			decoder_input = [decoder_static, decoder_dynamic]

		with torch.no_grad():
			# start = time.time()
			tour_indices, tour_logp, pack_logp, reward = actor(encoder_static, encoder_dynamic, decoder_input)
			# valid_time = time.time() - start

		# reward = reward.mean().item()
		print(tour_indices)

		render(encoder_static, tour_indices, **kwargs)


def test_pack(save_dir):

	kwargs = get_params()

	kwargs['test_index'] = np.random.randint(0, 10000)
	print('test_index: ', kwargs['test_index'])
	kwargs['save_dir'] = save_dir

	# date = datetime.datetime.now()
	# now = '%s' % date.date()
	# now += '-%s' % date.hour
	# now += '-%s' % date.minute
	# now = str(now)
	# kwargs['save_dir'] = os.path.join('test_result', now + '_' + kwargs['dataset'] + '-' + str(kwargs['test_index']))
	if not os.path.exists(kwargs['save_dir']):
		os.makedirs(kwargs['save_dir'])

	if kwargs['input_type'] == 'simple':
		STATIC_SIZE = kwargs['obj_dim']
		DYNAMIC_SIZE = kwargs['num_nodes']
	elif kwargs['input_type'] == 'rot':
		STATIC_SIZE = kwargs['obj_dim']
		DYNAMIC_SIZE = kwargs['num_nodes']
	elif kwargs['input_type'] == 'bot' or kwargs['input_type'] == 'bot-rot':
		STATIC_SIZE = kwargs['obj_dim']
		DYNAMIC_SIZE = kwargs['num_nodes'] * 3
	elif kwargs['input_type'] == 'use-static' or kwargs['input_type'] == 'use-pnet':
		STATIC_SIZE = kwargs['obj_dim']
		DYNAMIC_SIZE = kwargs['num_nodes'] * 3
	elif kwargs['input_type'] == 'mul':
		STATIC_SIZE = kwargs['obj_dim']
		DYNAMIC_SIZE = kwargs['num_nodes'] * 3
	elif kwargs['input_type'] == 'mul-with':
		STATIC_SIZE = kwargs['obj_dim'] + 1
		DYNAMIC_SIZE = kwargs['num_nodes'] * 3
	elif kwargs['input_type'] == 'rot-old':
		STATIC_SIZE = kwargs['obj_dim'] + 1
		DYNAMIC_SIZE = kwargs['num_nodes'] + 1
	else:
		print('Unkown input_type')

	if kwargs['dataset'] == 'MIX':
		print('not done yet')
	elif kwargs['dataset'] == 'PPSG':
		valid_file = './data/ppsg_2d/pack-valid-10-10000-7-1-5/'
	elif kwargs['dataset'] == 'RAND':
		valid_file = './data/rand_2d/pack-valid-10-10000-7-1-5/'
	else:
		print('Unkown dataset type')

	valid_data = PACKDataset(
		valid_file,
		kwargs['test_index'],
		kwargs['num_nodes'],
		# kwargs['valid_size'],
		10000,
		kwargs['seed'] + 1,
		kwargs['input_type'],
		kwargs['heightmap_type'],
		kwargs['allow_rot'],
		kwargs['container_width'],
		unit=kwargs['unit']
	)

	if kwargs['reward_type'] == 'C+P+S-G-soft' or kwargs['reward_type'] == 'C+P+S-LG-soft':
		network = DRL_RNN
	elif kwargs['reward_type'] == 'C+P+S-SL-soft' or kwargs['reward_type'] == 'C+P+S-RL-soft':
		network = DRL_L
	else:
		network = DRL

	actor = network(
		STATIC_SIZE,
		DYNAMIC_SIZE,
		kwargs['encoder_hidden_size'],
		kwargs['decoder_hidden_size'],
		kwargs['use_cuda'],
		kwargs['input_type'],
		kwargs['allow_rot'],
		kwargs['container_width'],
		kwargs['container_height'],
		kwargs['obj_dim'],
		kwargs['reward_type'],
		kwargs['decoder_input_type'],
		kwargs['heightmap_type'],
		kwargs['packing_strategy'],
		pack.update_dynamic,
		pack.update_mask,
		kwargs['num_layers'],
		kwargs['dropout'],
		kwargs['unit']
	)

	if kwargs['use_cuda']:
		actor = actor.cuda()

	kwargs['valid_data'] = valid_data

	if kwargs['checkpoint']:
		path = os.path.join(kwargs['checkpoint'], 'actor.pt')
		actor.load_state_dict(torch.load(path))
		print('Loading pre-train model: ', path)

	test(actor, **kwargs)

	# return kwargs['save_dir']
	# return json.dumps(kwargs)


if __name__ == "__main__":
	
	test_pack('test_result/debug/')