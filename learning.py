from random import random,randint, shuffle, seed
import sys

sys.path.append("game/")
from game2048 import Game2048

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer, InputLayer, batch_norm, DropoutLayer
from lasagne.layers import  MergeLayer, ReshapeLayer, FlattenLayer, ConcatLayer
from lasagne.layers.dnn import Conv2DDNNLayer
from lasagne.regularization import regularize_network_params, l1, l2, regularize_layer_params_weighted
from lasagne.nonlinearities import rectify, elu, softmax, sigmoid
from lasagne.init import Constant, Sparse
import math
from collections import defaultdict

from c2048 import push


floatX = theano.config.floatX

#==================================#
input_var = T.tensor4()
target_var = T.vector()

N_FILTERS = 512
N_FILTERS2 = 4096

_ = InputLayer(shape=(None, 16, 4, 4), input_var=input_var)

conv_a =  Conv2DDNNLayer(_, N_FILTERS, (2,1), pad='valid')#, W=Winit((N_FILTERS, 16, 2, 1)))
conv_b =  Conv2DDNNLayer(_, N_FILTERS, (1,2), pad='valid')#, W=Winit((N_FILTERS, 16, 1, 2)))

conv_aa =  Conv2DDNNLayer(conv_a, N_FILTERS2, (2,1), pad='valid')#, W=Winit((N_FILTERS2, N_FILTERS, 2, 1)))
conv_ab =  Conv2DDNNLayer(conv_a, N_FILTERS2, (1,2), pad='valid')#, W=Winit((N_FILTERS2, N_FILTERS, 1, 2)))

conv_ba =  Conv2DDNNLayer(conv_b, N_FILTERS2, (2,1), pad='valid')#, W=Winit((N_FILTERS2, N_FILTERS, 2, 1)))
conv_bb =  Conv2DDNNLayer(conv_b, N_FILTERS2, (1,2), pad='valid')#, W=Winit((N_FILTERS2, N_FILTERS, 1, 2)))

_ = ConcatLayer([FlattenLayer(x) for x in [conv_aa, conv_ab, conv_ba, conv_bb, conv_a, conv_b]])
l_out = DenseLayer(_, num_units=1,  nonlinearity=None)

prediction = lasagne.layers.get_output(l_out)
P = theano.function([input_var], prediction)
loss = lasagne.objectives.squared_error(prediction, target_var).mean()/2
accuracy = lasagne.objectives.squared_error(prediction, target_var).mean()
params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.adam(loss, params, beta1=0.5)


train_fn = theano.function([input_var, target_var], loss, updates=updates)
loss_fn = theano.function([input_var, target_var], loss)
accuracy_fn =theano.function([input_var, target_var], accuracy)


table ={2**i:i for i in range(1,16)}
table[0]=0

#arrow=[Keys.ARROW_LEFT, Keys.ARROW_UP, Keys.ARROW_RIGHT, Keys.ARROW_DOWN]
#DOWN, RIGHT, UP, LEFT = range(4)
mapper = {0:3,1:2,2:1,3:0}
ACTIONS = 4
logf=open("logf-rl-theano-n-tuple-6", "w")

#===================================#
def make_input(grid):
	g0 = grid
	r = np.zeros(shape=(16, 4, 4), dtype=floatX)
	for i in range(4):
		for j in range(4):
			v = int(g0[i, j])
			r[table[v],i, j]=1
	return r
  
def Vchange(grid, v):
	g0 = grid
	g1 = g0[:,::-1,:]
	g2 = g0[:,:,::-1]
	g3 = g2[:,::-1,:]
	r0 = grid.swapaxes(1,2)
	r1 = r0[:,::-1,:]
	r2 = r0[:,:,::-1]
	r3 = r2[:,::-1,:]
	xtrain = np.array([g0,g1,g2,g3,r0,r1,r2,r3], dtype=floatX)
	ytrain = np.array([v]*8, dtype=floatX)
	train_fn(xtrain, ytrain)

def gen_sample_and_learn(game):
	game_len = 0
	game_score = 0
	last_grid = None
	keep_playing =False
	
	do_nothing = np.zeros(ACTIONS)
	do_nothing[0] = 1
	grid_array,r_0,terminal = game.step(do_nothing)
	grid_array[np.isnan(grid_array)] = 0
	grid_array = grid_array.astype(int)
	#print(grid_array)
	while "1" != "2":
		board_list = []
		if grid_array is not None:
			for m in range(4):
				g = grid_array.copy()
				s = push(g, m%4) #<<<<<<<<<<<<<<<<<<<<<
				if s >= 0:
					board_list.append( (g, m, s) )
		if board_list:
			boards = np.array([make_input(g) for g,m,s in board_list], dtype=floatX)
			p = P(boards).flatten()
			game_len+=1
			best_move = -1
			best_v = None
			for i, (g,m,s) in enumerate(board_list):
				v = 2*s + p[i]
				if best_v is None or v > best_v:
					best_v = v
					best_move = m
					best_score = 2*s
					best_grid = boards[i]
			action = np.zeros(ACTIONS)
			action[mapper[best_move]] = 1
			grid_array,r_0,terminal = game.step(action)
			grid_array[np.isnan(grid_array)] = 0
			grid_array = grid_array.astype(int)
			game_score += best_score
		else:
			best_v = 0
			best_grid = None
		if last_grid is not None:
			Vchange(last_grid, best_v)
		last_grid = best_grid
		if not board_list:
			break
	return game_len, grid_array.max(), game_score



def main():
	for j in range(200):
		game = Game2048()
		result = gen_sample_and_learn(game)
		print(j, result)

if __name__ == "__main__":
    main()