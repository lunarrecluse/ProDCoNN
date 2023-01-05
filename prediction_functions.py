# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:31:32 2019

@author: Yang Chen
"""

# Test
# Input a protein with the form: [[amino acid1],[amino acid2],...] 
# for each amino acid, the form should be [[atom1_type,x,y,z],[atom2_type,x,y,z],...]
# The overall structure of the input data is [[[type,x,y,z],[type,x,y,z],...],[[type,x,y,z],[type,x,y,z],...],...]
# The output is a list of the labels. [label1,label2,...]

# There are 4 models in total, there will be a argument called "mode" to determine the model we are going to use.
# "mode" will take value from {'BBO30','BBO90','BBS30','BBS90'}, default value should be 'BBO90'


from keras.models import load_model
import numpy as np
from scipy.sparse import coo_matrix


# Global parameters
numchannel = {'BBO90':6, 'BBO30':6, 'BBS90':26, 'BBS30':26}
BBSatom =     [0,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.65,0.7,0.7,0.6,0.6,0.7,0.7,0.7,0.7,0.7]
BBOatom =   [0,0.65,0.7,0.7,0.6,0.6,0.7]
atom_sizes = {'BBO90':BBOatom, 'BBO30':BBOatom, 'BBS90':BBSatom, 'BBS30':BBSatom}
width = 18
aat_list=["ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU","MET","ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR"]

def get_remainder_sq(np_array):
    remainder = np_array - np.round(np_array)
    ret = np.column_stack((remainder + 1, remainder, remainder - 1))
    return np.square(ret)


def get_neighbor(np_array):
    neighbor = np.round(np_array)
    ret = np.column_stack((neighbor - 1, neighbor, neighbor + 1))
    return ret


def get_atom_sizes2(at_arr, atom_sizes2):
    ret = [atom_sizes2[idx] for idx in at_arr]
    return np.asarray(ret, dtype=np.float)


def trim(np_arr, low, high):
    np_arr = np.maximum(np_arr, low)
    np_arr = np.minimum(np_arr, high)
    return np_arr


def processpdb(atom_list_T, half_cutoff, cutoff, width, bs, bs_2, channel_choose, atom_sizes2):

    at_arr = np.asarray(atom_list_T[0], dtype = np.int)
    x_arr = np.asarray(atom_list_T[1], dtype = np.float)
    y_arr = np.asarray(atom_list_T[2], dtype = np.float)
    z_arr = np.asarray(atom_list_T[3], dtype = np.float)
    #print(at_arr)
    num_atoms = len(at_arr)

    x_arr = x_arr + half_cutoff
    y_arr = y_arr + half_cutoff
    z_arr = z_arr + half_cutoff

    # if x1 >= 0 and y1 >= 0 and z1 >= 0 and x1 <= cutoff and y1 <= cutoff and z1 <= cutoff:
    is_valid_x = np.logical_and(x_arr >= 0, x_arr <= cutoff)
    is_valid_y = np.logical_and(y_arr >= 0, y_arr <= cutoff)
    is_valid_z = np.logical_and(z_arr >= 0, z_arr <= cutoff)
    is_valid = np.logical_and(is_valid_x, is_valid_y)
    is_valid = np.logical_and(is_valid, is_valid_z)
    #is_valid = is_valid.astype(np.int)

    atom_size2_arr = get_atom_sizes2(at_arr, atom_sizes2)
    atom_size2_arr = atom_size2_arr.reshape(num_atoms, 1)

    x_rem_sq = get_remainder_sq(x_arr)
    y_rem_sq = get_remainder_sq(y_arr)
    z_rem_sq = get_remainder_sq(z_arr)

    x_rem_sq = x_rem_sq.reshape((num_atoms, 3, 1, 1))
    y_rem_sq = y_rem_sq.reshape((num_atoms, 1, 3, 1))
    z_rem_sq = z_rem_sq.reshape((num_atoms, 1, 1, 3))

    dr = x_rem_sq + y_rem_sq + z_rem_sq
    dr = dr.reshape(num_atoms, 27)
    gauss_arr = np.exp(-dr / atom_size2_arr)

    row_sum = np.sum(gauss_arr, axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1)
    gauss_arr = gauss_arr / row_sum

    x_neighbor = get_neighbor(x_arr)
    y_neighbor = get_neighbor(y_arr)
    z_neighbor = get_neighbor(z_arr)

    x_neighbor = trim(x_neighbor, 0, bs - 1).reshape((num_atoms, 3, 1, 1))
    y_neighbor = trim(y_neighbor, 0, bs - 1).reshape((num_atoms, 1, 3, 1))
    z_neighbor = trim(z_neighbor, 0, bs - 1).reshape((num_atoms, 1, 1, 3))

    idx_arr = x_neighbor * bs_2 + y_neighbor * bs + z_neighbor
    idx_arr = idx_arr.reshape(num_atoms, 27)
    idx_arr = idx_arr.astype(np.int)

    row_idx = idx_arr.flatten()
    col_idx = np.repeat(at_arr.reshape(num_atoms, 1), 27, axis=1)
    col_idx = col_idx.flatten() - 1
    values = gauss_arr.flatten()

    is_valid = np.repeat(is_valid.reshape(num_atoms, 1), 27, axis=1).flatten()
    row_idx = row_idx * is_valid
    col_idx = col_idx * is_valid
    values = values * is_valid

    this_coo_mat = coo_matrix((values, (row_idx, col_idx)), shape=(width * width * width, channel_choose))
    
    return this_coo_mat


def Preprocessing(atom_list,nchannel, atom_sizes2):
    cutoff = float(width)
    half_cutoff = cutoff/2
    bs = width
    bs_2 = bs*bs
    at = []
    x = []
    y = []
    z = []
    for atom in atom_list:
        at.append(atom[0])
        x.append(atom[1])
        y.append(atom[2])
        z.append(atom[3])
    atom_list_T = []
    atom_list_T.append(at)
    atom_list_T.append(x)
    atom_list_T.append(y)
    atom_list_T.append(z)
    this_coo_mat = processpdb(atom_list_T, half_cutoff, cutoff, width, bs, bs_2, nchannel, atom_sizes2)
    this_coo_mat = this_coo_mat.toarray()
    this_coo_mat = np.expand_dims(this_coo_mat, axis=0)
    return this_coo_mat

def test(Xtest, mode = 'BBO90'):
    # parameters & variables
    atom_sizes2 = [2*(x)*(x) for x in atom_sizes[mode]] #sigma^2    
    nchannel = numchannel[mode]
    N = len(Xtest)
    prediction = []
    
    # loadmodel
    model_path = mode + '.hdf5'
    model = load_model(model_path)
    
    for index in range(0,N):
        # Preprocessing, transfer [[type,x,y,z],[type,x,y,z],...] to one box with appropriate number of channels
        # feed the made-up data to the model, then get prediction.
        prediction.append(aat_list[int(model.predict_classes(Preprocessing(Xtest[index], nchannel, atom_sizes2)))])
     
    return prediction






#if __name__ == '__main__':
#    Xtest = []
#    prediction = test(Xtest, 'BBO90')
    
    
    
   
