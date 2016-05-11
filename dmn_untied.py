import random
import os
import numpy as np

import nltk

import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne
from lasagne import layers
from lasagne import nonlinearities
import cPickle as pickle

from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix

import utils
import nn_utils

#theano.config.floatX = 'float32'
floatX = theano.config.floatX


class DMN_untied:
    
    def __init__(self, babi_train_raw, babi_test_raw, word2vec, word_vector_size, sent_vector_size, 
                dim, mode, answer_module, input_mask_mode, memory_hops, l2, 
                normalize_attention, batch_norm, dropout, dropout_in, **kwargs):

        print "==> not used params in DMN class:", kwargs.keys()
        self.vocab = {None: 0}
        self.ivocab = {0: None}
        
        self.word2vec = word2vec
        self.word_vector_size = word_vector_size
        self.sent_vector_size = sent_vector_size
        self.dim = dim
        self.mode = mode
        self.answer_module = answer_module
        self.input_mask_mode = input_mask_mode
        self.memory_hops = memory_hops
        self.l2 = l2
        self.normalize_attention = normalize_attention
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_in = dropout_in

        self.max_inp_sent_len = 0
        self.max_q_len = 0

        """
        #To Use All Vocab
        self.vocab = {None: 0, 'jason': 134.0, 'office': 14.0, 'yellow': 78.0, 'bedroom': 24.0, 'go': 108.0, 'yes': 15.0, 'antoine': 138.0, 'milk': 139.0, 'before': 46.0, 'grabbed': 128.0, 'fit': 100.0, 'how': 105.0, 'swan': 73.0, 'than': 96.0, 'to': 13.0, 'does': 99.0, 's,e': 110.0, 'east': 102.0, 'rectangle': 82.0, 'gave': 149.0, 'then': 39.0, 'evening': 48.0, 'triangle': 79.0, 'garden': 37.0, 'get': 131.0, 'football,apple,milk': 179.0, 'they': 41.0, 'not': 178.0, 'bigger': 95.0, 'gray': 77.0, 'school': 6.0, 'apple': 142.0, 'did': 127.0, 'morning': 44.0, 'discarded': 146.0, 'julius': 72.0, 'she': 29.0, 'went': 11.0, 'where': 30.0, 'jeff': 152.0, 'square': 84.0, 'who': 153.0, 'tired': 124.0, 'there': 130.0, 'back': 12.0, 'lion': 70.0, 'are': 50.0, 'picked': 143.0, 'e,e': 119.0, 'pajamas': 129.0, 'Mary': 157.0, 'blue': 83.0, 'what': 63.0, 'container': 98.0, 'rhino': 76.0, 'daniel': 31.0, 'bernhard': 67.0, 'milk,football': 172.0, 'above': 80.0, 'got': 136.0, 'emily': 60.0, 'red': 88.0, 'either': 3.0, 'sheep': 58.0, 'football': 137.0, 'jessica': 61.0, 'do': 106.0, 'Bill': 155.0, 'football,apple': 168.0, 'fred': 1.0, 'winona': 59.0, 'objects': 161.0, 'put': 147.0, 'kitchen': 17.0, 'box': 90.0, 'received': 154.0, 'journeyed': 25.0, 'of': 52.0, 'wolf': 62.0, 'afternoon': 47.0, 'or': 7.0, 'south': 112.0, 's,w': 114.0, 'afterwards': 32.0, 'sumit': 123.0, 'color': 75.0, 'julie': 23.0, 'one': 163.0, 'down': 148.0, 'nothing': 167.0, 'n,n': 113.0, 'right': 86.0, 's,s': 116.0, 'gertrude': 54.0, 'bathroom': 26.0, 'from': 109.0, 'west': 104.0, 'chocolates': 91.0, 'two': 165.0, 'frog': 66.0, '.': 9.0, 'cats': 57.0, 'apple,milk,football': 175.0, 'passed': 158.0, 'apple,football,milk': 176.0, 'white': 71.0, 'john': 35.0, 'was': 45.0, 'mary': 10.0, 'apple,football': 170.0, 'north': 103.0, 'n,w': 111.0, 'that': 28.0, 'park': 8.0, 'took': 141.0, 'chocolate': 101.0, 'carrying': 162.0, 'n,e': 120.0, 'mice': 49.0, 'travelled': 22.0, 'he': 33.0, 'none': 164.0, 'bored': 133.0, 'e,n': 117.0, None: 0, 'Jeff': 159.0, 'this': 43.0, 'inside': 93.0, 'bill': 16.0, 'up': 144.0, 'cat': 64.0, 'will': 125.0, 'below': 87.0, 'greg': 74.0, 'three': 166.0, 'suitcase': 97.0, 'following': 36.0, 'e,s': 115.0, 'and': 40.0, 'thirsty': 135.0, 'cinema': 19.0, 'is': 2.0, 'moved': 18.0, 'yann': 132.0, 'sphere': 89.0, 'dropped': 145.0, 'in': 4.0, 'mouse': 56.0, 'football,milk': 171.0, 'pink': 81.0, 'afraid': 51.0, 'no': 20.0, 'Fred': 156.0, 'w,s': 121.0, 'handed': 151.0, 'w,w': 118.0, 'brian': 69.0, 'chest': 94.0, 'w,n': 122.0, 'you': 107.0, 'many': 160.0, 'lily': 65.0, 'hallway': 34.0, 'why': 126.0, 'after': 27.0, 'yesterday': 42.0, 'sandra': 38.0, 'fits': 92.0, 'milk,football,apple': 173.0, 'the': 5.0, 'milk,apple': 169.0, 'a': 55.0, 'give': 150.0, 'longer': 177.0, 'maybe': 21.0, 'hungry': 140.0, 'apple,milk': 174.0, 'green': 68.0, 'wolves': 53.0, 'left': 85.0}
        self.ivocab = {0: None, 1: 'fred', 2: 'is', 3: 'either', 4: 'in', 5: 'the', 6: 'school', 7: 'or', 8: 'park', 9: '.', 10: 'mary', 11: 'went', 12: 'back', 13: 'to', 14: 'office', 15: 'yes', 16: 'bill', 17: 'kitchen', 18: 'moved', 19: 'cinema', 20: 'no', 21: 'maybe', 22: 'travelled', 23: 'julie', 24: 'bedroom', 25: 'journeyed', 26: 'bathroom', 27: 'after', 28: 'that', 29: 'she', 30: 'where', 31: 'daniel', 32: 'afterwards', 33: 'he', 34: 'hallway', 35: 'john', 36: 'following', 37: 'garden', 38: 'sandra', 39: 'then', 40: 'and', 41: 'they', 42: 'yesterday', 43: 'this', 44: 'morning', 45: 'was', 46: 'before', 47: 'afternoon', 48: 'evening', 49: 'mice', 50: 'are', 51: 'afraid', 52: 'of', 53: 'wolves', 54: 'gertrude', 55: 'a', 56: 'mouse', 57: 'cats', 58: 'sheep', 59: 'winona', 60: 'emily', 61: 'jessica', 62: 'wolf', 63: 'what', 64: 'cat', 65: 'lily', 66: 'frog', 67: 'bernhard', 68: 'green', 69: 'brian', 70: 'lion', 71: 'white', 72: 'julius', 73: 'swan', 74: 'greg', 75: 'color', 76: 'rhino', 77: 'gray', 78: 'yellow', 79: 'triangle', 80: 'above', 81: 'pink', 82: 'rectangle', 83: 'blue', 84: 'square', 85: 'left', 86: 'right', 87: 'below', 88: 'red', 89: 'sphere', 90: 'box', 91: 'chocolates', 92: 'fits', 93: 'inside', 94: 'chest', 95: 'bigger', 96: 'than', 97: 'suitcase', 98: 'container', 99: 'does', 100: 'fit', 101: 'chocolate', 102: 'east', 103: 'north', 104: 'west', 105: 'how', 106: 'do', 107: 'you', 108: 'go', 109: 'from', 110: 's,e', 111: 'n,w', 112: 'south', 113: 'n,n', 114: 's,w', 115: 'e,s', 116: 's,s', 117: 'e,n', 118: 'w,w', 119: 'e,e', 120: 'n,e', 121: 'w,s', 122: 'w,n', 123: 'sumit', 124: 'tired', 125: 'will', 126: 'why', 127: 'did', 128: 'grabbed', 129: 'pajamas', 130: 'there', 131: 'get', 132: 'yann', 133: 'bored', 134: 'jason', 135: 'thirsty', 136: 'got', 137: 'football', 138: 'antoine', 139: 'milk', 140: 'hungry', 141: 'took', 142: 'apple', 143: 'picked', 144: 'up', 145: 'dropped', 146: 'discarded', 147: 'put', 148: 'down', 149: 'gave', 150: 'give', 151: 'handed', 152: 'jeff', 153: 'who', 154: 'received', 155: 'Bill', 156: 'Fred', 157: 'Mary', 158: 'passed', 159: 'Jeff', 160: 'many', 161: 'objects', 162: 'carrying', 163: 'one', 164: 'none', 165: 'two', 166: 'three', 167: 'nothing', 168: 'football,apple', 169: 'milk,apple', 170: 'apple,football', 171: 'football,milk', 172: 'milk,football', 173: 'milk,football,apple', 174: 'apple,milk', 175: 'apple,milk,football', 176: 'apple,football,milk', 177: 'longer', 178: 'not', 179: 'football,apple,milk'}
        #self.vocab = {'jason': 134.0, 'office': 14.0, 'yellow': 78.0, 'bedroom': 24.0, 'go': 108.0, 'yes': 15.0, 'antoine': 138.0, 'milk': 139.0, 'before': 46.0, 'grabbed': 128.0, 'fit': 100.0, 'how': 105.0, 'swan': 73.0, 'than': 96.0, 'to': 13.0, 'does': 99.0, 's,e': 110.0, 'east': 102.0, 'rectangle': 82.0, 'gave': 149.0, 'then': 39.0, 'evening': 48.0, 'triangle': 79.0, 'garden': 37.0, 'get': 131.0, 'football,apple,milk': 179.0, 'they': 41.0, 'not': 178.0, 'bigger': 95.0, 'gray': 77.0, 'school': 6.0, 'apple': 142.0, 'did': 127.0, 'morning': 44.0, 'discarded': 146.0, 'julius': 72.0, 'she': 29.0, 'went': 11.0, 'where': 30.0, 'jeff': 152.0, 'square': 84.0, 'who': 153.0, 'tired': 124.0, 'there': 130.0, 'back': 12.0, 'lion': 70.0, 'are': 50.0, 'picked': 143.0, 'e,e': 119.0, 'pajamas': 129.0, 'Mary': 157.0, 'blue': 83.0, 'what': 63.0, 'container': 98.0, 'rhino': 76.0, 'daniel': 31.0, 'bernhard': 67.0, 'milk,football': 172.0, 'above': 80.0, 'got': 136.0, 'emily': 60.0, 'red': 88.0, 'either': 3.0, 'sheep': 58.0, 'football': 137.0, 'jessica': 61.0, 'do': 106.0, 'Bill': 155.0, 'football,apple': 168.0, 'fred': 1.0, 'winona': 59.0, 'objects': 161.0, 'put': 147.0, 'kitchen': 17.0, 'box': 90.0, 'received': 154.0, 'journeyed': 25.0, 'of': 52.0, 'wolf': 62.0, 'afternoon': 47.0, 'or': 7.0, 'south': 112.0, 's,w': 114.0, 'afterwards': 32.0, 'sumit': 123.0, 'color': 75.0, 'julie': 23.0, 'one': 163.0, 'down': 148.0, 'nothing': 167.0, 'n,n': 113.0, 'right': 86.0, 's,s': 116.0, 'gertrude': 54.0, 'bathroom': 26.0, 'from': 109.0, 'west': 104.0, 'chocolates': 91.0, 'two': 165.0, 'frog': 66.0, '.': 9.0, 'cats': 57.0, 'apple,milk,football': 175.0, 'passed': 158.0, 'apple,football,milk': 176.0, 'white': 71.0, 'john': 35.0, 'was': 45.0, 'mary': 10.0, 'apple,football': 170.0, 'north': 103.0, 'n,w': 111.0, 'that': 28.0, 'park': 8.0, 'took': 141.0, 'chocolate': 101.0, 'carrying': 162.0, 'n,e': 120.0, 'mice': 49.0, 'travelled': 22.0, 'he': 33.0, 'none': 164.0, 'bored': 133.0, 'e,n': 117.0, None: 0, 'Jeff': 159.0, 'this': 43.0, 'inside': 93.0, 'bill': 16.0, 'up': 144.0, 'cat': 64.0, 'will': 125.0, 'below': 87.0, 'greg': 74.0, 'three': 166.0, 'suitcase': 97.0, 'following': 36.0, 'e,s': 115.0, 'and': 40.0, 'thirsty': 135.0, 'cinema': 19.0, 'is': 2.0, 'moved': 18.0, 'yann': 132.0, 'sphere': 89.0, 'dropped': 145.0, 'in': 4.0, 'mouse': 56.0, 'football,milk': 171.0, 'pink': 81.0, 'afraid': 51.0, 'no': 20.0, 'Fred': 156.0, 'w,s': 121.0, 'handed': 151.0, 'w,w': 118.0, 'brian': 69.0, 'chest': 94.0, 'w,n': 122.0, 'you': 107.0, 'many': 160.0, 'lily': 65.0, 'hallway': 34.0, 'why': 126.0, 'after': 27.0, 'yesterday': 42.0, 'sandra': 38.0, 'fits': 92.0, 'milk,football,apple': 173.0, 'the': 5.0, 'milk,apple': 169.0, 'a': 55.0, 'give': 150.0, 'longer': 177.0, 'maybe': 21.0, 'hungry': 140.0, 'apple,milk': 174.0, 'green': 68.0, 'wolves': 53.0, 'left': 85.0}
        #self.ivocab = {1: 'fred', 2: 'is', 3: 'either', 4: 'in', 5: 'the', 6: 'school', 7: 'or', 8: 'park', 9: '.', 10: 'mary', 11: 'went', 12: 'back', 13: 'to', 14: 'office', 15: 'yes', 16: 'bill', 17: 'kitchen', 18: 'moved', 19: 'cinema', 20: 'no', 21: 'maybe', 22: 'travelled', 23: 'julie', 24: 'bedroom', 25: 'journeyed', 26: 'bathroom', 27: 'after', 28: 'that', 29: 'she', 30: 'where', 31: 'daniel', 32: 'afterwards', 33: 'he', 34: 'hallway', 35: 'john', 36: 'following', 37: 'garden', 38: 'sandra', 39: 'then', 40: 'and', 41: 'they', 42: 'yesterday', 43: 'this', 44: 'morning', 45: 'was', 46: 'before', 47: 'afternoon', 48: 'evening', 49: 'mice', 50: 'are', 51: 'afraid', 52: 'of', 53: 'wolves', 54: 'gertrude', 55: 'a', 56: 'mouse', 57: 'cats', 58: 'sheep', 59: 'winona', 60: 'emily', 61: 'jessica', 62: 'wolf', 63: 'what', 64: 'cat', 65: 'lily', 66: 'frog', 67: 'bernhard', 68: 'green', 69: 'brian', 70: 'lion', 71: 'white', 72: 'julius', 73: 'swan', 74: 'greg', 75: 'color', 76: 'rhino', 77: 'gray', 78: 'yellow', 79: 'triangle', 80: 'above', 81: 'pink', 82: 'rectangle', 83: 'blue', 84: 'square', 85: 'left', 86: 'right', 87: 'below', 88: 'red', 89: 'sphere', 90: 'box', 91: 'chocolates', 92: 'fits', 93: 'inside', 94: 'chest', 95: 'bigger', 96: 'than', 97: 'suitcase', 98: 'container', 99: 'does', 100: 'fit', 101: 'chocolate', 102: 'east', 103: 'north', 104: 'west', 105: 'how', 106: 'do', 107: 'you', 108: 'go', 109: 'from', 110: 's,e', 111: 'n,w', 112: 'south', 113: 'n,n', 114: 's,w', 115: 'e,s', 116: 's,s', 117: 'e,n', 118: 'w,w', 119: 'e,e', 120: 'n,e', 121: 'w,s', 122: 'w,n', 123: 'sumit', 124: 'tired', 125: 'will', 126: 'why', 127: 'did', 128: 'grabbed', 129: 'pajamas', 130: 'there', 131: 'get', 132: 'yann', 133: 'bored', 134: 'jason', 135: 'thirsty', 136: 'got', 137: 'football', 138: 'antoine', 139: 'milk', 140: 'hungry', 141: 'took', 142: 'apple', 143: 'picked', 144: 'up', 145: 'dropped', 146: 'discarded', 147: 'put', 148: 'down', 149: 'gave', 150: 'give', 151: 'handed', 152: 'jeff', 153: 'who', 154: 'received', 155: 'Bill', 156: 'Fred', 157: 'Mary', 158: 'passed', 159: 'Jeff', 160: 'many', 161: 'objects', 162: 'carrying', 163: 'one', 164: 'none', 165: 'two', 166: 'three', 167: 'nothing', 168: 'football,apple', 169: 'milk,apple', 170: 'apple,football', 171: 'football,milk', 172: 'milk,football', 173: 'milk,football,apple', 174: 'apple,milk', 175: 'apple,milk,football', 176: 'apple,football,milk', 177: 'longer', 178: 'not', 179: 'football,apple,milk'}
        #"""
        
        self.train_input, self.train_q, self.train_answer, self.train_input_mask = self._process_input(babi_train_raw)
        self.test_input, self.test_q, self.test_answer, self.test_input_mask = self._process_input(babi_test_raw)
        self.vocab_size = len(self.vocab)

        self.input_var = T.imatrix('input_var')
        self.q_var = T.ivector('question_var')
        self.answer_var = T.iscalar('answer_var')
        self.input_mask_var = T.ivector('input_mask_var')
        
        self.attentions = []

        self.pe_matrix_in = self.pe_matrix(self.max_inp_sent_len)
        self.pe_matrix_q = self.pe_matrix(self.max_q_len)

            
        print "==> building input module"

        #positional encoder weights
        self.W_pe = nn_utils.normal_param(std=0.1, shape=(self.vocab_size, self.dim))

        #biGRU input fusion weights
        self.W_inp_res_in_fwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.sent_vector_size))
        self.W_inp_res_hid_fwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_res_fwd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_upd_in_fwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.sent_vector_size))
        self.W_inp_upd_hid_fwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_upd_fwd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_hid_in_fwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.sent_vector_size))
        self.W_inp_hid_hid_fwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_hid_fwd = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_inp_res_in_bwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.sent_vector_size))
        self.W_inp_res_hid_bwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_res_bwd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_upd_in_bwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.sent_vector_size))
        self.W_inp_upd_hid_bwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_upd_bwd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_hid_in_bwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.sent_vector_size))
        self.W_inp_hid_hid_bwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_hid_bwd = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        #self.V_f = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        #self.V_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))

        self.inp_sent_reps, _ = theano.scan(
                                fn=self.sum_pos_encodings_in,
                                sequences=self.input_var)

        self.inp_sent_reps_stacked = T.stacklists(self.inp_sent_reps)
        #self.inp_c = self.input_module_full(self.inp_sent_reps_stacked)

        self.inp_c = self.input_module_full(self.inp_sent_reps)

        self.q_q = self.sum_pos_encodings_q(self.q_var)
                
        print "==> creating parameters for memory module"
        self.W_mem_res_in = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, self.dim))
        self.W_mem_res_hid = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, self.dim))
        self.b_mem_res = nn_utils.constant_param(value=0.0, shape=(self.memory_hops, self.dim,))
        
        self.W_mem_upd_in = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, self.dim))
        self.W_mem_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, self.dim))
        self.b_mem_upd = nn_utils.constant_param(value=0.0, shape=(self.memory_hops, self.dim,))
        
        self.W_mem_hid_in = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, self.dim))
        self.W_mem_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, self.dim))
        self.b_mem_hid = nn_utils.constant_param(value=0.0, shape=(self.memory_hops, self.dim,))
        
        #self.W_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        #self.W_1 = nn_utils.normal_param(std=0.1, shape=(self.dim, 7 * self.dim + 0))
        self.W_1 = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, 4 * self.dim + 0))
        self.W_2 = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, 1, self.dim))
        self.b_1 = nn_utils.constant_param(value=0.0, shape=(self.memory_hops, self.dim,))
        self.b_2 = nn_utils.constant_param(value=0.0, shape=(self.memory_hops, 1,))


        print "==> building episodic memory module (fixed number of steps: %d)" % self.memory_hops
        memory = [self.q_q.copy()]
        for iter in range(1, self.memory_hops + 1):
            self.mem_weight_num = int(iter - 1)
            current_episode = self.new_episode(memory[iter - 1])
            memory.append(self.GRU_update(memory[iter - 1], current_episode,
                                          self.W_mem_res_in[self.mem_weight_num], self.W_mem_res_hid[self.mem_weight_num], self.b_mem_res[self.mem_weight_num], 
                                          self.W_mem_upd_in[self.mem_weight_num], self.W_mem_upd_hid[self.mem_weight_num], self.b_mem_upd[self.mem_weight_num],
                                          self.W_mem_hid_in[self.mem_weight_num], self.W_mem_hid_hid[self.mem_weight_num], self.b_mem_hid[self.mem_weight_num]))
        
        last_mem_raw = memory[-1].dimshuffle(('x', 0))
        
        net = layers.InputLayer(shape=(1, self.dim), input_var=last_mem_raw)
        if self.dropout > 0 and self.mode == 'train':
            net = layers.DropoutLayer(net, p=self.dropout)
        last_mem = layers.get_output(net)[0]
        
        print "==> building answer module"
        self.W_a = nn_utils.normal_param(std=0.1, shape=(self.vocab_size, self.dim))
        
        if self.answer_module == 'feedforward':
            self.prediction = nn_utils.softmax(T.dot(self.W_a, last_mem))
        
        elif self.answer_module == 'recurrent':
            self.W_ans_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            self.W_ans_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            self.W_ans_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
            def answer_step(prev_a, prev_y):
                a = self.GRU_update(prev_a, T.concatenate([prev_y, self.q_q]),
                                  self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                                  self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                                  self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid)
                
                y = nn_utils.softmax(T.dot(self.W_a, a))
                return [a, y]
            
            # add conditional ending?
            dummy = theano.shared(np.zeros((self.vocab_size, ), dtype=floatX))
            
            results, updates = theano.scan(fn=answer_step,
                outputs_info=[last_mem, T.zeros_like(dummy)],
                n_steps=1)
            self.prediction = results[1][-1]
        
        else:
            raise Exception("invalid answer_module")
        
        
        print "==> collecting all parameters"
        self.params = [self.W_pe,
                  self.W_inp_res_in_fwd, self.W_inp_res_hid_fwd, self.b_inp_res_fwd, 
                  self.W_inp_upd_in_fwd, self.W_inp_upd_hid_fwd, self.b_inp_upd_fwd,
                  self.W_inp_hid_in_fwd, self.W_inp_hid_hid_fwd, self.b_inp_hid_fwd,
                  self.W_inp_res_in_bwd, self.W_inp_res_hid_bwd, self.b_inp_res_bwd, 
                  self.W_inp_upd_in_bwd, self.W_inp_upd_hid_bwd, self.b_inp_upd_bwd,
                  self.W_inp_hid_in_bwd, self.W_inp_hid_hid_bwd, self.b_inp_hid_bwd, 
                  self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid, #self.W_b
                  self.W_1, self.W_2, self.b_1, self.b_2, self.W_a]

        if self.answer_module == 'recurrent':
            self.params = self.params + [self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                              self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                              self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid]
        
        
        print "==> building loss layer and computing updates"
        self.loss_ce = T.nnet.categorical_crossentropy(self.prediction.dimshuffle('x', 0), 
                                                       T.stack([self.answer_var]))[0]

        if self.l2 > 0:
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
        else:
            self.loss_l2 = 0
        
        self.loss = self.loss_ce + self.loss_l2
        
        #updates = lasagne.updates.adadelta(self.loss, self.params)
        updates = lasagne.updates.adam(self.loss, self.params)
        updates = lasagne.updates.adam(self.loss, self.params, learning_rate=0.0001, beta1=0.5) #from DCGAN paper
        #updates = lasagne.updates.adadelta(self.loss, self.params, learning_rate=0.0005)
        #updates = lasagne.updates.momentum(self.loss, self.params, learning_rate=0.0003)
        
        self.attentions = T.stack(self.attentions)
        if self.mode == 'train':
            print "==> compiling train_fn"
            self.train_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.input_mask_var], 
                                            outputs=[self.prediction, self.loss, self.attentions],
                                            updates=updates,
                                            on_unused_input='warn',
                                            allow_input_downcast=True)
        
        print "==> compiling test_fn"
        self.test_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.input_mask_var],
                                       outputs=[self.prediction, self.loss, self.attentions],
                                       on_unused_input='warn',
                                       allow_input_downcast=True)
    

    def pe_matrix(self, num_words):
        embedding_size = self.dim

        pe_matrix = np.ones((num_words, embedding_size))

        for j in range(num_words):
            for i in range(embedding_size):
                value = (i + 1. - (embedding_size + 1.) / 2.) * (j + 1. - (num_words + 1.) / 2.)
                pe_matrix[j,i] = float(value)
        pe_matrix = 1. + 4. * pe_matrix / (float(embedding_size) * num_words)

        return pe_matrix

    def sum_pos_encodings_in(self, statement):
        pe_matrix = self.pe_matrix_in
        pe_weights = pe_matrix * self.W_pe[statement]

        #'''
        if self.dropout_in > 0 and self.mode == 'train':
            pe_weights_d = pe_weights.dimshuffle(('x', 0, 1))
            net = layers.InputLayer(shape=(1, self.max_inp_sent_len, self.dim), input_var=pe_weights_d)
            net = layers.DropoutLayer(net, p=self.dropout_in)
            pe_weights = layers.get_output(net)[0]
        #'''

        pe_weights = T.cast(pe_weights, floatX)
        memories = T.sum(pe_weights, axis=0)
        return memories
        #return memories[-1]

    def sum_pos_encodings_q(self, statement):
        pe_matrix = self.pe_matrix_q
        pe_weights = pe_matrix * self.W_pe[statement]
        pe_weights = T.cast(pe_weights, floatX)
        memories = T.sum(pe_weights, axis=0)
        return memories
    
    def get_sentence_representation(self, statements):
        sent_rep, _ = theano.scan(fn = self.sum_pos_encodings,
            sequences = statements)
        return sent_rep

    def bi_GRU_fwd(self, x_fwd, prev_h):
        fwd_gru = self.GRU_update(prev_h, x_fwd, self.W_inp_res_in_fwd, self.W_inp_res_hid_fwd, self.b_inp_res_fwd, 
                                 self.W_inp_upd_in_fwd, self.W_inp_upd_hid_fwd, self.b_inp_upd_fwd,
                                 self.W_inp_hid_in_fwd, self.W_inp_hid_hid_fwd, self.b_inp_hid_fwd)
        '''
        if self.dropout_in > 0 and self.mode == 'train':
            fwd_gru_swap = fwd_gru.dimshuffle(('x', 0))
            net = layers.InputLayer(shape=(1, self.dim), input_var=fwd_gru_swap)
            net = layers.DropoutLayer(net, p=self.dropout_in)
            fwd_gru_d = layers.get_output(net)[0]
            fwd_gru = fwd_gru_d
        #'''
        return fwd_gru

    def bi_GRU_bwd(self, x_bwd, prev_h):
        bwd_gru = self.GRU_update(prev_h, x_bwd, self.W_inp_res_in_bwd, self.W_inp_res_hid_bwd, self.b_inp_res_bwd, 
                                 self.W_inp_upd_in_bwd, self.W_inp_upd_hid_bwd, self.b_inp_upd_bwd,
                                 self.W_inp_hid_in_bwd, self.W_inp_hid_hid_bwd, self.b_inp_hid_bwd)
        '''
        if self.dropout_in > 0 and self.mode == 'train':
            bwd_gru_swap = bwd_gru.dimshuffle(('x', 0))
            net = layers.InputLayer(shape=(1, self.dim), input_var=bwd_gru_swap)
            net = layers.DropoutLayer(net, p=self.dropout_in)
            bwd_gru_d = layers.get_output(net)[0]
            bwd_gru = bwd_gru_d
        #'''
        return bwd_gru

    def input_module_full(self, x):
        '''
        based on https://github.com/uyaseen/theano-recurrence/blob/master/model/gru.py
        based on Kyle_Kastner's comment: https://news.ycombinator.com/item?id=11237125
        '''
        x_fwd = x
        x_bwd = x[::-1]

        h_fwd_gru, _ = theano.scan(fn=self.bi_GRU_fwd, 
                    #sequences=self.inp_sent_reps,
                    sequences=x_fwd,
                    outputs_info=T.zeros_like(self.b_inp_hid_fwd))
                    #outputs_info=T.zeros_like(self.W_inp_hid_hid))

        h_bwd_gru, _ = theano.scan(fn=self.bi_GRU_bwd, 
                    #sequences=self.inp_sent_reps,
                    sequences=x_bwd,
                    outputs_info=T.zeros_like(self.b_inp_hid_bwd))
                    #outputs_info=T.zeros_like(self.W_inp_hid_hid))

        h_bwd_gru = h_bwd_gru[::-1]

        '''
        #axis=0 and no transposes is original that works
        ctx = T.concatenate([h_fwd_gru, h_bwd_gru], axis=0)
        ht = ctx
        #'''

        '''
        #axis=1 and transpose parts & whole also works
        ctx = T.concatenate([h_fwd_gru.T, h_bwd_gru.T], axis=1)
        ht = ctx.T
        #'''

        '''
        #weighted sum version
        #h_t = T.dot(h_fwd_gru, self.V_f) + T.dot(h_bwd_gru, self.V_b)
        '''

        h_t = h_bwd_gru + h_fwd_gru

        return h_t 

    def episode_compute_z(self, fi, prev_g, mem, q_q):
        #euclid square version
        z = T.concatenate([fi * q_q, fi * mem, (fi - q_q) ** 2, (fi - mem) ** 2])
        
        #T.abs_ version
        #z = T.concatenate([fi * q_q, fi * mem, T.abs_(fi - q_q), T.abs_(fi - mem)])

        l_1 = T.dot(self.W_1[self.mem_weight_num], z) + self.b_1[self.mem_weight_num]
        l_1 = T.tanh(l_1)
        l_2 = T.dot(self.W_2[self.mem_weight_num], l_1) + self.b_2[self.mem_weight_num]

        exp_l_2 = T.exp(l_2)

        return exp_l_2

    def episode_compute_g(self, z_i, z_all):
        G = z_i/(T.sum(z_all, axis=0))
        G = G[0]
        return G

    def episode_attend(self, x, g, h):
        r = T.nnet.sigmoid(T.dot(self.W_mem_res_in[self.mem_weight_num], x) + T.dot(self.W_mem_res_hid[self.mem_weight_num], h) + self.b_mem_res[self.mem_weight_num])
        _h = T.tanh(T.dot(self.W_mem_hid_in[self.mem_weight_num], x) + r * T.dot(self.W_mem_hid_hid[self.mem_weight_num], h) + self.b_mem_hid[self.mem_weight_num])
        #ht =  g * h + (1. - g) * _h
        ht =  g * _h + (1. - g) * h     #swapped version from paper that converges better for some reason
        return ht

    def episode_update(c_t, prev_m, q_q, W_t, b):
        m = T.nnet.relu(W_t[T.concatenate([prev_m, c_t, q_q])]+b)
        return m 
    
    def GRU_update(self, h, x, W_res_in, W_res_hid, b_res,
                         W_upd_in, W_upd_hid, b_upd,
                         W_hid_in, W_hid_hid, b_hid):
        """ mapping of our variables to symbols in DMN paper: 
        W_res_in = W^r
        W_res_hid = U^r
        b_res = b^r
        W_upd_in = W^z
        W_upd_hid = U^z
        b_upd = b^z
        W_hid_in = W
        W_hid_hid = U
        b_hid = b^h
        """
        z = T.nnet.sigmoid(T.dot(W_upd_in, x) + T.dot(W_upd_hid, h) + b_upd)
        r = T.nnet.sigmoid(T.dot(W_res_in, x) + T.dot(W_res_hid, h) + b_res)
        _h = T.tanh(T.dot(W_hid_in, x) + r * T.dot(W_hid_hid, h) + b_hid)
        #return z * h + (1. - z) * _h
        return z * _h + (1. - z) * h    #swapped version from paper that converges better for some reason
    
    
    def input_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                                     self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                                     self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid)
    
    def new_episode(self, mem):
        z, z_updates = theano.scan(fn=self.episode_compute_z,
            sequences=self.inp_c,
            non_sequences=[mem, self.q_q],
            outputs_info=T.zeros_like(self.b_2[0]))

        g, g_updates = theano.scan(fn=self.episode_compute_g,
            sequences=z,
            non_sequences=z,)
            
        #'''
        if (self.normalize_attention):
            g = nn_utils.softmax(g)
        #'''  

        self.attentions.append(g)

        e, e_updates = theano.scan(fn=self.episode_attend,
            sequences=[self.inp_c, g],
            outputs_info=T.zeros_like(self.inp_c[0]))
        
        return e[-1] 
    
    def save_params(self, file_name, epoch, **kwargs):
        with open(file_name, 'w') as save_file:
            pickle.dump(
                obj = {
                    'params' : [x.get_value() for x in self.params],
                    'epoch' : epoch, 
                    'gradient_value': (kwargs['gradient_value'] if 'gradient_value' in kwargs else 0)
                },
                file = save_file,
                protocol = -1
            )
    
    
    def load_state(self, file_name):
        print "==> loading state %s" % file_name
        with open(file_name, 'r') as load_file:
            dict = pickle.load(load_file)
            loaded_params = dict['params']
            for (x, y) in zip(self.params, loaded_params):
                x.set_value(y)

    def _process_input(self, data_raw):
        max_inp_sent_len = 0.
        max_inp_num_sents = 0.
        max_q_len = 0.
        self.max_fact_count = 0.
        for x in data_raw:

            #this splits it into sentences
            sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
            x["C"] = sent_detector.tokenize(x["C"])
            
            inp = []
            for i in range(len(x["C"])): 
                inp.append(x["C"][i].lower().split(' ')) 
                inp[i] = [w for w in inp[i] if len(w) > 0]
                max_inp_sent_len = max(max_inp_sent_len, len(inp[i]))

            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]
            
            if (self.input_mask_mode == 'word'):
                fact_count = len(inp)
            elif (self.input_mask_mode == 'sentence'):
                fact_count = len([0 for w in inp if w == '.'])
            else:
                raise Exception("unknown input_mask_mode")
            
            max_inp_num_sents = max(max_inp_num_sents, len(inp))
            max_q_len = max(max_q_len, len(q))
            self.max_fact_count = max(self.max_fact_count, fact_count)

        questions = []
        inputs = []
        answers = []
        input_masks = []
        for x in data_raw:

            #sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
            #x["C"] = sent_detector.tokenize(x["C"])
            
            inp = []
            for i in range(len(x["C"])): 
                inp.append(x["C"][i].lower().split(' ')) 
                inp[i] = [w for w in inp[i] if len(w) > 0]
            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]
            
            inp_vector = []
            for i in range(len(inp)):            
                
                inp_i = [utils.process_word(word = inp[i][w], 
                                            word2vec = self.word2vec, 
                                            vocab = self.vocab, 
                                            ivocab = self.ivocab, 
                                            word_vector_size = self.word_vector_size, 
                                            to_return = "index") for w in range(len(inp[i]))]

                inp_vector.append(inp_i)

                #is this still needed?
                while(len(inp_vector[i]) < max_inp_sent_len):
                    inp_vector[i].append(0)
                #'''

                '''
                #VERSION FOR IF YOU SCRAP SENTENCE ENCODER 
                while(len(inp_vector[i]) < 80):
                    inp_vector[i].append(0)
                #'''
            
            #'''
            #is this still needed?
            while (len(inp_vector) < max_inp_num_sents):
                inp_vector.append([0] * (max_inp_sent_len))
            #'''

            '''
            #VERSION FOR IF YOU SCRAP SENTENCE ENCODER 
            while (len(inp_vector) < max_inp_num_sents):
                inp_vector.append([0] * (80))
            #'''
                                        
            q_vector = [utils.process_word(word = q[w], 
                                        word2vec = self.word2vec, 
                                        vocab = self.vocab, 
                                        ivocab = self.ivocab, 
                                        word_vector_size = self.word_vector_size, 
                                        to_return = "index") for w in range(len(q))]
                                        
            '''
            q_vector = [utils.process_word(word = w, 
                                        word2vec = self.word2vec, 
                                        vocab = self.vocab, 
                                        ivocab = self.ivocab, 
                                        word_vector_size = self.word_vector_size, 
                                        to_return = "word2vec") for w in q]
                                        '''

            while(len(q_vector) < max_q_len):
                q_vector.append(0)


            inputs.append(inp_vector)
            questions.append(q_vector)

            answers.append(utils.process_word(word = x["A"], 
                                            word2vec = self.word2vec, 
                                            vocab = self.vocab, 
                                            ivocab = self.ivocab, 
                                            word_vector_size = self.word_vector_size, 
                                            to_return = "index"))

            # NOTE: here we assume the answer is one word! 
            if self.input_mask_mode == 'word':
                input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.float32)) 
            elif self.input_mask_mode == 'sentence': 
                input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.float32)) 
            else:
                raise Exception("invalid input_mask_mode")

        '''#THIS TURN ON ONE HOT ENCODER
        #inputs = utils.one_hot_encoding(s, self.sent_vector_size, embedding_size)
        inputs = utils.one_hot_encoding_trip(inputs, vocab_size, max_inp_sent_len)
        questions = utils.one_hot_encoding_doub(questions, vocab_size, max_q_len)   
        #'''

        inputs = np.array(inputs).astype(floatX)
        questions = np.array(questions).astype(floatX)

        self.max_inp_sent_len = max_inp_sent_len
        self.max_q_len = max_q_len

        return inputs, questions, answers, input_masks
    
    def get_batches_per_epoch(self, mode):
        if (mode == 'train'):
            return len(self.train_input)
        elif (mode == 'test'):
            return len(self.test_input)
        else:
            raise Exception("unknown mode")
   
    def shuffle_train_set(self):
        print "==> Shuffling the train set"
        combined = zip(self.train_input, self.train_q, self.train_answer, self.train_input_mask)
        random.shuffle(combined)
        self.train_input, self.train_q, self.train_answer, self.train_input_mask = zip(*combined)    
    
    def step(self, batch_index, mode):
        if mode == "train" and self.mode == "test":
            raise Exception("Cannot train during test mode")
        
        if mode == "train":
            theano_fn = self.train_fn 
            inputs = self.train_input
            qs = self.train_q
            answers = self.train_answer
            input_masks = self.train_input_mask
        elif mode == "test":    
            theano_fn = self.test_fn 
            inputs = self.test_input
            qs = self.test_q
            answers = self.test_answer
            input_masks = self.test_input_mask
        else:
            raise Exception("Invalid mode")
            
        inp = inputs[batch_index]
        q = qs[batch_index]
        ans = answers[batch_index]
        input_mask = input_masks[batch_index]

        ret = theano_fn(inp, q, ans, input_mask)
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        
        return {"prediction": np.array([ret[0]]),
                "answers": np.array([ans]),
                "current_loss": ret[1],
                "skipped": 0,
                "log": "pn: %.3f" % param_norm,
                "inp": np.array([inp]),
                "q" : np.array([q]),
                "probabilities": np.array([ret[0]]),
                "attentions": np.array([ret[2]]),
                }
                            
    def predict(self, data):
        # data is an array of objects like {"Q": "question", "C": "sentence ."}
        data[0]["A"] = "."
        print "==> predicting:", data
        inputs, questions, answers, input_masks = self._process_input(data)
        probabilities, loss, attentions = self.test_fn(inputs[0], questions[0], answers[0], input_masks[0])
        return probabilities, attentions
