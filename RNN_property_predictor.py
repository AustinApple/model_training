from __future__ import print_function

import numpy as np
import pickle
# cannot sure which version of tensorflow is, so this is a workaround
try:
       import tensorflow.compat.v1 as tf 
       tf.compat.v1.disable_v2_behavior()
except:
       import tensorflow as tf

class Model(object):

    def __init__(self, seqlen_x, dim_x, dim_y, dim_z=100, dim_h=250, n_hidden=3, batch_size=200, char_set=[' ']):

        self.seqlen_x, self.dim_x, self.dim_y, self.dim_z, self.dim_h, self.n_hidden, self.batch_size = seqlen_x, dim_x, dim_y, dim_z, dim_h, n_hidden, batch_size
        
        self.char_to_int = dict((c,i) for i,c in enumerate(char_set))
        self.int_to_char = dict((i,c) for i,c in enumerate(char_set))
        
        self.G = tf.Graph()   
        self.G.as_default()

        ## variables for labeled data
        self.x_L = tf.placeholder(tf.float32, [None, self.seqlen_x, self.dim_x], name="x_L")          
        self.y_L = tf.placeholder(tf.float32, [None, self.dim_y], name="y_L")

        ## functions for labeled data
        self.y_L_mu = self._rnnpredictor(self.x_L, self.dim_h, self.dim_y, reuse = False)
        
        ## variables for unlabeled data
        self.x_U = tf.placeholder(tf.float32, [None, self.seqlen_x, self.dim_x], name='x_U')
        
        ## functions for unlabeled data
        self.y_U_mu = self._rnnpredictor(self.x_U, self.dim_h, self.dim_y, reuse = True)

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        

    def train(self, trnX_L, trnY_L, epochs):
        objYpred_MAE = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(self.y_L,self.y_L_mu)), 1))
        train_op = tf.train.AdamOptimizer().minimize(objYpred_MAE)
        
        batch_size_L=int(self.batch_size)
        n_batch=int(len(trnX_L)/batch_size_L)
        
        self.session.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            print('######### epoch : '+ str(epoch)+'#############')
            [trnX_L, trnY_L]=self._permutation([trnX_L, trnY_L])
            
            for i in range(n_batch):
                start_L=i*batch_size_L
                end_L=start_L+batch_size_L
                trn_res = self.session.run([train_op, objYpred_MAE],
                                           feed_dict = {self.x_L: trnX_L[start_L:end_L], 
                                                        self.y_L: trnY_L[start_L:end_L]})
            print(trn_res[1])
           
            # this is for the remain data
            trn_res = self.session.run([train_op, objYpred_MAE,summary_objYpred_MSE],
                                        feed_dict = {self.x_L: trnX_L[end_L:], 
                                                     self.y_L: trnY_L[end_L:]})
            
            


                # writer.add_summary(trn_res[5], epoch * n_batch + i)             
            # writer.add_summary(trn_res[2], epoch)
            # val_res = []
            # for i in range(10):
            #     start_L=i*batch_size_val_L
            #     end_L=start_L+batch_size_val_L
            #     val_res.append(self.session.run([objYpred_MAE],
            #                     feed_dict = {self.x_L: valX_L[start_L:end_L], 
            #                                  self.y_L: valY_L[start_L:end_L]}))
                
            # writer.add_summary(summary_val, epoch * int(10)+ i) 
            # writer.add_summary(summary_val, epoch)
            
            
            # val_res=np.mean(val_res,axis=0)
            # print('---', ['Validation', 'cost_val', val_res[0]])
            
            # val_log[epoch] = val_res[0] 

           
       
    def predict(self, x_input):

        return self.session.run(self.y_U_mu, feed_dict = {self.x_U: x_input})
    
    
    def reload(self, model_name):    
        with self.session.as_default() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph(model_name+'.meta')
            saver.restore(sess,model_name)

    def _permutation(self, set):

        permid=np.random.permutation(len(set[0]))
        for i in range(len(set)):
            set[i]=set[i][permid]

        return set


    def _rnnpredictor(self, x, dim_h, dim_y, reuse=False):

        with tf.variable_scope('rnnpredictor', reuse=reuse):

            cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(dim_h) for _ in range(self.n_hidden)])
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(dim_h) for _ in range(self.n_hidden)])
            init_state_fw = cell_fw.zero_state(tf.shape(x)[0], tf.float32)
            init_state_bw = cell_bw.zero_state(tf.shape(x)[0], tf.float32)
            
            _, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw)
            res = tf.layers.dense(tf.concat([final_state[0][-1],final_state[1][-1]], 1), dim_y)
            
            
        return res


