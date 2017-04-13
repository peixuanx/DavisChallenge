import tensorflow as tf
from tensorflow.contrib import rnn

class SRNN:

    def __init__(self):
        self.batch_size = 1
        self.edge_names = ['mask','opticalFlow']
        self.node_names = ['image']
        self.inputs = {}

    def inference(self, video, num_units, frame_seq, num_edges=2, num_classes=2):
   
        # clear memory when new video comes in 
        if frame_seq==0:
            self.edgeRNN = {}
            self.nodeRNN = {}
            self.states = {}
            self.outputs = {}
            self.node_inputs = {}
            self.node_outputs = []

            for edge in self.edge_names:
                self.edgeRNN[edge] = rnn.BasicLSTMCell(num_units, forget_bias=1.0)
                self.states[edge] = self.edgeRNN[edge].zero_state(self.batch_size,tf.float32)
            for node in self.node_names:
                self.nodeRNN[node] = rnn.BasicLSTMCell(num_units, forget_bias=1.0)
                self.states[node] = self.nodeRNN[node].zero_state(self.batch_size,tf.float32)

        # memeory
        self.inputs['mask'] = video[:,:,:,3]
        self.inputs['opticalFlow'] = video[:,:,:,4:7]

        with tf.variable_scope("SRNN"):
            if frame_seq > 0: tf.get_variable_scope().reuse_variables()
            for edge in self.edge_names:
                inputs = self.inputs[edge][:,0,:]
                state = self.states[edge]
                scope = 'lstm_'+edge
                self.outputs[edge], self.states[edge] = self.edgeRNN[edge]( tf.reshape(inputs,[-1,num_units]), state, scope=scope)

            self.node_inputs['image'] = tf.concat([self.outputs['mask'], self.outputs['opticalFlow'], tf.reshape(video[:,:,:,:3],[-1,num_units])], axis=1)
            for node in self.node_names:
                inputs = self.node_inputs[node]
                state = self.states[node]
                scope = 'lstm_'+node
                self.outputs[node], self.states[node] = self.nodeRNN[node](inputs, state, scope=scope)
            self.node_outputs.append(self.outputs['image'])

        '''
        with tf.variable_scope("SRNN"):
            for time_step in range(num_frames):
                if frame_seq > 0: tf.get_variable_scope().reuse_variables()
                for edge in self.edge_names:
                    inputs = self.inputs[edge][:,time_step,:]
                    state = states[edge]
                    scope = 'lstm_'+edge
                    outputs[edge], states[edge] = edgesRNN[edge](inputs, state, scope=scope)

                node_inputs['image'] = tf.concat([outputs['mask'], outputs['opticalFlow'], video[:,:,:,:3]], 3)
                for node in nodes_names:
                    inputs = node_inputs[node]
                    state = states[node]
                    scope = 'lstm_'+node
                    outputs[node], states[node] = nodesRNN[node](inputs, state, scope=scope)
                node_outputs.append(outputs['image'])
        '''

        final_output = tf.concat(self.node_outputs, 0, name='output_lastCells')
        #final_states = states
        return final_output
        #targets = tf.placeholder(tf.float32, shape=(None,num_classes), name='targets')
        #logits = tf.matmul(final_output, weights['out'], name="logits") + biases['out']
        #with tf.name_scope('cross_entropy'):
        #    self.cost = tf.reduce_mean(
        #        tf.nn.softmax_cross_entropy_with_logits(logits, targets))
