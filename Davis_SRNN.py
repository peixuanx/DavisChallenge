import tensorflow as tf
from tensorflow.contrib import rnn

class SRNN:

    def __init__(self):
        self.batch_size = 1
        self.edge_names = ['mask','opticalFlow']
        self.node_names = ['image']
        self.inputs = {}

    def inference(self, video, num_edges=2, num_units, num_classes=2):
        num_frames = tf.shape(video)[0]
        edgeRNN = {}
        nodeRNN = {}
        states = {}
        outputs = {}
        node_inputs = {}
        node_outputs = []
        #weights = {'out' : tf.Variable(tf.random_normal([num_units*num_frames,num_classes]))}
        #biases = {'out' : tf.Variable(tf.random_normal([num_classes]))}

        self.inputs['mask'] = video[:,:,:,3]
        self.inputs['opticalFlow'] = video[:,:,:,4:7]
        for edge in edge_names:
            edgeRNN[edge] = rnn.BasicLSTMCell(num_units, forget_bias=1.0)
            states[edge] = edgeRNN[edge].zero_state(self.batch_size,tf.float32)
        for node in node_names:
            nodeRNN[node] = rnn.BasicLSTMCell(num_units, forget_bias=1.0)
            states[node] = nodeRNN[node].zero_state(self.batch_size,tf.float32)

        with tf.variable_scope("SRNN"):
            for time_step in range(num_frames):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
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
        final_output = tf.concat(node_outputs, 0, name='output_lastCells')
        final_states = states

        #targets = tf.placeholder(tf.float32, shape=(None,num_classes), name='targets')
        #logits = tf.matmul(final_output, weights['out'], name="logits") + biases['out']
        #with tf.name_scope('cross_entropy'):
        #    self.cost = tf.reduce_mean(
        #        tf.nn.softmax_cross_entropy_with_logits(logits, targets))
