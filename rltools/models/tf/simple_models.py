import prettytensor as pt
import tensorflow as tf

def softmax_mlp(input_data, output_size, layers=[32,32], activation=tf.nn.relu):

    net = pt.wrap(input_data).sequential()
    for l_size in layers:
        net.fully_connected(l_size, activation_fn=activation) 
    output_layer, _ = net.softmax_classifier(output_size)

    return output_layer


def softmax_conv(input_data, output_size, kernels=[3,3], depths=[16,32], strides=[1,1], fc_layers=[32], activation=tf.nn.relu):

    net = pt.wrap(input_data).sequential()
    count = 0
    for k, d, s in zip(kernels, depths, strides):
        net.conv2d(k, d, stride=s, activation_fn=activation, name="conv_" + str(count))
        count += 1
    net.flatten()
    for i, fc in enumerate(fc_layers):
        net.fully_connected(fc, activation_fn=activation, name="fc_" + str(i))
    output_layer, _ = net.softmax_classifier(output_size)

    return output_layer


def softmax_lstm(input_data, output_size, timesteps, fc_layers=[32], lstm_layers=[32,32], activation=tf.nn.relu):

    net = pt.wrap(input_data).sequential()
    for l_size in fc_layers:
        net.fully_connected(l_size, activation_fn=activation) 
    net.cleave_sequence(timesteps)
    for l_size in lstm_layers:
        net.sequence_lstm(l_size)
    net.squash_sequence()
    output_layer, _ = net.softmax_classifier(output_size)

    return output_layer



def dqn(input_data, output_size, kernels=[8,4,3], depths=[32,64,64], strides=[4,2,1], fc_layers=[512], activation=tf.nn.relu):
    
    net = pt.wrap(input_data).sequential()
    for k, d, s in zip(kernels, depths, strides):
        net.conv2d(k, d, stride=s, activation_fn=activation)
    net.flatten()
    for fc in fc_layers:
        net.fully_connected(fc, activation_fn=activation)
    output_layer = net.fully_connected(output_size)
    return output_layer

