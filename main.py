import sys
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

KEEP_PROB = 0.5
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MODEL_VERSION = 1

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()
    return graph.get_tensor_by_name(vgg_input_tensor_name), \
            graph.get_tensor_by_name(vgg_keep_prob_tensor_name), \
            graph.get_tensor_by_name(vgg_layer3_out_tensor_name), \
            graph.get_tensor_by_name(vgg_layer4_out_tensor_name), \
            graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Input = 160x576x3
    # Layer 1 = 160x576x64
    # Layer 2 = 80x288x128
    # Layer 3 = 40x144x256
    # Layer 4 = 20x72x512
    # Layer 5 = 10x36x512
    # Layer 6 = 5x18x512
    # Layer 7 = 5x18x4096
    # Layer 7 output is already a convolution, we can implement the decoder stage directly

    # TODO add skip layers
    upsample_kernel_size = (2,2)
    upsample_stride_size = (2,2)
    # Decoder layer 1, 5x18xnum_classes
    decoder_layer_1 = tf.layers.conv2d_transpose(vgg_layer7_out, num_classes, (1,1), (1,1))

    # Decoder layer 2, 10x36xnum_classes
    decoder_layer_2 = tf.layers.conv2d_transpose(decoder_layer_1, num_classes, upsample_kernel_size, upsample_stride_size)
    
    # Decoder layer 3, 20x72xnum_classes
    decoder_layer_3 = tf.layers.conv2d_transpose(decoder_layer_2, num_classes, upsample_kernel_size, upsample_stride_size)

    # Decoder layer 4, 40x144xnum_classes
    decoder_layer_4 = tf.layers.conv2d_transpose(decoder_layer_3, num_classes, upsample_kernel_size, upsample_stride_size)

    # Decoder layer 5, 80x288xnum_classes
    decoder_layer_5 = tf.layers.conv2d_transpose(decoder_layer_4, num_classes, upsample_kernel_size, upsample_stride_size)

    # Decoder layer 6, 160x576xnum_classes
    decoder_layer_6 = tf.layers.conv2d_transpose(decoder_layer_5, num_classes, upsample_kernel_size, upsample_stride_size)


    return decoder_layer_6
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    global LEARNING_RATE
    global KEEP_PROB
    sample=0
    loss_history = []
    for epoch in range(epochs):
        for image, image_c in get_batches_fn(batch_size):
            _,loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image: image,
                correct_label: image_c,
                keep_prob: KEEP_PROB,
                learning_rate: LEARNING_RATE
            })
            sample = sample + batch_size
        print('Epoch {} of {} - Loss: {}'.format(epoch, epochs, loss))
        loss_history.append(loss)
    return loss_history
tests.test_train_nn(train_nn)

def run():
    global EPOCHS, KEEP_PROB, BATCH_SIZE

    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)

        input_image, keep_prob, l3_out, l4_out, l7_out = load_vgg(sess, vgg_path)
        last_layer = layers(l3_out, l4_out, l7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        loss_history = train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        output_dir = helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_run_loss_and_parameters(output_dir, loss_history, KEEP_PROB, BATCH_SIZE, EPOCHS, MODEL_VERSION)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
