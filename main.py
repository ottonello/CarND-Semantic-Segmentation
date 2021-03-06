import sys
import os.path
import time
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np

KEEP_PROB = 0.5
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
MODEL_VERSION = 3
PROCESS_VIDEO = True
WRITE_SUMMARY = True

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

def kernel_initializer():
    return tf.truncated_normal_initializer(stddev=0.01)

def kernel_regularizer():
    return tf.contrib.layers.l2_regularizer(1e-2)

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
    # Layer 3 = 40x144x256
    # Layer 4 = 20x72x512
    # Layer 7 = 5x18x4096

    upsample_kernel_size = (2,2)
    upsample_stride_size = (2,2)

    # 1x1 conv layer 7, 5x18xnum_classes
    layer_7_conv = tf.layers.conv2d_transpose(vgg_layer7_out, 64, (1,1), (1,1), name='1x1_vgg_7',kernel_initializer=kernel_initializer())
    # 1x1 conv layer 4, 10x36xnum_classes
    layer_4_conv = tf.layers.conv2d_transpose(vgg_layer4_out, 64, (1,1), (1,1), name='1x1_vgg_4',kernel_initializer=kernel_initializer())
    # 1x1 conv layer 3, 20x72xnum_classes
    layer_3_conv = tf.layers.conv2d_transpose(vgg_layer3_out, 64, (1,1), (1,1), name='1x1_vgg_3',kernel_initializer=kernel_initializer())

    # Decoder layer 1, 10x36xnum_classes
    decoder_layer_1 = tf.layers.conv2d_transpose(layer_7_conv, 64, upsample_kernel_size, upsample_stride_size, name='decoder_1',kernel_initializer=kernel_initializer())

    skip1 = tf.add(decoder_layer_1, layer_4_conv)

    # Decoder layer 2, 20x72xnum_classes
    decoder_layer_2 = tf.layers.conv2d_transpose(skip1, 64, upsample_kernel_size, upsample_stride_size, name='decoder_2', kernel_initializer=kernel_initializer())

    skip2 = tf.add(decoder_layer_2, layer_3_conv)

    # Decoder layer 5, 160x576xnum_classes
    decoder_layer_4 = tf.layers.conv2d_transpose(skip2, 32, (2,2), (2,2), padding='same', name='decoder_4',kernel_initializer=kernel_initializer())

    # Decoder layer 5, 160x576xnum_classes
    decoder_layer_5 = tf.layers.conv2d_transpose(decoder_layer_4, num_classes, (8,8), (4,4), padding='same', name='decoder_5',kernel_initializer=kernel_initializer())

    return decoder_layer_5
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the leartning rate
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

def augment_op(images):
    def augment_pipeline(img):
        rand_flip = tf.image.random_flip_left_right(img)
        rand_bright = tf.image.random_brightness(rand_flip, max_delta=50.0)
        rand_contrast = tf.image.random_contrast(rand_bright, lower=0.0, upper=0.5)
        max_val = tf.minimum(rand_bright, 255.0)
        min_val = tf.maximum(max_val, 0.0)

        return min_val
    return tf.map_fn(lambda img: augment_pipeline(img), images)
# Test will fail with no data
tests.test_augmentation(augment_op, False, True, 15)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, augment, images):
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
            # augmented = sess.run([augment], feed_dict={
            #         images: image
            # })
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

    print("Start training ---")
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)

        images = tf.placeholder(tf.float32, shape=(None, 160, 576, 3))
        augment = augment_op(images)
        input_image, keep_prob, l3_out, l4_out, l7_out = load_vgg(sess, vgg_path)        
        last_layer = layers(l3_out, l4_out, l7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, num_classes)


        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        loss_history = train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate, augment, images)

        # Save inference data using helper.save_inference_samples
        # Make folder for current run
        output_dir = os.path.join(runs_dir, str(time.time()))
        helper.save_run_loss_and_parameters(output_dir, loss_history, KEEP_PROB, BATCH_SIZE, EPOCHS, MODEL_VERSION)
        helper.save_inference_samples(output_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        if WRITE_SUMMARY:
            writer = tf.summary.FileWriter(os.path.join(output_dir, 'logs'), sess.graph)            
            writer.close()

        # Apply the trained model to a video
        if PROCESS_VIDEO:
            helper.process_video(sess, logits, keep_prob, input_image, image_shape, output_dir, 'sample.mp4')
            helper.process_video(sess, logits, keep_prob, input_image, image_shape, output_dir, 'sample2.mp4')
            helper.process_video(sess, logits, keep_prob, input_image, image_shape, output_dir, 'sample3.mp4')

if __name__ == '__main__':
    run()
