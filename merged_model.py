from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN_Caps(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='mnist',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None,
         caps_on_g=True, caps_on_d=True,
         train_g_twice=False):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optbional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # New options
        self.caps_on_d = caps_on_d
        self.caps_on_g = caps_on_g
        self.train_g_twice = train_g_twice

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        if self.dataset_name == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
            self.c_dim = self.data_X[0].shape[-1]
        else:
            self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
            imreadImg = imread(self.data[0])
            if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1

        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
          tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(
          tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        # Model inits
        if self.caps_on_d:
            self.D, self.D_logits = self.discriminator(True, inputs, reuse=False)
        else:
            self.D, self.D_logits = self.discriminator(False, inputs, reuse=False)
        if self.caps_on_g:
            self.G = self.generator(True, self.z, self.z_dim)
            self.sampler = self.sampler(True, self.z, self.z_dim)
            G_reshaped = tf.reshape(self.G, [-1,28,28])
            G_reshaped = tf.expand_dims(G_reshaped, 3)
            self.D_, self.D_logits_ = self.discriminator(self.caps_on_d, G_reshaped, reuse=True)
        else:
            self.G = self.generator(False, self.z, self.z_dim)
            self.sampler = self.sampler(False, self.z, self.z_dim)
            self.D_, self.D_logits_ = self.discriminator(self.caps_on_d, self.G, reuse=True)


        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
          self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

        if config.dataset == 'mnist':
            sample_inputs = self.data_X[0:self.sample_num]
            sample_labels = self.data_y[0:self.sample_num]
        else:
            sample_files = self.data[0:self.sample_num]
            sample = [
                get_image(sample_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for sample_file in sample_files]
            if (self.grayscale):
                sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_epoch = 0
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
            start_epoch = int(counter / 282)
        else:
            print(" [!] Load failed...")

        for epoch in xrange(start_epoch, config.epoch):
            if config.dataset == 'mnist':
                batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
            else:
                self.data = glob(os.path.join(
                "./data", config.dataset, self.input_fname_pattern))
                batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                if config.dataset == 'mnist':
                    batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                else:
                    batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch = [
                        get_image(batch_file,
                                input_height=self.input_height,
                                input_width=self.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                crop=self.crop,
                                grayscale=self.grayscale) for batch_file in batch_files]
                    if self.grayscale:
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    else:
                        batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                if config.dataset == 'mnist':
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={
                          self.inputs: batch_images,
                          self.z: batch_z,
                          self.y:batch_labels,
                        })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={
                          self.z: batch_z,
                          self.y:batch_labels,
                        })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    if self.train_g_twice:
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                          feed_dict={ self.z: batch_z, self.y:batch_labels })
                        self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({
                        self.z: batch_z,
                        self.y:batch_labels
                    })
                    errD_real = self.d_loss_real.eval({
                        self.inputs: batch_images,
                        self.y:batch_labels
                    })
                    errG = self.g_loss.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                else:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.inputs: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    if self.train_g_twice:
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                          feed_dict={ self.z: batch_z })
                        self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
                    errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
                    errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run(
                          [self.sampler, self.d_loss, self.g_loss],
                          feed_dict={
                              self.z: sample_z,
                              self.inputs: sample_inputs,
                              self.y:sample_labels,
                          }
                        )
                        save_images(samples, image_manifold_size(samples.shape[0]),
                              './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    else:
                        try:
                            samples, d_loss, g_loss = self.sess.run(
                                [self.sampler, self.d_loss, self.g_loss],
                                feed_dict={
                                    self.z: sample_z,
                                    self.inputs: sample_inputs,
                                },
                            )
                            save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                        except:
                            print("one pic error!...")

                if np.mod(counter, 282) == 0:
                    self.save(config.checkpoint_dir, counter)


    def discriminator(self, caps_on_d, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if caps_on_d:
                # Capsule version
                image.get_shape()

                #Carefully check the code below
                # First convolutional and pool layers
                # These search for 256 different 5 x 5 pixel features
                #We’ll start off by passing the image through a convolutional layer.
                #First, we create our weight and bias variables through tf.get_variable.
                #Our first weight matrix (or filter) will be of size 5x5 and will have a output depth of 256.
                #It will be randomly initialized from a normal distribution.
                d_w1 = tf.get_variable('d_w1', [9, 9, 1, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
                #tf.constant_init generates tensors with constant values.
                d_b1 = tf.get_variable('d_b1', [256], initializer=tf.constant_initializer(0))

                #tf.nn.conv2d() is the Tensorflow’s function for a common convolution.
                #It takes in 4 arguments. The first is the input volume (our 28 x 28 x 1 image in this case).
                #The next argument is the filter/weight matrix. Finally, you can also change the stride and
                #padding of the convolution. Those two values affect the dimensions of the output volume.
                #"SAME" tries to pad evenly left and right, but if the amount of columns to be added is odd,
                #it will add the extra column to the right,
                #strides = [batch, height, width, channels]
                d1 = tf.nn.conv2d(input=image, filter=d_w1, strides=[1, 1, 1, 1], padding='VALID')
                #d1 = tf.contrib.layers.conv2d(inputs=x_image, num_outputs=256,weights_initializer = d_w1,
                #                                        kernel_size=8, stride=1,padding='SAME')
                #add the bias
                d1 = d1 + d_b1

                #here comes the capsNet
                with tf.variable_scope('PrimaryCaps_layer'):
                	primaryCaps = CapsConv(num_units=8, with_routing=False)
                	caps1 = primaryCaps(self.sess, d1, num_outputs=256, batch_size=self.batch_size, kernel_size=9, stride=2)
                	#assert caps1.get_shape() == [128, 1152, 8, 1]

                # DigitCaps layer, [batch_size, 10, 16, 1]
                with tf.variable_scope('DigitCaps_layer'):
                	digitCaps = CapsConv(num_units=16, with_routing=True)
                	caps2 = digitCaps(self.sess, caps1, num_outputs=10, batch_size=self.batch_size)

                #and then followed by a series of fully connected layers.
                # First fully connected layer
                d_w3 = tf.get_variable('d_w3', [16*10, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
                d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
                d3 = tf.reshape(caps2, [-1, 16*10])
                d3 = tf.matmul(d3, d_w3)
                d3 = d3 + d_b3
                d3 = tf.nn.relu(d3)

                #The last fully-connected layer holds the output, such as the class scores.
                # Second fully connected layer
                d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
                d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))

                #At the end of the network, we do a final matrix multiply and
                #return the activation value.
                #For those of you comfortable with CNNs, this is just a simple binary classifier. Nothing fancy.
                # Final layer
                d4 = tf.matmul(d3, d_w4) + d_b4
                # d4 dimensions: batch_size x 1

                return tf.nn.sigmoid(d4), d4

            else:
                # CNN version
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

                return tf.nn.sigmoid(h4), h4

    def generator(self, caps_on_g, z, z_dim):
        with tf.variable_scope("generator") as scope:
            if caps_on_g:
                # Caps version
                z = tf.truncated_normal([self.batch_size, z_dim], mean=0, stddev=1, name='z')
                #first deconv block
                g_w1 = tf.get_variable('g_w1', [z_dim,102400 ], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
                g_b1 = tf.get_variable('g_b1', [102400], initializer=tf.truncated_normal_initializer(stddev=0.02))
                g1 = tf.matmul(z, g_w1) + g_b1
                g1 = tf.reshape(g1, [-1, 20, 20, 256])
                g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
                g1 = tf.nn.relu(g1)
                print(g1.get_shape())
                #CapsNet Implementation

                # Primary Capsules layer, return [batch_size, 1152, 8, 1]
                with tf.variable_scope('PrimaryCaps_layerGen'):
                    primaryCaps = CapsConv2(num_units=8, with_routing=False)
                    caps1Gen = primaryCaps(self.sess, g1, num_outputs=256, batch_size=self.batch_size, kernel_size=9, stride=2)
                    #assert caps1.get_shape() == [128, 1152, 8, 1]

                # DigitCaps layer, [batch_size, 10, 16, 1]
                with tf.variable_scope('DigitCaps_layerGen'):
                    digitCaps = CapsConv2(num_units=16, with_routing=True)
                    caps2Gen = digitCaps(self.sess, caps1Gen, num_outputs=10, batch_size=self.batch_size)


                # Decoder structure in Fig. 2
                # 1. Do masking, how:
                with tf.variable_scope('Masking'):
                    # a). calc ||v_c||, then do softmax(||v_c||)
                    # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
                    v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2Gen),
                    									  axis=2, keep_dims=True) + 1e-9)
                    softmax_v = tf.nn.softmax(v_length, dim=1)
                    assert softmax_v.get_shape() == [64, 10, 1, 1]

                    # b). pick out the index of max softmax val of the 10 caps
                    # [batch_size, 10, 1, 1] => [batch_size] (index)
                    argmax_idx = tf.to_int32(tf.argmax(softmax_v, axis=1))
                    assert argmax_idx.get_shape() == [64, 1, 1]
                    argmax_idx = tf.reshape(argmax_idx, shape=(64, ))

                    # Method 1.
                    if not True:
                        # c). indexing
                        # It's not easy to understand the indexing process with argmax_idx
                        # as we are 3-dim animal
                        masked_v = []
                        for batch_size in range(64):
                        	v = caps2Gen[batch_size][argmax_idx[batch_size], :]
                        	masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                        masked_v = tf.concat(masked_v, axis=0)
                        assert masked_v.get_shape() == [64, 1, 16, 1]
                    # Method 2. masking with true label, default mode
                    else:
                        # masked_v = tf.matmul(tf.squeeze(caps2), tf.reshape(Y, (-1, 10, 1)), transpose_a=True)
                        masked_v = tf.multiply(tf.squeeze(caps2Gen), tf.reshape(np.random.uniform(0.0,9.0,size=10).astype(np.float32), (-1, 10, 1)))
                        v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2Gen), axis=2, keep_dims=True) + 1e-9)


                # 2. Reconstruct the MNIST images with 3 FC layers
                # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
                with tf.variable_scope('Decoder'):
                    vector_j = tf.reshape(caps2Gen, shape=(64, -1))
                    fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
                    assert fc1.get_shape() == [64, 512]
                    fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
                    assert fc2.get_shape() == [64, 1024]
                    decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

                return tf.reshape(decoded, shape=(64, 28, 28, 1))

            else:
                # DN version
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

                return tf.nn.tanh(h4)


    def sampler(self, caps_on_g, z, z_dim):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            if caps_on_g:
                # Caps version
                z = tf.truncated_normal([self.batch_size, z_dim], mean=0, stddev=1, name='z')
                #first deconv block
                g_w1 = tf.get_variable('g_w1', [z_dim,102400 ], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
                g_b1 = tf.get_variable('g_b1', [102400], initializer=tf.truncated_normal_initializer(stddev=0.02))
                g1 = tf.matmul(z, g_w1) + g_b1
                g1 = tf.reshape(g1, [-1, 20, 20, 256])
                g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
                g1 = tf.nn.relu(g1)
                print(g1.get_shape())

                #CapsNet Implementation

                # Primary Capsules layer, return [batch_size, 1152, 8, 1]
                with tf.variable_scope('PrimaryCaps_layerGen'):
                    primaryCaps = CapsConv2(num_units=8, with_routing=False)
                    caps1Gen = primaryCaps(self.sess, g1, num_outputs=256, batch_size=self.batch_size, kernel_size=9, stride=2)
                    #assert caps1.get_shape() == [128, 1152, 8, 1]

                # DigitCaps layer, [batch_size, 10, 16, 1]
                with tf.variable_scope('DigitCaps_layerGen'):
                    digitCaps = CapsConv2(num_units=16, with_routing=True)
                    caps2Gen = digitCaps(self.sess, caps1Gen, num_outputs=10, batch_size=self.batch_size)


                # Decoder structure in Fig. 2
                # 1. Do masking, how:
                with tf.variable_scope('Masking'):
                    # a). calc ||v_c||, then do softmax(||v_c||)
                    # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
                    v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2Gen),
                    									  axis=2, keep_dims=True) + 1e-9)
                    softmax_v = tf.nn.softmax(v_length, dim=1)
                    assert softmax_v.get_shape() == [64, 10, 1, 1]

                    # b). pick out the index of max softmax val of the 10 caps
                    # [batch_size, 10, 1, 1] => [batch_size] (index)
                    argmax_idx = tf.to_int32(tf.argmax(softmax_v, axis=1))
                    assert argmax_idx.get_shape() == [64, 1, 1]
                    argmax_idx = tf.reshape(argmax_idx, shape=(64, ))

                    # Method 1.
                    if not True:
                        # c). indexing
                        # It's not easy to understand the indexing process with argmax_idx
                        # as we are 3-dim animal
                        masked_v = []
                        for batch_size in range(64):
                            v = caps2Gen[batch_size][argmax_idx[batch_size], :]
                            masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                        masked_v = tf.concat(masked_v, axis=0)
                        assert masked_v.get_shape() == [64, 1, 16, 1]
                    # Method 2. masking with true label, default mode
                    else:
                    	# masked_v = tf.matmul(tf.squeeze(caps2), tf.reshape(Y, (-1, 10, 1)), transpose_a=True)
                        masked_v = tf.multiply(tf.squeeze(caps2Gen), tf.reshape(np.random.uniform(0.0,9.0,size=10).astype(np.float32), (-1, 10, 1)))
                        v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2Gen), axis=2, keep_dims=True) + 1e-9)


                    # 2. Reconstructe the MNIST images with 3 FC layers
                    # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
                with tf.variable_scope('Decoder'):
                    vector_j = tf.reshape(caps2Gen, shape=(64, -1))
                    fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
                    assert fc1.get_shape() == [64, 512]
                    fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
                    assert fc2.get_shape() == [64, 1024]
                    decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

                return tf.reshape(decoded, shape=(64, 28, 28, 1))

            else:
                # DN version
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
                    [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)

        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0

        return X/255.,y_vec

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
