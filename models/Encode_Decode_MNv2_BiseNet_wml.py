from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import functools
import numpy as np


def DepthwiseSeparableConvBlock(inputs, n_filters, kernel_size=[3, 3],scope="depthwise_conv"):

    with tf.variable_scope(scope,"depthwise_conv",[inputs]):
        net = slim.separable_convolution2d(inputs, num_outputs=None, depth_multiplier=1, kernel_size=[3, 3],
                                           activation_fn=None)
        net = slim.batch_norm(net, fused=True)
        net = tf.nn.relu(net)
        net = slim.conv2d(net, n_filters, kernel_size=[1, 1], activation_fn=None)
        net = slim.batch_norm(net, fused=True)
        net = tf.nn.relu(net)

	return net

def bottleneck(input,output,stride,k=6,scope="bottleneck"):
    """
    build inverted bottleneck
    :param input:
    :param output:
    :param stride:
    :param k: expansion ration
    :return:
    """
    with tf.variable_scope(scope,"bottleneck",):
        in_channel = input.get_shape()[-1]
        #conv1
        net = slim.conv2d(input,in_channel*k,[1,1],stride=stride,activation_fn=None)
        net = slim.batch_norm(net)
        net = tf.nn.relu6(net)
        #conv2--depthwise conv
        net = DepthwiseSeparableConvBlock(net,in_channel*k)
        #conv3
        net = slim.conv2d(net,output,[1,1],stride=1,activation_fn=None)
        net = slim.batch_norm(net)

        if stride == 1 :

            if in_channel == output :
                shortcut = input
            else:
                shortcut = slim.conv2d(input,output,kernel_size=[3,3])

            result= tf.add(shortcut,net)
        else:

            result = net

    return result

def mobilenet_v2(inputs,is_training=True,output_collections=None,scope = "mobilenet_v2"):

    medium_result = {}
    with tf.variable_scope(scope,"mobilenet_v2",[inputs]) as sc :

        end_points_collection = sc.name + "end_points"

        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'is_training': is_training}
        with slim.arg_scope([slim.conv2d],outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm],**batch_norm_params):
                net = inputs

                net = slim.conv2d(net, num_outputs=32,kernel_size=[3,3],stride=2,activation_fn=None)
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)
                #[expansion_ration,channel,numbers,stride]
                bottlenecks = [[1,16,1,1],[6,24,2,2],[6,32,3,2],[6,64,4,2],[6,96,3,1],[6,160,3,2],[6,320,1,1]]

                for index,item in enumerate(bottlenecks):

                    expansion_ration, channel, numbers, stride = item
                    net = bottleneck(net,channel,stride,expansion_ration,scope="bottleneck_%s"%(index+1))
                    medium_result["bottleneck_%s"%(index+1)] = net
                    # print(net)
                net = slim.conv2d(net,num_outputs=1280,kernel_size=[3,3]) #(1, 7, 7, 1280)
                # net = slim.avg_pool2d(net,kernel_size=[7,7]) #(1, 1, 1, 1280)

    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    end_points.update(medium_result)

    return net,end_points



def attention_refinement_module(input_feature,is_training,scope):

    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training}

    with variable_scope.variable_scope(scope, 'arm', [input_feature]):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):

            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                depth = 32
                _, height, width, _ = input_feature.shape
                height = int(height)
                width = int(width)

                new_depth_f = slim.conv2d(input_feature, depth, 1, activation_fn=None, normalizer_fn=None)

                image_feature = slim.avg_pool2d(new_depth_f, [height, width], [height, width],padding='VALID')
                image_feature = slim.conv2d(image_feature, depth, 1, activation_fn=None)
                image_feature = tf.image.resize_bilinear(mage_feature, [height, width], align_corners=True)
                image_feature.set_shape([None, height, width, depth])
                image_feature = tf.sigmoid(image_feature)

                result = tf.multiply(new_depth_f, image_feature)

                return result

def feature_fusion_module(spatial_feature,context_feature,is_training,scope):

    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training}

    with variable_scope.variable_scope(scope, 'feature_fusion_module', [spatial_feature, context_feature]):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):

            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                depth = 32

                _, height, width, _ = spatial_feature.shape
                height = int(height)
                width = int(width)

                concated_f = tf.concat([spatial_feature, context_feature], axis=3)
                conv_f = slim.conv2d(concated_f, depth, [3, 3], padding='SAME',activation_fn=None)  # 64
                conv_f = slim.batch_norm(conv_f)
                conv_f = tf.nn.relu(conv_f)

                image_feature = slim.avg_pool2d(conv_f, [height, width], [height, width],padding='VALID')
                image_feature = slim.conv2d(image_feature, depth, 1, normalizer_fn=None)
                image_feature = slim.conv2d(image_feature, depth, 1, activation_fn=None, normalizer_fn=None)
                image_feature = tf.image.resize_bilinear(image_feature, [height, width], align_corners=True)
                image_feature.set_shape([None, height, width, depth])
                image_feature = tf.sigmoid(image_feature)

                mul_f = tf.multiply(conv_f, image_feature)

                add_f = tf.add(mul_f, conv_f)

                result = add_f

                return result

def spatial_path():

    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training}

    with variable_scope.variable_scope(scope, 'spatialPath', [image]):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                spatial_feature = slim.conv2d(image, 32, [3, 3], stride=2, padding='SAME')
                spatial_feature = slim.conv2d(spatial_feature, 32, [3, 3], stride=2, padding='SAME')
                spatial_feature = slim.conv2d(spatial_feature, 32, [3, 3], stride=2, padding='SAME')
                return spatial_feature

# def context_path():
#     batch_norm_params = {
#         'decay': 0.997,
#         'epsilon': 1e-5,
#         'scale': True,
#         'updates_collections': tf.GraphKeys.UPDATE_OPS,
#         'is_training': is_training}


def bisenet(input,endpoints,scope="BiSeNet"):

    #spatial path
    spatial_feature = spatial_path(input,scope)

    # context path
    feature_16 = endpoints['bottleneck_3']
    feature_32 = endpoints['bottleneck_4']
    feature_last = endpoints['bottleneck_7']
    arm16 = attention_refinement_module(feature_16)
    arm32 = attention_refinement_module(feature_32)
    concat_1 = tf.concat(feature_last, arm32)
    concat_1_up = tf.image.resize_bilinear(concat_1, [concat_1.get_shape()[1] * 2, concat_1.get_shape()[2] * 2],
                                           align_corners=True)
    concat_2 = tf.concat(concat_1_up, arm16)
    context_feature = concat_2

    # ffm
    ffm_feature = feature_fusion_module(spatial_feature,context_feature)
    result = ffm_feature

    return result

def build_encoder_decoder(inputs,num_classes,preset_model="Encoder-Decoder_MNv2_BiseNet",dropout_p=0.5,scope=None):
    """
    build encode-decode structure , encode --> mobilenetv2 , decode --> bisenet.
    :param inputs:
    :param num_classes:
    :param preset_model:
    :param dropout_p:
    :param scope:
    :return:
    """

    #####################
    # Downsampling path #
    #####################

    encode,endpoints = mobilenet_v2(input)

    #####################
    # Upsampling path #
    #####################


    decode = bisenet(input=endpoints['bottleneck_3'],endpoints=endpoints)


    #####################
    #      Softmax      #
    #####################

    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
    return net


#test
input = np.ones([1,224,224,3],dtype=np.float32)
# output,endpoints = mobilenet_v2(input)
output = build_encoder_decoder(input,2)
print np.shape(output)