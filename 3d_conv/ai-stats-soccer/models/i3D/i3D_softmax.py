"""Script for creating Inception-v1 Inflated 3D ConvNet model with softmax output"""

from .i3D_base import Inception_Inflated3d
from keras.layers import Reshape
from keras.layers import Dense
from keras.models import Model

# Possible weights are:
# 'rgb_kinetics_only', 'flow_kinetics_only', 'rgb_imagenet_and_kinetics', 'flow_imagenet_and_kinetics'
# If using flow models dimension has to be changed for dim=(24,224,224,2)

def import_model(classes = 11, dim=(24,224,224,3), weight_name='rgb_imagenet_and_kinetics', dropout_prob = 0):
    model_no_top = Inception_Inflated3d(weights=weight_name, input_shape=dim, include_top=False, dropout_prob=dropout_prob)
    x = Reshape((int(model_no_top.output.shape[1]*model_no_top.output.shape[-1]),))(model_no_top.output)
    dense1 = Dense(512, activation='relu')(x)
    output = Dense(classes, activation='softmax')(dense1)
    model = Model(inputs=[model_no_top.input], outputs=[output])
    return model
