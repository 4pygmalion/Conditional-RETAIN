# -*- coding: utf-8 -*-
# written by Ho Heon kim
# last update : 2020.08.26

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras import Model, regularizers


class RETAIN(object):

    def __init__(self, config):
        '''
        Parameters
        ----------
        config: Dict
            key: 'n_features', 'steps', 'hidden_units'

        '''
        self.n_features = config['n_features']
        self.steps = config['steps']
        self.hidden_units = config['hidden_units']
        

    def build_model(self,
                    name='base_model',
                    l2_penalty=0.01,
                    beta_activation='tanh',
                    Bidirectional_activation='relu',
                    LSTM_initializer='glorot_uniform',
                    kernel_initializer='random_uniform',
                    predict='regression',
                    ):
        '''
        Build model with tensorflow.keras framework
        
        Parameters
        ----------
        use_x_aux: bool. whether use conditional RNN



        LSTM_initializer: str

            (default: glorot_uniform)
            - 'he_normal'
            - 'random_uniform'
            - 'Constant'
            - 'Zeros'
            - 'Ones'
            - 'RandomNormal'

            see. https://keras.io/initializers/


        kernel_initializer: str

            (default: glorot_uniform)
            - 'he_normal'
            - 'random_uniform'
            - 'Constant'
            - 'Zeros'
            - 'Ones'
            - 'RandomNormal'

            see. https://keras.io/initializers/

        beta_activation: str.
            tanh
            tan
            linear

        l2_penalty: float

        predict: str.
            (default: regession)
            - 'classification': for binary classification
            - 'regression': for regression model

        Returns
        -------
        model: tensorflow.keras.Model
        '''


        # Definition
        def reshape(data):
            '''Reshape the context vectors to 3D vector'''  #
            return K.reshape(x=data, shape=(-1, self.n_features))  # backend.shape(data)[0]

        alpha = Bidirectional(LSTM(self.hidden_units ,
                                   activation=Bidirectional_activation,
                                   implementation=2,
                                   return_sequences=True,
                                   kernel_initializer=LSTM_initializer,
                                   activity_regularizer=regularizers.l2(l2_penalty)), name='alpha')

        alpha_dense = Dense(1, activity_regularizer=regularizers.l2(l2_penalty))

        beta = Bidirectional(LSTM(self.hidden_units ,
                                  activation=Bidirectional_activation,
                                  implementation=2,
                                  return_sequences=True,
                                  kernel_initializer=LSTM_initializer,
                                  activity_regularizer=regularizers.l2(l2_penalty)), name='beta')

        beta_dense = Dense(self.n_features, activation=beta_activation)

        # Regression:
        if predict == 'regression':
            output_layer = Dense(1,
                                 kernel_regularizer=regularizers.l2(l2_penalty),
                                 kernel_initializer=kernel_initializer, name='output')
        # with classification
        else:
            output_layer = Dense(1, activation='sigmoid', name='output')


        # -- main : operation --
        x_input = Input(shape=(self.steps, self.n_features), name='X')  # feature

        # 2-1. alpha
        alpha_out = alpha(x_input)
        alpha_out = TimeDistributed(alpha_dense, name='alpha_dense')(alpha_out)
        alpha_out = Softmax(axis=1, name='alpha_softmax')(alpha_out)  # 논문 본문에 alpha1, alph2, alph3..을 의미

        # 2-2. beta
        beta_out = beta(x_input)
        beta_out = TimeDistributed(beta_dense, name='beta_dense')(beta_out)  # 논문 내 beta1 ,beta2, beta3을 의미.

        # 3. Context vector
        c_t = Multiply()([alpha_out, beta_out, x_input])
        c_t = Lambda(lambda x: K.sum(x, axis=1), name='lamdaSum')(c_t)

        # Output layer
        output_final = output_layer(c_t)

        # Model
        model = Model(x_input, output_final, name=name)

        return model



class ConditionalRETAIN(RETAIN):
    '''
    Conditional RNN type of RETAIN
    '''

    def __init__(self, config):
        ''' Conditional RNN type for RETAIN

        Parameters
        ----------
        config: Dict
            key: 'n_features', 'steps', 'hidden_units', 'n_auxs'
            
            
        Attribution
        -----------
        self.n_features = config['n_features']
        self.steps = config['steps']
        self.hidden_units = config['hidden_units']

        '''
        RETAIN.__init__(self, config)
        self.n_axus = config['n_auxs']
        
        
    def build_model(self,
                    name='base_model',
                    problem='regression',
                    l2_penalty=0.25,
                    beta_activation='tanh',
                    Bidirectional_activation='relu',
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='random_uniform',
                    other_initializer='zeros'
                    ):
        
        
        
        def reshape(data):
            '''Reshape the context vectors to 3D vector''' # 
            return K.reshape(x=data, shape=(-1, x_time_vect_size)) # backend.shape(data)[0]

        alpha = Bidirectional(LSTM(self.hidden_units,
                                   activation=Bidirectional_activation, 
                                   implementation=2, 
                                   return_sequences=True,
                                   kernel_initializer=kernel_initializer,
                                   activity_regularizer=regularizers.l2(l2_penalty)), name='alpha') 

        alpha_dense = Dense(1, activity_regularizer=regularizers.l2(l2_penalty))

        beta = Bidirectional(LSTM(self.hidden_units,
                                  activation=Bidirectional_activation, 
                                  implementation=2, 
                                  return_sequences=True,
                                  kernel_initializer=kernel_initializer,
                                  activity_regularizer=regularizers.l2(l2_penalty)), name='beta') 

        beta_dense = Dense(self.n_features, activation=beta_activation)

        # Regression:
        if problem == 'regression':
            output_layer = Dense(1, kernel_regularizer=regularizers.l2(l2_penalty), kernel_initializer=other_initializer, name='output')
        else:
            output_layer = Dense(1, activation='sigmoid', name='output')

        # Model define
        x_input = Input(shape=(self.steps, self.n_features), name='X') # feature
        x_input_fixed = Input(shape=(self.n_axus), name='x_input_fixed')

        # 2-1. alpha
        alpha_out = alpha(x_input)
        alpha_out = TimeDistributed(alpha_dense, name='alpha_dense')(alpha_out) 
        alpha_out = Softmax(axis=1, name='alpha_softmax')(alpha_out) # 논문 본문에 alpha1, alph2, alph3..을 의미

        # 2-2. beta
        beta_out = beta(x_input)
        beta_out = TimeDistributed(beta_dense, name='beta_dense')(beta_out) # 논문 내 beta1 ,beta2, beta3을 의미.

        # 3. Context vector
        c_t = Multiply()([alpha_out, beta_out, x_input])
        context = Lambda(lambda x: K.sum(x, axis=1) , name='lamdaSum')(c_t) 

        # Output layer
        c_concat = concatenate([context, x_input_fixed])
        output_final = output_layer(c_concat)     

        # Model
        model = Model([x_input, x_input_fixed] , output_final, name=name)

        return model

    





