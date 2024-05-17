import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # INFO and WARNING messages are not printed
import numpy as np
import random
from collections import deque
from tensorflow import keras
from keras.models import Model, clone_model
from keras.layers import Input, Conv2D, Flatten, Dense, concatenate
from keras.optimizers import RMSprop


class DQNAgent:
    def __init__(self, discount_rate, exploration_rate, learning_rate, memory_capacity, action_size, batch_size, update_rate):
        self._discount_rate = discount_rate
        self._exploration_rate = exploration_rate
        self._learning_rate = learning_rate
        self._replay_memory = deque(maxlen=memory_capacity)
        self._action_size = action_size
        self._batch_size = batch_size
        self._update_rate = update_rate
        self._model = self._build_model()
        self._target_model = clone_model(self._model)
        self._target_model.set_weights(self._model.get_weights())
    
    def _build_model(self):
        '''
        Build and compile the neural network for Deep-Q learning.
        '''
        # create the stacked sub-network with the position matrix
        position_input = Input(shape=(12, 12, 1))
        position_L1 = Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation='relu')(position_input)
        position_L2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), activation='relu')(position_L1)
        position_L3 = Flatten()(position_L2)

        # create the stacked sub-network with the speed matrix
        speed_input = Input(shape=(12, 12, 1))
        speed_L1 = Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation='relu')(speed_input)
        speed_L2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), activation='relu')(speed_L1)
        speed_L3 = Flatten()(speed_L2)

        # flatten the traffic signal state
        traffic_signal_state_input = Input(shape=(2, 1))
        traffic_signal_state_L3 = Flatten()(traffic_signal_state_input)

        # concatenate flattened outputs of the two stacked sub-networks with the traffic signal state
        input_L3 = concatenate([position_L3, speed_L3, traffic_signal_state_L3])

        # create two fully connected layers that feed into the final output layer
        L3 = Dense(128, activation='relu')(input_L3)
        L4 = Dense(64, activation='relu')(L3)
        outputs = Dense(2, activation='linear')(L4)

        # create and compile model using RMSprop and MSE loss
        model = Model(inputs=[position_input, speed_input, traffic_signal_state_input], outputs=outputs)
        model.compile(optimizer=RMSprop(learning_rate=self._learning_rate), loss='mse')

        return model
    
    def add_experience(self, state, action, reward, next_state, done):
        '''
        Add an experience to the replay memory.
        '''
        self._replay_memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        '''
        Choose an action to take given the current state.
        '''
        if np.random.rand() <= self._exploration_rate:
            return random.randrange(self._action_size)
        action_values = self._model.predict(state, verbose=0)
        
        return np.argmax(action_values[0])

    def replay_experience(self):
        '''
        Samples a mini-batch of experiences from the replay memory and use them to train the Q-network.
        '''
        if len(self._replay_memory) < self._batch_size:
            return
        
        minibatch = random.sample(self._replay_memory, self._batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self._discount_rate * np.amax(self._target_model.predict(next_state, verbose=0)[0])
            target_f = self._model.predict(state, verbose=0)
            target_f[0][action] = target
            self._model.fit(state, target_f, epochs=1, verbose=0)
        self.soft_update_target_network()

    def soft_update_target_network(self):
        '''
        Performs a soft update on the weights of the target model.
        '''
        model_weights = self._model.get_weights()
        target_model_weights = self._target_model.get_weights()
        self._target_model.set_weights([self._update_rate * w + (1 - self._update_rate) * tw for w, tw in zip(model_weights, target_model_weights)])

    def load_model_weights(self, model_file_name):
        '''
        Load the weights for the model.
        '''
        self._model.load_weights(model_file_name)
    
    def load_target_model_weights(self, target_model_file_name):
        '''
        Load the weights for the target model.
        '''
        self._target_model.load_weights(target_model_file_name)

    def save_model_weigths(self, model_file_name):
        '''
        Save the current weights for the model.
        '''
        self._model.save_weights(model_file_name)

    def save_target_model_weights(self, target_model_file_name):
        '''
        Save the current weights for the target model.
        '''
        self._target_model.save_weights(target_model_file_name)
