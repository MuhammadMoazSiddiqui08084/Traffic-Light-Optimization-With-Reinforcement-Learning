import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys
import random

from tensorflow import keras
from keras import layers
from keras import losses
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model
from keras.saving import register_keras_serializable
from keras.layers import Lambda

# -------------------------------
# Base DQN (your original TrainModel)
# -------------------------------
class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)


    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(learning_rate=self._learning_rate))

        return model
    

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state, verbose=0)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states, verbose=0)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        os.makedirs(path, exist_ok=True)
        self._model.save(os.path.join(path, 'trained_model.h5'))
        try:
            plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)
        except Exception:
            # plotting is optional; do not fail if graphviz not present
            pass


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size



# -------------------------------
# Double DQN (target network)
# -------------------------------
class DoubleDQNTrainModel(TrainModel):
    """
    Double DQN: maintains a separate target network for stable Q-value targets.
    Interface preserved: predict_one, predict_batch, train_batch, save_model, batch_size property.
    """
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, target_update_freq=10):
        # build main model first
        super().__init__(num_layers, width, batch_size, learning_rate, input_dim, output_dim)
        # create target model and copy weights
        self._target_model = self._build_model(num_layers, width)
        self._target_update_freq = target_update_freq
        self._update_counter = 0
        self.update_target_model()


    def update_target_model(self):
        """
        Copy weights from main network to target network
        """
        self._target_model.set_weights(self._model.get_weights())


    # keep predict_one / predict_batch from base (they use self._model)
    # provide predict_target to allow training loop to use target network
    def predict_target(self, states):
        return self._target_model.predict(states, verbose=0)


    def train_batch(self, states, q_sa):
        """
        Train the main network using provided targets (q_sa).
        After training, update target network periodically.
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)
        self._update_counter += 1
        if self._update_counter % self._target_update_freq == 0:
            self.update_target_model()


# -------------------------------
# Dueling DQN
# -------------------------------
@register_keras_serializable()
def dueling_combine(inputs):
    v, a = inputs
    a_mean = tf.reduce_mean(a, axis=1, keepdims=True)
    return v + (a - a_mean)

class DuelingDQNTrainModel(TrainModel):
    """
    Dueling DQN: separate streams for state value and advantage
    """
    def _build_model(self, num_layers, width):
        inputs = layers.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)

        # Value stream
        V = layers.Dense(1, activation='linear')(x)
        # Advantage stream
        A = layers.Dense(self._output_dim, activation='linear')(x)
        # Combine into Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))

        outputs = Lambda(dueling_combine)([V, A])

        model = keras.Model(inputs=inputs, outputs=outputs, name='DuelingDQN')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(learning_rate=self._learning_rate))

        return model


# -------------------------------
# Tabular Q-Learning
# -------------------------------
class TabularTrainModel:
    """
    Classic Tabular Q-Learning (for small state-action spaces)
    Q-table is stored as a 2D numpy array [state_index, action_index]
    This class intentionally uses the same method names used by NN models where applicable:
      - predict_one(state_index) -> returns 2D array [1, num_actions]
      - train_one(state_index, action, reward, next_state_index)
      - save_model(path), load_model(path)
    """
    def __init__(self, num_states, num_actions, learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((self.num_states, self.num_actions))
        # Add these so training code doesn't crash
        self.batch_size = None
        self.update_target_frequency = None

    # compatibility methods
    def predict_one(self, state_index):
        # Returns Q-values for a given discrete state index (2D array for compatibility)
        return self.q_table[int(state_index):int(state_index)+1]


    def choose_action(self, state_index):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return int(np.argmax(self.q_table[int(state_index)]))


    def train_one(self, state_index, action, reward, next_state_index):
        state_index = int(state_index)
        next_state_index = int(next_state_index)
        best_next = np.max(self.q_table[next_state_index])
        td_target = reward + self.gamma * best_next
        self.q_table[state_index, action] += self.lr * (td_target - self.q_table[state_index, action])
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "q_table.npy"), self.q_table)


    def load_model(self, path):
        file_path = os.path.join(path, "q_table.npy")
        if os.path.isfile(file_path):
            self.q_table = np.load(file_path)
        else:
            sys.exit("Q-table not found")


# -------------------------------
# TestModel
# -------------------------------
class TestModel:
    """
    Unified TestModel for NN and Tabular Q-learning
    - if is_tabular=True loads 'q_table.npy' from model_path
    - otherwise loads Keras model 'trained_model.h5'
    """
    def __init__(self, input_dim, model_path, is_tabular=True):
        self._input_dim = input_dim
        self._is_tabular = is_tabular

        if is_tabular:
            self.q_table_path = os.path.join(model_path, "q_table.npy")
            if os.path.isfile(self.q_table_path):
                self.q_table = np.load(self.q_table_path)
                self.num_states = len(self.q_table)
            else:
                sys.exit("Q-table not found")
        else:
            model_file_path = os.path.join(model_path, 'trained_model.h5')
            if os.path.isfile(model_file_path):
                self._model = load_model(model_file_path)
            else:
                sys.exit("Model not found")


    def predict_one(self, state):
        if self._is_tabular:
            # state must be mapped to a discrete index outside this class
            state_index = int(state)
            return self.q_table[state_index:state_index+1]
        else:
            state = np.reshape(state, [1, self._input_dim])
            return self._model.predict(state, verbose=0)


    @property
    def input_dim(self):
        return self._input_dim
