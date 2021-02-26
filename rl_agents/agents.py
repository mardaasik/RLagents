import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.optimizers import Adam
import numpy as np
import json
import keras.backend as K
import tensorflow_probability as tfp
from keras import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam

#import buffer to store transitions
from rl_agents.buffers.buffer import Buffer

tf.compat.v1.enable_eager_execution()


class DeepQNetwork():
    def __init__(self):
        self.__train_counter = 0

    def get_train_counter(self):
        return self.__train_counter

    def set_params(self, learning_rate, gamma, n_actions,
                   epsilon, batch_size, input_dims, epsilon_decay=0.999, epsilon_end=0.01,
                   buffer_size=1000, network_params=[256, 256], target_network_update_steps=1000):

        self.__action_space = [i for i in range(n_actions)]
        self.__n_actions = n_actions
        self.__learning_rate = learning_rate
        self.__gamma = gamma
        self.__epsilon = epsilon
        self.__batch_size = batch_size
        self.__inputs_dims = input_dims
        self.__epsilon_decay = epsilon_decay
        self.__epsilon_end = epsilon_end
        self.__buffer_size = buffer_size
        self.__network_params = network_params
        self.__target_network_update_steps = target_network_update_steps

        # create buffer
        self.__buffer = Buffer(input_dims, n_actions, capacity=buffer_size)

        # settings for network and initiation
        self.__network = self.__create_model(learning_rate, n_actions,
                                             input_dims, network_params=self.__network_params)

        # create target network
        self.__target_network = self.__create_model(learning_rate, n_actions,
                                                    input_dims, network_params=self.__network_params)

        # copy weights
        self.target_network_update()

    def target_network_update(self):
        # copy weights
        self.__target_network.set_weights(self.__network.get_weights())

    def store_experience(self, state, action, reward, new_state, done):
        self.__buffer.store_transition(state, action, new_state, reward, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random_sample()
        if rand < self.__epsilon:
            action = np.random.choice(self.__action_space)
        else:
            action = np.argmax(self.__network.predict(state))

        return action

    # chooses the action without randomness
    def choose_greedy_action(self, state):
        state = state[np.newaxis, :]
        return np.argmax(self.__network.predict(state))


    # name changed from "learn" to "train"
    def train(self):
        # if there is no enough samples, then don't learn
        if self.__buffer.get_buffer_counter() < self.__batch_size:
            return

        #sample from buffer
        states, actions, next_states, rewards, dones = self.__buffer.sample(self.__batch_size)

        q_values_current = self.__network.predict(states)
        q_values_next = self.__target_network.predict(next_states)

        #apply DQN formula
        for _ in range(self.__batch_size):
            if dones[_]:
                q_values_current[_, int(actions[_])] = rewards[_]
            else:
                q_values_current[_, int(actions[_])] = rewards[_] + self.__gamma * np.max(q_values_next[_])

        #train
        self.__network.train_on_batch(states, q_values_current)

        # reassign epsilon after each learning
        self.__reassign_epsilon()

        #update target network
        if self.__train_counter > 0 and self.__train_counter % self.__target_network_update_steps == 0:
            self.target_network_update()

        #increase train counter
        self.__train_counter += 1


    def __reassign_epsilon(self):
        self.__epsilon *= self.__epsilon_decay
        self.__epsilon = max(self.__epsilon, self.__epsilon_end)

    def get_epsilon(self):
        return self.__epsilon

    def __create_model(self, learning_rate, n_actions, input_dims, network_params):
        model = keras.Sequential()
        model.add(Input(shape=(input_dims,)))
        for layer_neurons in network_params:
            model.add(Dense(layer_neurons, activation="relu"))
        model.add(Dense(n_actions, activation="linear"))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        return model

    def save_model(self, path):
        """
        Saves the agent parameters, networks and buffer
        - path/agent_params.json -> agent parameters
        - path/network -> network
        - path/buffer/ -> buffer
        :param path:
        :return:
        """

        # save network
        self.__network.save("{0}/network".format(path))

        #save the agent parameters
        params = {
            "train_counter":self.__train_counter,
            "action_space": self.__n_actions,
            "input_dims": self.__inputs_dims,
            "learning_rate": self.__learning_rate,
            "gamma": self.__gamma,
            "epsilon": self.__epsilon,
            "batch_size": self.__batch_size,
            "epsilon_decay": self.__epsilon_decay,
            "epsilon_end": self.__epsilon_end,
            "buffer_size": self.__buffer_size,
            "network_params": self.__network_params,
            "target_network_update_steps":self.__target_network_update_steps
        }

        with open("{0}/agent_params.json".format(path),"w") as f:
            json.dump(params, f)

        # save buffer
        self.__buffer.save("{0}/".format(path))

    def load_model(self, path):
        """
        Loads agent parameters, network and buffer
        :param path:
        :return:
        """

        # load agent parameters
        with open("{0}/agent_params.json".format(path),"r") as f:
            params = json.load(f)

            self.__action_space = [i for i in range(params["action_space"])]
            self.__train_counter = params["train_counter"]
            self.__n_actions = params["action_space"]
            self.__learning_rate = params["learning_rate"]
            self.__gamma = params["gamma"]
            self.__epsilon = params["epsilon"]
            self.__batch_size = params["batch_size"]
            self.__inputs_dims = params["input_dims"]
            self.__epsilon_decay = params["epsilon_decay"]
            self.__epsilon_end = params["epsilon_end"]
            self.__buffer_size = params["buffer_size"]
            self.__network_params = params["network_params"]
            self.__target_network_update_steps = params["target_network_update_steps"]

        # load network
        self.__network = keras.models.load_model("{0}/network".format(path))
        self.__target_network = keras.models.load_model("{0}/network".format(path))

        # create buffer
        self.__buffer = Buffer(self.__inputs_dims, self.__n_actions, capacity=self.__buffer_size)

        # load buffer
        self.__buffer.load("{0}/".format(path))


class REINFORCE():
    def __init__(self):
        self.__train_counter = 0

        # initialize memory as seperate lists
        self.__state_memory = []
        self.__action_memory = []
        self.__reward_memory = []

    def set_params(self, gamma, learning_rate, n_actions, input_dims, network_params):
        self.__gamma = gamma
        self.__learning_rate = learning_rate
        self.__n_actions = n_actions
        self.__input_dims = input_dims
        self.__network_params = network_params

        #  initialize network
        self.__create_network()

    def choose_action(self, state):
        state = state[np.newaxis, :]
        probabilities = self.__network.predict(state)[0]
        action = np.random.choice(self.__n_actions, p=probabilities)
        return action

    def store_experience(self, state, action, reward):
        self.__state_memory.append(state)
        self.__action_memory.append(action)
        self.__reward_memory.append(reward)

    def train(self):
        """
        Trains according to REINFORCE algorithm
        - uses stored experience as an episode
        for t=1:T-1:
            theta <- theta + alpha * grad (log(policy(s,a)) * G(t))
        """
        # get G(t) functions which is discount sum of rewards from time t to end
        G = []
        reward_sum = 0
        self.__reward_memory.reverse()
        for reward in self.__reward_memory:
            reward_sum = reward + self.__gamma * reward_sum
            G.append(reward_sum)
        G.reverse()
        for state, action, G_ in zip(self.__state_memory, self.__action_memory, G):
            with tf.GradientTape() as gradient_tape:
                # calculate action probabilities
                # probabilities = self.__network.predict(state[np.newaxis, :])
                probabilities = self.__network(state[np.newaxis, :])
                # clip probabilities to solve gradient bump issue
                # np.clip(probabilities, 1e-8, 1-1e-8, out=probabilities)
                # probabilities = tf.constant(probabilities)
                probabilities = K.clip(probabilities, 1e-8, 1 - 1e-8)
                # calculate the loss and put minus (-) to get gradient ascent
                loss = -self.__reinforce_loss(probabilities, action, G_)

            gradients = gradient_tape.gradient(loss, self.__network.trainable_weights)
            self.__optimizer.apply_gradients(zip(gradients, self.__network.trainable_weights))

        # clear memory
        self.__state_memory.clear()
        self.__action_memory.clear()
        self.__reward_memory.clear()

        #increase train counter
        self.__train_counter += 1

    def __reinforce_loss(self, probabilities, action, G):
        prob_dist = tfp.distributions.Categorical(probs=probabilities, dtype=tf.float32)
        action_log_prob = prob_dist.log_prob(action)
        return action_log_prob * G

    def __create_network(self):
        """
        Creates a neural network with provided network_params
        """
        model = Sequential()
        model.add(Input(shape=(self.__input_dims,)))
        for layer in self.__network_params:
            model.add(Dense(layer, kernel_initializer=tf.keras.initializers.glorot_normal(), activation="relu"))
        model.add(Dense(self.__n_actions, activation="softmax"))
        self.__optimizer = Adam(learning_rate=self.__learning_rate)
        self.__network = model

    def save_model(self, path):
        """
        - path/network -> network
        - path/agent_params.json -> agent parameters
        :param path:
        :return:
        """
        self.__network.save("{0}/network".format(path))

        # save the agent parameters
        params = {
            "train_counter":self.__train_counter,
            "gamma":self.__gamma,
            "learning_rate":self.__learning_rate,
            "input_dims":self.__input_dims,
            "n_actions":self.__n_actions,
        }

        with open("{0}/agent_params.json".format(path), "w") as f:
            json.dump(params, f)

    def load_model(self, path):
        """
        - path/network -> network
        :param path:
        :return:
        """
        self.__network = keras.models.load_model("{0}/network".format(path))

        # load agent parameters
        with open("{0}/agent_params.json".format(path),"r") as f:
            params = json.load(f)

            self.__train_counter = params["train_counter"]
            self.__gamma = params["gamma"]
            self.__learning_rate = params["learning_rate"]
            self.__input_dims = params["input_dims"]
            self.__n_actions = params["n_actions"]

class ActorCriticAgent():
    def __init__(self):
        self.__train_counter = 0

        # initialize memory as seperate lists
        self.__state_memory = []
        self.__next_state_memory = []
        self.__done_memory = []
        self.__action_memory = []
        self.__reward_memory = []

    def set_params(self, gamma, actor_learning_rate, critic_learning_rate, n_actions, input_dims, actor_network_params,
                 critic_network_params):
        self.__gamma = gamma
        self.__actor_learning_rate = actor_learning_rate
        self.__critic_learning_rate = critic_learning_rate
        self.__n_actions = n_actions
        self.__input_dims = input_dims
        self.__actor_network_params = actor_network_params
        self.__critic_network_params = critic_network_params

        # initialize network
        self.__actor_network, self.__actor_optimizer = self.__create_actor_network()
        self.__critic_network, self.__critic_optimizer = self.__create_critic_network()

    def choose_action(self, state):
        state = state[np.newaxis, :]
        probabilities = self.__actor_network.predict(state)[0]
        action = np.random.choice(self.__n_actions, p=probabilities)
        return action

    def store_experience(self, state, action, next_state, reward, done):
        self.__state_memory.append(state)
        self.__action_memory.append(action)
        self.__reward_memory.append(reward)
        self.__next_state_memory.append(next_state)
        self.__done_memory.append(done)

    def train(self):
        """
        Uses the Vanilla policy gradient scheme
        at each time step t
        - calculate A(t) advantage function
        theta <- theta + alpha * A(t) * grad( log( prob(action) ) )
        delta = r + gamma * V(s') - V(s)
        w <- w + alpha_2 * delta * grad(V)
        """
        for state, action, next_state, done, reward in zip(self.__state_memory, self.__action_memory,
                                                           self.__next_state_memory, self.__done_memory,
                                                           self.__reward_memory):
            with tf.GradientTape() as gradient_tape1, tf.GradientTape() as gradient_tape2:
                # calculate action probabilities
                probabilities = self.__actor_network(state[np.newaxis, :], training=True)
                probabilities = K.clip(probabilities, 1e-8, 1 - 1e-8)

                V = self.__critic_network(state[np.newaxis, :], training=True)
                V_next = self.__critic_network(next_state[np.newaxis, :], training=True)

                # calculate temporal difference
                td = reward + self.__gamma * V_next * (1 - int(done)) - V

                # calculate the loss and put minus (-) to get gradient ascent
                actor_loss = -self.__actor_loss(probabilities, action, td)

                # calculate critic loss -> minimize td error
                critic_loss = self.__critic_loss(td)

            actor_gradients = gradient_tape1.gradient(actor_loss, self.__actor_network.trainable_variables)
            self.__actor_optimizer.apply_gradients(zip(actor_gradients, self.__actor_network.trainable_variables))

            critic_gradients = gradient_tape2.gradient(critic_loss, self.__critic_network.trainable_variables)
            self.__critic_optimizer.apply_gradients(zip(critic_gradients, self.__critic_network.trainable_variables))

        # clear memory
        self.__state_memory.clear()
        self.__action_memory.clear()
        self.__next_state_memory.clear()
        self.__reward_memory.clear()
        self.__done_memory.clear()

    def __actor_loss(self, probabilities, action, A):
        prob_dist = tfp.distributions.Categorical(probs=probabilities, dtype=tf.float32)
        action_log_prob = prob_dist.log_prob(action)
        return action_log_prob * A

    def __critic_loss(self, td):
        return td ** 2

    def __create_actor_network(self):
        """
        Creates a neural network with provided network_params
        """
        model = Sequential()
        model.add(Input(shape=(self.__input_dims,)))
        for layer in self.__actor_network_params:
            model.add(Dense(layer, kernel_initializer=tf.keras.initializers.glorot_normal(), activation="relu"))
        model.add(Dense(self.__n_actions, activation="softmax"))
        # self.__actor_optimizer = Adam(learning_rate=self.__actor_learning_rate)
        # self.__actor_network = model
        return model, Adam(learning_rate=self.__actor_learning_rate)

    def __create_critic_network(self):
        """
        Creates critic network given critic network parameters.
        """
        model = Sequential()
        model.add(Input(shape=(self.__input_dims,)))
        for layer in self.__critic_network_params:
            #model.add(Dense(layer, kernel_initializer=tf.keras.initializers.glorot_normal(), activation="relu"))
            model.add(Dense(layer, activation="relu"))
        model.add(Dense(1))
        # self.__critic_optimizer = Adam(learning_rate=self.__critic_learning_rate)
        # self.__critic_network = model
        return model, Adam(learning_rate=self.__critic_learning_rate)

    def save_model(self, path):
        """
        Save agent params, actor and critic networks
        - path/agent_params.json -> agent params
        - path/actor_network
        - path/critic_network
        :param path:
        :return:
        """
        self.__actor_network.save("{0}/actor_network".format(path))
        self.__critic_network.save("{0}/critic_network".format(path))

        # save the agent parameters
        params = {
            "train_counter":self.__train_counter,
            "gamma":self.__gamma,
            "actor_learning_rate":self.__actor_learning_rate,
            "critic_learning_rate":self.__critic_learning_rate,
            "input_dims":self.__input_dims,
            "n_actions":self.__n_actions,
        }

        with open("{0}/agent_params.json".format(path), "w") as f:
            json.dump(params, f)

    def load_model(self, path):
        """
        load agent params, actor and critic networks and optimizers
        - path/agent_params.json -> agent params
        - path/actor_network
        - path/critic_network

        *The function has a naive assumption that OPTIMIZERS are not required to be reloaded, they can be recreated.

        :param path:
        :return:
        """
        # load agent parameters
        with open("{0}/agent_params.json".format(path),"r") as f:
            params = json.load(f)

            self.__train_counter = params["train_counter"]
            self.__gamma = params["gamma"]
            self.__actor_learning_rate = params["actor_learning_rate"]
            self.__critic_learning_rate = params["critic_learning_rate"]
            self.__input_dims = params["input_dims"]
            self.__n_actions = params["n_actions"]

        self.__actor_optimizer = Adam(learning_rate=self.__actor_learning_rate)
        self.__critic_optimizer = Adam(learning_rate=self.__critic_learning_rate)

        self.__actor_network = keras.models.load_model("{0}/actor_network".format(path))
        self.__critic_network = keras.models.load_model("{0}/critic_network".format(path))

