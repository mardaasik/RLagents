import numpy as np
import json

class Buffer():
    def __init__(self, state_dim, n_actions, capacity):
        self.__capacity = capacity
        self.__state_dim = state_dim
        self.__n_actions = n_actions

        #initialize buffer counter
        self.__buffer_counter = 0

        self.__state_buffer = np.zeros((self.__capacity, self.__state_dim))
        self.__next_state_buffer = np.zeros((self.__capacity, self.__state_dim))
        self.__action_buffer = np.zeros((self.__capacity, 1))
        self.__reward_buffer = np.zeros((self.__capacity, 1))
        self.__done_buffer = np.zeros((self.__capacity, 1))

    def store_transition(self, state, action, next_state, reward, done):
        index = self.__buffer_counter % self.__capacity
        self.__state_buffer[index] = state
        self.__next_state_buffer[index] = next_state
        self.__action_buffer[index] = action
        self.__reward_buffer[index] = reward
        self.__done_buffer[index] = int(done)

        self.__buffer_counter += 1

    def sample(self, batch_size):
        #if buffer has more elements than batch_size
        if self.__buffer_counter >= batch_size:
            # get min of buffer counter and capacity since buffer counter goes unlimited if clear() is not run
            range = min(self.__buffer_counter, self.__capacity)
            batch_indices = np.random.choice(range, batch_size)
            return self.__state_buffer[batch_indices], \
                   self.__action_buffer[batch_indices], \
                   self.__next_state_buffer[batch_indices], \
                   self.__reward_buffer[batch_indices], \
                   self.__done_buffer[batch_indices]
        # else return none
        else:
            return None

    def get_buffer(self):
        #return every sample in buffer
        return self.__state_buffer[:], \
               self.__action_buffer[:], \
               self.__next_state_buffer[:], \
               self.__reward_buffer[:], \
               self.__done_buffer[:]

    def clear_buffer(self):
        # setting counter to 0 is equivalent to clearing
        self.__buffer_counter = 0

    def get_buffer_counter(self):
        return self.__buffer_counter

    def save(self, path):
        with open("{0}_state_buffer.npy".format(path), "wb") as f_state_buffer:
            np.save(f_state_buffer, self.__state_buffer)
        with open("{0}_action_buffer.npy".format(path), "wb") as f_action_buffer:
            np.save(f_action_buffer, self.__action_buffer)
        with open("{0}_next_state_buffer.npy".format(path), "wb") as f_next_state_buffer:
            np.save(f_next_state_buffer, self.__next_state_buffer)
        with open("{0}_reward_buffer.npy".format(path), "wb") as f_reward_buffer:
            np.save(f_reward_buffer, self.__reward_buffer)
        with open("{0}_done_buffer.npy".format(path), "wb") as f_done_buffer:
            np.save(f_done_buffer, self.__done_buffer)

        params = {
            "state_dim": self.__state_dim,
            "n_actions": self.__n_actions,
            "capacity": self.__capacity,
            "buffer_counter": self.__buffer_counter
        }

        with open("{0}_buffer_params.json".format(path),"w") as f_buffer_params:
            f_buffer_params.write(json.dumps(params))


    def load(self, path):
        with open("{0}_state_buffer.npy".format(path), "rb") as f_state_buffer:
            self.__state_buffer = np.load(f_state_buffer)
        with open("{0}_action_buffer.npy".format(path), "rb") as f_action_buffer:
            self.__action_buffer = np.load(f_action_buffer)
        with open("{0}_next_state_buffer.npy".format(path), "rb") as f_next_state_buffer:
            self.__next_state_buffer = np.load(f_next_state_buffer)
        with open("{0}_reward_buffer.npy".format(path), "rb") as f_reward_buffer:
            self.__reward_buffer = np.load(f_reward_buffer)
        with open("{0}_done_buffer.npy".format(path), "rb") as f_done_buffer:
            self.__done_buffer = np.load(f_done_buffer)

        with open("{0}_buffer_params.json".format(path),"r") as f_buffer_params:
            params = json.loads(f_buffer_params.read())
            self.__state_dim = params["state_dim"]
            self.__n_actions = params["n_actions"]
            self.__capacity = params["capacity"]
            self.__buffer_counter = params["buffer_counter"]