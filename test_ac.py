import rl_agents.agents
import gym

env = gym.make("CartPole-v1")
env.reset()

agent = rl_agents.agents.ActorCriticAgent()
#agent.set_params(gamma=0.99, learning_rate=0.0001, n_actions=env.action_space.n, input_dims=4, network_params=[256, 256])
agent.set_params(gamma=0.99, actor_learning_rate=0.00005, critic_learning_rate=0.0001, n_actions=env.action_space.n
                 ,input_dims=4, actor_network_params=[256,256], critic_network_params=[256,256])

for i in range(0, 10000):
    observation = env.reset()
    total_reward = 0
    done = False
    step_counter = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        total_reward += reward
        agent.store_experience(observation, action, observation_, reward, done)
        observation = observation_
        step_counter += 1
    agent.train()

    print("Episode {0}, reward: {1}, total steps: {2}".format(i, total_reward, step_counter))
    if i % 50 == 0:
        agent.save_model("models/test_ac/_{0}".format(i))

env.close()