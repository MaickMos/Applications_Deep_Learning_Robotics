import gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("AntBulletEnv-v0")
env.render(mode="human")
#Some diferent enviroment
# env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('BipedalWalker-v3')

#policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=[512, 512])
# Instantiate the agent
#model = PPO('MlpPolicy', env,learning_rate=0.0001,policy_kwargs=policy_kwargs, verbose=1)

#del model  # delete trained model to demonstrate loading
# Load the trained agent
model = PPO.load("D:/Users/maick/Desktop/Tesis/RL/DataSet/Codigos/ppo_Ant_2_3_1000")

# # Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Execute the agent
obs = env.reset()
for i in range(100):
    dones = False
    game_score = 0
    steps = 0
    while not dones:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)
        # import ipdb;ipdb.set_trace()
        game_score+=rewards
        steps+=1
        
        env.render()
    print("game ", i ," steps   ",steps, " game score %.3f"%game_score)
    obs = env.reset()
    # break
