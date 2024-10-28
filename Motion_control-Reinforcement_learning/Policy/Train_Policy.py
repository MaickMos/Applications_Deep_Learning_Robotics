import gym
import pybullet, pybullet_envs
import torch as th
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import keyboard
import sys

#ppo_Ant_learning_rate 10^-X,timesteps 10^X,Valor de goal_reward
learning_rate="5"
timesteps="5"
goal_reward="3000"
train="___ppo_Ant_"+learning_rate+"_"+timesteps+"_"+goal_reward

global end
end=False

def on_press(key):
    if key.name == 'f8':
        global end
        end=True
        print("Deteniendo entrenamiento... Espera un momento")


# Create environment
# env = gym.make('LunarLanderContinu-ous-v2')

#env = gym.make('BipedalWalker-v3')
env = gym.make("AntBulletEnv-v0")
env.render(mode="human")

policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=[512, 512])
# Instantiate the agent
model = PPO("MlpPolicy",env,learning_rate=10**-(int(learning_rate)),policy_kwargs=policy_kwargs,verbose=1)



archivo = open(train+".csv", "w", newline='')
# Crea un objeto escritor
escritor = csv.writer(archivo)
# Escribe los encabezados
escritor.writerow(["Iteracion", "mean_reward"])
keyboard.on_press(on_press)

# Train the agent
for i in range(8000):
    
    print("Training itteration ",i)
    model.learn(total_timesteps=10**int(timesteps))
    # Save the agent
    model.save(train)
    mean_reward, std_reward = evaluate_policy(model,model.get_env(), n_eval_episodes=5)

    print("mean_reward ", mean_reward)
    escritor.writerow([i,mean_reward])
    print("End Training itteration ",i)
    
    if mean_reward>=int(goal_reward):
        print("Agent Trained with average reward",mean_reward)
        archivo.close()
        break

    if end==True:
        print("Entrenamiento detenido, mean reward:",mean_reward)
        archivo.close()
        break

#del model  # delete trained model to demonstrate loading
# Load the trained agent
# model = PPO.load("ppo_Ant")

# Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
# obs = env.reset()
# for i in range(100):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
