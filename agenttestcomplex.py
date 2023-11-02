from complexenvtest import WaterSupplyEnv
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from rl.agents import DQNAgent
from keras.optimizers import Adam
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory



#Load the environment

env1=WaterSupplyEnv()

#Get action and state domain
states = env1.observation_space.shape[0]
actions=env1.action_space.n

#build the ANN architecture
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))

    return model

model = build_model(states, actions)

#Define agents hyperparameters
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,gamma=0.999,batch_size=8,
                  nb_actions=actions, nb_steps_warmup=1000, target_model_update=10000,enable_double_dqn=True, enable_dueling_network=True, dueling_type='avg')
    return dqn

dqn = build_agent(model, actions)

dqn.compile(Adam(learning_rate=1e-4), metrics=['mae'])

#Training
history=dqn.fit(env1, nb_steps=10000000, visualize=False, verbose=1)
scores = dqn.test(env1, nb_episodes=10, visualize=False)
dqn.save_weights("dqn_weight30000.h5f",overwrite=True)
plt.style.use(['science','no-latex'])
plt.figure(figsize=(8,4))
plt.plot(history.history["episode_reward"])
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Total Reward Evolution')
plt.savefig("Detrain")
# Show the plot
plt.show()
print("custo: ",env1.cost,"€")
print("tarifario: ",env1.numero)
print(scores)

#Testing (uncomment)
#
# dqn.load_weights("dqn_weight500.h5f")
#
# for i in range(1):
#
#     env1.day=-1
#     env1.numero=26
#     env1.mes=6
#     scores = dqn.test(env1, nb_episodes=i+1, visualize=False)
#     plt.plot(env1.states)
#     plt.axhline(y=8, color='r')
#     plt.axhline(y=2, color='r')
#     plt.legend(["Tank level", "Pump state", "Tariff"])
#
#     plt.xlabel('t [timestep=30min]')
#     plt.ylabel('Tank level/Pump State/Normalized Tariffs(x10)')
#     plt.title('Proposed solution:Cost {} €'.format(round(env1.cost, 2)))
#     plt.savefig('training')
#     plt.show()
#     # Show the plot
#     print("custo: ", env1.cost, "€")
#     print("tarifario: ", env1.numero)
#     print(scores)