# Rllib docs: https://docs.ray.io/en/latest/rllib.html
# Malmo XML docs: https://docs.ray.io/en/latest/rllib.html

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from ray.tune.logger import pretty_print

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo
import matplotlib.pyplot as plt
import numpy as np


class CrossTheFireLine(gym.Env):

    def __init__(self, env_config):
        # Static Parameters
        self.size = 50
        self.obs_size = 5
        #self.max_episode_steps = 100
        self.log_frequency = 10
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'move -1',  # Move one block backward
            2: 'turn 1',  # Turn 90 degrees to the right
            3: 'turn -1'  # Turn 90 degrees to the left
        }

        # Rllib Parameters
        self.action_space = Discrete(len(self.action_dict))
        # self.action_space = Box(low = -1, high = 1, shape=(2,))  # move, turn
        # self.observation_space = Box(0, 1, shape=(2 * self.obs_size * self.obs_size, ), dtype=np.float32)
        self.observation_space = Box(0, 1, shape=(1 * self.obs_size * self.obs_size, ), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # CrossTheFireLine Parameters
        self.obs = None
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []
        self.reward = 0

        #self.prevTime = 0
        self.previousLife = 20
        self.aimed_pos = []

        self.survivalTime = []
        self.episode_survivalTime = 0



    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo

        world_state = self.init_malmo()


        # Reset Variables
        print(self.episode_return)
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        self.previousLife = 20 ###

        self.survivalTime.append(self.episode_survivalTime)
        print(self.survivalTime)
        self.episode_survivalTime = 0

        # Log
        if len(self.returns) > self.log_frequency + 1 and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()
            self.log_survivalTime()

        # Get Observation
        self.obs = self.get_observation(world_state)

        return self.obs

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        # Get Action
        command = self.action_dict[action]
        self.agent_host.sendCommand(command)

        # action_type = ["move","turn"]
        # for i in range(len(action_type)):
        #     command = str(action_type[i]) + " " + str(action[i])
        #     self.agent_host.sendCommand(command)
        #     time.sleep(.1)

        self.episode_step += 1

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state)

        # Get Done
        done = not world_state.is_mission_running

        # Get Reward
        reward = 1 ##changed
        for r in world_state.rewards:
            reward += r.getValue()

        if len(world_state.observations)>0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            if observations["Life"] < self.previousLife:
                reward -= (self.previousLife - observations["Life"]) * 10
            self.previousLife = observations["Life"]

            # if "entities" in observations:
            #     for entity in observations["entities"]:
            #         if entity["name"] == "Fireball" and self.previousLife <= observations["Life"]:
            #             reward += 10

            # if observations["TimeAlive"] > self.prevTime:
            #     reward += (observations["TimeAlive"]-self.prevTime)
            #     self.prevTime = observations["TimeAlive"]

            if observations["TimeAlive"] > 0:
                self.episode_survivalTime=observations["TimeAlive"]

        print("\n===", reward, "===")
        self.episode_return += reward

        return self.obs, reward, done, dict()

    def get_mission_xml(self): #<ContinuousMovementCommands/>
        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                        <Summary>CrossTheFireLine</Summary>
                    </About>

                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>0</StartTime>
                                <AllowPassageOfTime>false</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;1;"/>
                            <DrawingDecorator>''' + \
                                "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-self.size, self.size, -self.size, self.size) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='stone'/>".format(-self.size, self.size, -self.size, self.size) + \
                                '''
                                <DrawCuboid x1='-30' x2='30' y1='2' y2='50' z1='-30' z2='-29' type='stone'/>
                                <DrawCuboid x1='-30' x2='30' y1='2' y2='50' z1='29' z2='30' type='stone'/>
                                <DrawCuboid x1='-30' x2='-29' y1='2' y2='50' z1='-30' z2='30' type='stone'/>
                                <DrawCuboid x1='29' x2='30' y1='2' y2='50' z1='-30' z2='30' type='stone'/>
                                <DrawCuboid x1='-30' x2='30' y1='51' y2='51' z1='-30' z2='30' type='stone'/>
                                <DrawBlock x='0'  y='2' z='0' type='air' />
                                <DrawBlock x='0'  y='1' z='0' type='stone' />

                                <DrawCuboid x1='-30' x2='30' y1='3' y2='5' z1='-30' z2='-29' type='sea_lantern'/>
                                <DrawCuboid x1='-30' x2='30' y1='3' y2='5' z1='29' z2='30' type='sea_lantern'/>
                                <DrawCuboid x1='-30' x2='-29' y1='3' y2='5' z1='-30' z2='30' type='sea_lantern'/>
                                <DrawCuboid x1='29' x2='30' y1='3' y2='5' z1='-30' z2='30' type='sea_lantern'/>

                                <DrawCuboid x1='-30' x2='30' y1='13' y2='15' z1='-30' z2='-29' type='sea_lantern'/>
                                <DrawCuboid x1='-30' x2='30' y1='13' y2='15' z1='29' z2='30' type='sea_lantern'/>
                                <DrawCuboid x1='-30' x2='-29' y1='13' y2='15' z1='-30' z2='30' type='sea_lantern'/>
                                <DrawCuboid x1='29' x2='30' y1='13' y2='15' z1='-30' z2='30' type='sea_lantern'/>

                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>
                    <AgentSection mode="Survival">
                        <Name>CrossTheFireLine</Name>
                        <AgentStart>
                            <Placement x="0.5" y="2" z="0.5" pitch="45" yaw="0"/>
                            <Inventory>
                            </Inventory>
                        </AgentStart>
                        <AgentHandlers>
                            <DiscreteMovementCommands/>
                            <ObservationFromFullStats/>
                            <ObservationFromRay/>
                            <ObservationFromGrid>
                                <Grid name="floorAll">
                                    <min x="-'''+str(int(self.obs_size/2))+'''" y="0" z="-'''+str(int(self.obs_size/2))+'''"/>
                                    <max x="'''+str(int(self.obs_size/2))+'''" y="0" z="'''+str(int(self.obs_size/2))+'''"/>
                                </Grid>
                            </ObservationFromGrid>
                            <ObservationFromNearbyEntities>
                                <Range name = "entities" xrange="30" yrange="50" zrange="30"/>
                            </ObservationFromNearbyEntities>
                            <ChatCommands/>
                            <RewardForTouchingBlockType>
                                <Block type="fire" reward="-10.0"/>
                            </RewardForTouchingBlockType>
                            <AgentQuitFromTouchingBlockType>
                                <Block type="bedrock" />
                            </AgentQuitFromTouchingBlockType>
                            <AgentQuitFromReachingCommandQuota total="1000" />
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''
#<AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''" />
    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'CrossTheFireLine' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)
        self.agent_host.sendCommand('chat /kill @e[type=!Player]')
        time.sleep(1)
        self.agent_host.sendCommand('chat /summon Ghast -10 5 10')
        return world_state

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a flattened 1 x 5 x 5 grid around the agent.
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> the state observation
        """
        return_obs = np.zeros((1 * self.obs_size * self.obs_size, ))
        obs = np.zeros((1 * self.obs_size * self.obs_size, ))

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)
                # print("\n242 here: " , observations)

                # if "entities" in observations:
                #     print("entity found")

                #Get observation
                if 'floorAll' in observations:
                    grid = observations['floorAll']
                    has_fireball = False
                    if "entities" in observations:
                        for entity in observations["entities"]:
                            if entity["name"] == "Fireball":
                                # # Plot the 3-D quiver graph
                                # ax = plt.figure().add_subplot(projection='3d')
                                # x = entity['x']
                                # y = entity['y']
                                # z = entity['z']
                                # u = entity['motionX']
                                # v = entity['motionY']
                                # w = entity['motionZ']
                                # ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
                                # plt.show()
                                # # end ploting

                                has_fireball = True
                                if not self.aimed_pos:  #check if aimed_pos is already recorded
                                    self.aimed_pos = [observations["XPos"], observations["ZPos"]]

                                current_pos = [observations["XPos"], observations["ZPos"]] #x,z
                                fireball_grid = np.zeros((1 * self.obs_size * self.obs_size), )

                                #left up
                                if current_pos[0]-self.obs_size/2<=self.aimed_pos[0]<=current_pos[0] and current_pos[1] <=self.aimed_pos[1]<=current_pos[1]+self.obs_size/2:
                                    for i in [0,1,2,5,6,7,10,11,12]:
                                        fireball_grid[i] = 1
                                #left down
                                elif current_pos[0]-self.obs_size/2<=self.aimed_pos[0]<=current_pos[0] and current_pos[1]-self.obs_size/2 <=self.aimed_pos[1]<=current_pos[1]:
                                    for i in [10,11,12,15,16,17,20,21,22]:
                                        fireball_grid[i] = 1
                                #right up
                                elif current_pos[0]<=self.aimed_pos[0]<=current_pos[0]+self.obs_size/2 and current_pos[1] <=self.aimed_pos[1]<=current_pos[1]+self.obs_size/2:
                                    for i in [2,3,4,7,8,9,12,13,14]:
                                        fireball_grid[i] = 1
                                #right down
                                elif current_pos[0]<=self.aimed_pos[0]<=current_pos[0]+self.obs_size/2 and current_pos[1]-self.obs_size/2 <=self.aimed_pos[1]<=current_pos[1]:
                                    for i in [12,13,14,17,18,19,22,23,24]:
                                        fireball_grid[i] = 1
                                # else fire ball out of bound


                    if not has_fireball: #if there is no fireball, reset aimed_pos
                        self.aimed_pos = []
                        fireball_grid = np.zeros((1 * self.obs_size * self.obs_size), )



                    for i, x in enumerate(grid):
                        obs[i] = x == 'fire'
                    # Rotate observation with orientation of agent
                    obs = obs.reshape((1, self.obs_size, self.obs_size))
                    fireball_grid = fireball_grid.reshape((1, self.obs_size, self.obs_size))
                    yaw = observations['Yaw']
                    if yaw < 0 : #from disucussion post 141
                        yaw = 360 + yaw
                    if yaw >= 225 and yaw < 315:
                        obs = np.rot90(obs, k=1, axes=(1, 2))
                        fireball_grid = np.rot90(fireball_grid, k=1, axes=(1, 2))
                    elif yaw >= 315 or yaw < 45:
                        obs = np.rot90(obs, k=2, axes=(1, 2))
                        fireball_grid = np.rot90(fireball_grid, k=2, axes=(1, 2))
                    elif yaw >= 45 and yaw < 135:
                        obs = np.rot90(obs, k=3, axes=(1, 2))
                        fireball_grid = np.rot90(fireball_grid, k=3, axes=(1, 2))
                    obs = obs.flatten()
                    fireball_grid = fireball_grid.flatten()
                    # combine fireball_grid and obs
                    return_obs = np.logical_or(obs,fireball_grid)
                    return_obs = 1*return_obs
                    # return_obs = np.vstack((obs, fireball_grid)).flatten()
                    break

        return return_obs

    def log_survivalTime(self):
        plt.clf()
        plt.plot(self.survivalTime)
        plt.title('CrossTheFireLine')
        plt.ylabel('survivalTime')
        plt.xlabel('episode')
        plt.savefig('survival.png')


    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('CrossTheFireLine')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value))


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=CrossTheFireLine, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    #load from checkpoint
    # trainer.restore("/tmp/rllib_checkpoint/checkpoint_1/checkpoint-1")

    while True:
        pretty_print(trainer.train())
    # for _ in range(8):
    #     print(pretty_print(trainer.train()))
    # trainer.save("/tmp/rllib_checkpoint")

    #
    # analysis = ray.tune.run(
    #     "PPO",
    #     config={
    #         'env_config': {},
    #         'env': "CrossTheFireLine",        # No environment parameters to configure
    #         'framework': 'torch',       # Use pyotrch instead of tensorflow
    #         'num_gpus': 0,              # We aren't using GPUs
    #         'num_workers': 0            # We aren't using parallelism
    #     },
    #     stop={"training_iteration": 10000},
    #     checkpoint_at_end=True)
    #


    # config = {
    #     "env_config": {},
    #     # Change the following line to `“framework”: “tf”` to use tensorflow
    #     "framework": "torch",
    #     'num_gpus': 0,
    #     'num_workers': 0,
    # }
    # stop = {"training_iteration": 10000}
    # ray.shutdown()
    # ray.init()
    #
    # # execute training
    # analysis = ray.tune.run(
    #   "PPO",
    #   config=config,
    #   stop=stop,
    #   checkpoint_at_end=True,
    # )


    # list of lists: one list per checkpoint; each checkpoint list contains
    # 1st the path, 2nd the metric value
    # checkpoints = analysis.get_trial_checkpoints_paths(
    #     trial=analysis.get_best_trial("episode_reward_mean"),
    #     metric="episode_reward_mean")

    # or simply get the last checkpoint (with highest "training_iteration")

    # last_checkpoint = analysis.get_last_checkpoint()

    # if there are multiple trials, select a specific trial or automatically
    # choose the best one according to a given metric
    # last_checkpoint = analysis.get_last_checkpoint(
    #     metric="episode_reward_mean", mode="max"
    # )


# Rllib docs: https://docs.ray.io/en/latest/rllib.html
# Malmo XML docs: https://docs.ray.io/en/latest/rllib.html
