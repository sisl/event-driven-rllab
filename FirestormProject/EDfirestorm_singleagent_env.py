from rllab.envs.env_spec import EnvSpec
import collections
from rllab.spaces import Discrete
import random
from numpy import sign
import pdb
from copy import deepcopy


class EDFirestorm_SingleAgent_Env(object):
    def __init__(self,n_row = 4,n_col = 4):
        '''
        Initialize the enviornment
        Input
        -----
        n_row : number of rows in environment grid
        n_cols : number of columns in enviroment grid
        '''
        self.n_row = n_row
        self.n_col = n_col
        self.n_fires = 3
        self.discount = 0.9

        # Initialization and game params
        self.start_XY = [0,1] # agent inital x,y on reset
        self.start_fire_status = [True]*3 # env initial fire status (all alive)
        self.fire_locations = [ [0,0], [1,3], [3,3] ] # x,y of 3 fires
        self.fire_rewards = [10,40,50]
        self.fire_extinguish_probs = [0.8,0.6,0.2] # probabilities that the 3 fires get extinguished
                                        # per timestep
        self.agent_death_prob = 0.2
        # State variables
        self.current_XY = self.start_XY
        self.current_fire_status = deepcopy(self.start_fire_status)
        self.sim_time = 0 # number of timestep since reset

    def XY_to_action_index(self,x,y):
        '''
        Go from (x,y) coordinates to index into action space
        Inputs
        -----
        x,y : x and y coordinates in grid
        Outputs
        -----
        state_ind : state index 
        '''
        return x*self.n_col + y
    def action_index_to_XY(self,action_index):
        # Inverse of XY_to_action
        x = action_index // self.n_col
        y = action_index - (x*self.n_col)
        return x,y

    def bool_array_to_int(self,bool_array):
        # Big endian
        int_out = 0
        for i, elem in enumerate(bool_array):
            if elem:
                int_out += 2**(len(bool_array) - i - 1)
        return int_out

    def state_to_obs_index(self,x,y,fire_alive):
        '''
        Go from (x,y,fire_live) coordinates to index into obs space
        Inputs
        -----
        x,y : x and y coordinates in grid
        fire_alive : bool array for status of fire 1..N (True=Alive)
        Outputs
        -----
        state_ind : state index 
        '''
        XY_offset = self.XY_to_action_index(x,y)
        int_rep_firestatus = self.bool_array_to_int(fire_alive)
        return self.n_row*self.n_col*int_rep_firestatus + XY_offset


    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """

        '''
        Event triggers:
        To make this event driven, the action is assumed to be the desired location 
        of the agent. The next observation will only be returned when either:
            1. the agent has reached the desired loc, and a fire no longer exists there
            2. the agent died
        '''
        # pdb.set_trace()
        desired_x,desired_y = self.action_index_to_XY(action)
        reward = 0.
        done = False

        # simulate until event triggered
        while(True):
            self.sim_time += 1
            # Check if agent died
            if(random.random() <= self.agent_death_prob):
                done = True
                break
            # Move agent toward desired loc using basic actions (Stay,L,R,U,D)
            #  Trajectory will be an L for simplicity, going verically first, then horiz
            curr_x = self.current_XY[0]
            curr_y = self.current_XY[1]
            
            if(desired_y != curr_y):
                # move vertically one space
                curr_y += sign(desired_y - curr_y)
            elif(desired_x != curr_x):
                # move horizontally one space
                curr_x += sign(desired_x - curr_x)
            self.current_XY = [curr_x,curr_y]

            # Check if fire extinguished in current position, assign reward
            in_loc_and_fire_extinguished = True # assigned here for convenience
            if self.current_XY in self.fire_locations: # check if fire exists
                ind = self.fire_locations.index(self.current_XY)
                if(self.current_fire_status[ind]): # check if fire is alive
                    in_loc_and_fire_extinguished = False
                    # try extinguishing
                    if(random.random() <= self.fire_extinguish_probs[ind]):
                        # if successful, assign reward
                        self.current_fire_status[ind] = False
                        dt = self.sim_time - 1
                        reward += self.fire_rewards[ind]*(self.discount**dt)
                        in_loc_and_fire_extinguished = True
                
            # If in desired location and fire extinguished, break
            if(self.current_XY == [desired_x,desired_y] and in_loc_and_fire_extinguished):
                break


        # Check if all fires extinguished
        if(not any(self.current_fire_status)):
            done = True

        obs = self.state_to_obs_index(self.current_XY[0],self.current_XY[1], 
            self.current_fire_status)

        return Step(observation=obs, reward=reward, done=done)

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.current_XY = self.start_XY
        self.current_fire_status = deepcopy(self.start_fire_status)
        self.sim_time = 0

        return

    @property
    def action_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        Action space is [Agent's desired X,Y loc represented in cartesian product form]
        """
        return Discrete(self.n_row * self.n_col) 

    @property
    def observation_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        State is [Agent X loc, Agent Y loc, Fire 1, Fire 2,... Fire N Alive?]
        Represented in cartesian product form as a Discrete space
        """
        return Discrete(self.n_row * self.n_col * 2**self.n_fires)

    # Helpers that derive from Spaces
    @property
    def action_dim(self):
        return self.action_space.flat_dim

    def render(self):
        pass

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    @property
    def horizon(self):
        """
        Horizon of the environment, if it has one
        """
        raise NotImplementedError


    def terminate(self):
        """
        Clean up operation,
        """
        pass


class NonEDFirestorm_SingleAgent_Env(EDFirestorm_SingleAgent_Env):

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """

        '''
        Event triggers:
        To make this event driven, the action is assumed to be the desired location 
        of the agent. The next observation will only be returned when either:
            1. the agent has reached the desired loc, and a fire no longer exists there
            2. the agent died
        '''
        # pdb.set_trace()
        desired_x,desired_y = self.action_index_to_XY(action)
        reward = 0.
        done = False

        # simulate until event triggered
        while(True):
            self.sim_time += 1
            # Check if agent died
            if(random.random() <= self.agent_death_prob):
                done = True
                break
            # Move agent toward desired loc using basic actions (Stay,L,R,U,D)
            #  Trajectory will be an L for simplicity, going verically first, then horiz
            curr_x = self.current_XY[0]
            curr_y = self.current_XY[1]
            
            if(action == 1):
                # up
                curr_y = min(self.n_row-1, curr_y + 1)
            elif(action == 2):
                # down
                curr_y = max(0, curr_y - 1)
            elif(action == 3):
                # left
                curr_x = min(self.n_col-1, curr_x + 1)
            elif(action == 4):
                # down
                curr_x = max(0, curr_x - 1)

            self.current_XY = [curr_x,curr_y]

            # Check if fire extinguished in current position, assign reward
            in_loc_and_fire_extinguished = True # assigned here for convenience
            if self.current_XY in self.fire_locations: # check if fire exists
                ind = self.fire_locations.index(self.current_XY)
                if(self.current_fire_status[ind]): # check if fire is alive
                    in_loc_and_fire_extinguished = False
                    # try extinguishing
                    if(random.random() <= self.fire_extinguish_probs[ind]):
                        # if successful, assign reward
                        self.current_fire_status[ind] = False
                        reward += self.fire_rewards[ind]
                        in_loc_and_fire_extinguished = True
                
            # If in desired location and fire extinguished, break
            if(self.current_XY == [desired_x,desired_y] and in_loc_and_fire_extinguished):
                break
            # Since non-ED, just do the one iter of the loop
            break 


        # Check if all fires extinguished
        if(not any(self.current_fire_status)):
            done = True

        obs = self.state_to_obs_index(self.current_XY[0],self.current_XY[1], 
            self.current_fire_status)

        return Step(observation=obs, reward=reward, done=done)

    @property
    def action_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        Action space is [Agent's desired X,Y loc represented in cartesian product form]
        """
        return Discrete(5) 


_Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])


def Step(observation, reward, done, **kwargs):
    """
    Convenience method creating a namedtuple with the results of the
    environment.step method.
    Put extra diagnostic info in the kwargs
    """
    return _Step(observation, reward, done, kwargs)
