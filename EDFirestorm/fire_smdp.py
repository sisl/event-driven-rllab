import numpy as np
import random
import time
from math import exp
import pdb
import operator

class fire_extinguish(object):


  # environment setup
  field_size = 1.0   # x from -1 to 1 and y from -1 to 1
  # disc_intval = 0.25 # becomes a 8-by-8 grid
  n_uav = 2 # number of uav
  n_fire = 3 # number of fires
  delta_t_min = 0.5
  speed_fixed = 0.5

  # uav information
  u_loca = [[-0.9,-0.9], [0.9,0.9]]
  t_fail = [0.01, 0.01]
  t_emit = [0.3, 0.3]
  t_wait = [0.00,0.005]
  #v_uav = 1.0

  # fire information
  l_fire = [[-0.5,0.5], [0.5,-0.5], [0.9,-0.9]]
  r_fire = [5.0, 1.0, 1.0]
  e_fire = [[0.05,0.9],[0.9,0.9],[0.9,0.9]]
  t_fire = 0.5
  d_fire = 0.1
  #d_fire = 0.5 * self.disc_intval

  def env_reset(self):

    initial_state = []

    for i_uav in range(self.n_uav):

      initial_state = initial_state + self.u_loca[i_uav]

    initial_state = initial_state + self.t_wait

    initial_state = initial_state + [1.0] * self.n_fire

    return initial_state




  def boundary_check(self,position):

    [x,y] = position

    x_check = x; y_check = y

    if x < -1.0 * self.field_size :

      x_check = -1.0 * self.field_size

    if x > 1.0 * self.field_size :

      x_check = 1.0 * self.field_size

    if y < -1.0 * self.field_size :

      y_check = -1.0 * self.field_size

    if x > 1.0 * self.field_size :

      y_check = 1.0 * self.field_size

    return [x_check,y_check]



  def get_coordinate_grid(self,position):

    [x,y] = position

    x_integer = int( (x + field_size) / disc_intval)
    y_integer = int( (x + field_size) / disc_intval)

    x_grid_center = (-1.0) * field_size + (x_integer + 0.5) * disc_intval
    y_grid_center = (-1.0) * field_size + (y_integer + 0.5) * disc_intval

    return [x_grid_center,y_grid_center]



  def get_reward(self,current_state,next_state):

    # state takes the form [x_0,y_0,...x_n,y_n,wt_0,...,wt_n,f_0,f_1,f_2,...,f_m]

    r = 0.0

    for i_fire in range(self.n_fire):

      if current_state[ 3*self.n_uav + i_fire ] > 0.001 and next_state[ 3*self.n_uav + i_fire ] < 0.001:

        r += self.r_fire[i_fire]

    return r


  def get_distance(self,current_position,target_position):

    delta_x = [target_position[0]-current_position[0], target_position[1]-current_position[1]]

    distance = ((delta_x[0])**2 + (delta_x[1])**2)**0.5

    return distance



  def get_true_delta_t(self,current_position,target_position):

    delta_x = [target_position[0]-current_position[0], target_position[1]-current_position[1]]

    distance = ((delta_x[0])**2 + (delta_x[1])**2)**0.5

    return distance/self.speed_fixed



  def agent_move_for_fixed_duration(self,current_position,target_position,delta_t):

    estimate_time = self.get_true_delta_t(current_position,target_position)

    if estimate_time < delta_t:

      return target_position

    else:

      time_ratio = delta_t / estimate_time

      x_next = current_position[0] + (target_position[0]-current_position[0]) * time_ratio
      y_next = current_position[1] + (target_position[1]-current_position[1]) * time_ratio

      return [x_next,y_next]




  def uav_on_fire(self,position_uav,position_fire):

    [u_x,u_y] = position_uav;
    [f_x,f_y] = position_fire;

    if abs(u_x-f_x) < self.d_fire and abs(u_y-f_y) < self.d_fire:

      return 1

    else:

      return 0


  def next_state_fire_update(self,current_state,next_state,delta_t):

    next_state_updated = next_state[:]

    for i_fire in range(self.n_fire):

      i_fire_location = self.l_fire[i_fire]

      n_uav_on_fire = 0

      for j_uav in range(self.n_uav):

        j_uav_location_current = current_state[2*j_uav:2*j_uav+2]
        j_uav_location_next = next_state[2*j_uav:2*j_uav+2]


        if self.uav_on_fire(j_uav_location_current,i_fire_location) and self.uav_on_fire(j_uav_location_next,i_fire_location):

          n_uav_on_fire += 1

      if n_uav_on_fire >= 2:

        next_state_updated[3*self.n_uav + i_fire] = current_state[3*self.n_uav + i_fire] - self.e_fire[i_fire][1] * delta_t

      elif n_uav_on_fire == 1:

        next_state_updated[3*self.n_uav + i_fire] = current_state[3*self.n_uav + i_fire] - self.e_fire[i_fire][0] * delta_t

      else:

        next_state_updated[3*self.n_uav + i_fire] = current_state[3*self.n_uav + i_fire]


    return next_state_updated


  def fire_state_parsing(self,fire_state):

    parsed_fire_state = [0] * self.n_fire

    for i_fire in range(self.n_fire):

      if fire_state[i_fire] > 0.001:

        parsed_fire_state[i_fire] = 1

    return parsed_fire_state



  def transition(self,current_state,action):

    # action for each agent takes the form (x_target,y_target)
    # state takes the form [x_0,y_0,...x_n,y_n,wt_0,...,wt_n,f_0,f_1,f_2,...,f_m]

    reset_determine = sum(self.fire_state_parsing(current_state[3*self.n_uav:]))

    if reset_determine == 0:

      return self.env_reset()

    # who receive action -> the one with minimum waiting time

    waiting_time = current_state[2 * self.n_uav : 3 * self.n_uav]


    
    idx, val = min(enumerate(waiting_time), key=operator.itemgetter(1))
    sojurn_t = val; actor = idx

    next_state = current_state[:]

    # update the target_location and waiting time for actor
    next_state[2*actor] = action[2*actor]
    next_state[2*actor+1] = action[2*actor+1]

    waiting_time[actor] = (self.get_true_delta_t(current_state[2*actor:2*actor+2],
                                                 next_state[2*actor:2*actor+2])
                           + self.delta_t_min * (1.0 + 0 *random.random()))



    # transition to next state

    idx, val = min(enumerate(waiting_time), key=operator.itemgetter(1))
    sojurn_t = val; next_actor = idx


    # update location for each agent and waiting time
    for i_uav in range(self.n_uav):


      i_uav_current_position = [current_state[2*i_uav], current_state[2*i_uav+1]]
      i_uav_target_position =  [action[2*i_uav],        action[2*i_uav+1]]

      next_position = self.agent_move_for_fixed_duration(i_uav_current_position,
                                                         i_uav_target_position,
                                                         sojurn_t)

      next_state[2*i_uav] = next_position[0]
      next_state[2*i_uav+1] = next_position[1]


      waiting_time[i_uav] -= sojurn_t

    next_state[2*self.n_uav : 3*self.n_uav] = waiting_time[:]


    # Get reward
    next_state = self.next_state_fire_update(current_state,next_state,sojurn_t)
    r = self.get_reward(current_state,next_state)



    # observation
    obs = [  [None]   ] * self.n_uav
    obs[next_actor] = next_state[0:2*self.n_uav] + self.fire_state_parsing(next_state[3*self.n_uav:]) + [sojurn_t]

    return (next_state, obs, r)














