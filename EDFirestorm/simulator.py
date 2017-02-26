import tensorflow as tf
import numpy as np
import random
import time
from math import exp
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Merge
from keras.optimizers import RMSprop, Adam

start_time = time.time()


class UAV_fire_extinguish(object):

  n_w = 4 # width of grid world
  n_uav = 2 # number of agents
  n_fire = 3 # number of fires
  u_loca = [0, 15] # initial location of agents
  t_fail = [0.02, 0.04] # probability for automatical failure
  t_emit = [0.5,   0.5] # probability for getting observation
  l_fire = [2,    7,     12] # location of fires
  r_fire = [5.0, 5.0, 50.0] # reward of putting down each fire
  e_fire = [[0.9,0.9],
            [0.9,0.9],
            [0.0,0.9]] # fire extinguish probability for each fire [down by 1 agent, down by 2 agent]

  l_bigf = [12] # location of big fire
  l_smlf = [2,7] # location of small fire
  s_init = u_loca + [1]*n_fire + [1]*n_uav # initial state of the system

  n_visi = 3 # length of local vision




##### Sampling method #####

def sampling_events(event,prob):

  n_length = len(event)

  x_rand = np.random.random()

  for i in range(n_length):

    x_rand = x_rand - prob[i]

    if x_rand <= 0:

      return event[i]


def mix_distribution(event1,prob1,event2,prob2):

  n_length_1 = len(event1)
  n_length_2 = len(event2)

  new_event = []
  new_prob = []

  for e1 in range(n_length_1):
    for e2 in range(n_length_2):
      e_new = event1[e1] + [event2[e2]]
      new_event.append(e_new)
      p_new = prob1[e1] * prob2[e2]
      new_prob.append(p_new)

  return (new_event,new_prob)

##### check boundary #####

def check_boundary(x,w):

  if x < 0:
    return 0
  elif x > w-1:
    return w-1
  else:
    return x

##################################
##### Mapping between states #####
##################################

def two_dim_to_one(l_cor,n_w):

  x = l_cor[0]
  y = l_cor[1]

  l = n_w * y + x

  return l

def one_dim_to_two(l,n_w):

  x = l%n_w
  y = (l-x)/n_w

  return [x,y]


############################
##### TRANSITION MODEL #####
############################

### simple movement of one agent due to action

def move_location_single(l_1d,a,n_w):

  if l_1d == n_w * n_w:

    return l_1d


  l = one_dim_to_two(l_1d,n_w)

  x_next = l[0]
  y_next = l[1]

  if a == 0: # up
    y_next = y_next + 1
  elif a == 1: # down
    y_next = y_next - 1
  elif a == 2: # left
    x_next = x_next - 1
  elif a == 3:
    x_next = x_next + 1
  else:
    pass


  x_next = check_boundary(x_next,n_w)
  y_next = check_boundary(y_next,n_w)

  l_next = two_dim_to_one((x_next,y_next),n_w)

  return l_next

######################################################
##### number of uavs at the location of the fire #####
######################################################

def fire_has_uavs(lf,l_tuple,n_cut):

  num = 0

  for i in range(len(l_tuple)):

      if lf == l_tuple[i]:

        num += 1


  if num > n_cut:

    num = n_cut

  return num

######################################################################
##### Obtain all possible sets and the corresponding probability #####
######################################################################

def transition_model(sys_cart_product_and_time_delay,a_joint,UAV_fire_extinguish):

  s_fail = UAV_fire_extinguish.n_w * UAV_fire_extinguish.n_w

  cart_product = sys_cart_product_and_time_delay[0: UAV_fire_extinguish.n_uav + UAV_fire_extinguish.n_fire]

  time_delay = sys_cart_product_and_time_delay[UAV_fire_extinguish.n_uav + UAV_fire_extinguish.n_fire :]

  ##### Terminal states #####

  die_product = 1

  ### if all agents are broken ###

  for i_uav in range(UAV_fire_extinguish.n_uav):

    if cart_product[i_uav] == s_fail:

      die_product = die_product * 1

    else:

      die_product = die_product * 0


  if die_product == 1:

    return ([UAV_fire_extinguish.u_loca + [1]*UAV_fire_extinguish.n_fire], [1.0], [1]*UAV_fire_extinguish.n_uav)

  ### if all fires are extinguished ###

  if sum(cart_product[UAV_fire_extinguish.n_uav:UAV_fire_extinguish.n_uav + UAV_fire_extinguish.n_fire]) == 0:

    return ([UAV_fire_extinguish.u_loca + [1]*UAV_fire_extinguish.n_fire], [1.0], [1]*UAV_fire_extinguish.n_uav)


  ##### Transition of the first UAV #####

  if cart_product[0] == s_fail:

    event_product = [[s_fail]]
    prob_product = [1.0]

  else:

    l0_next = move_location_single(cart_product[0],a_joint[0],UAV_fire_extinguish.n_w)
    event_product = [[l0_next],[s_fail]]
    prob_product =  [1.0 - UAV_fire_extinguish.t_fail[0], UAV_fire_extinguish.t_fail[0]]

  ##### Transition of the second UAV #####

  for i_uav in range(1,UAV_fire_extinguish.n_uav):

    if cart_product[i_uav] == s_fail:

      event_set_1 = [s_fail]
      prob_set_1 = [1.0]

    else:

      l1_next = move_location_single(cart_product[i_uav],a_joint[i_uav],UAV_fire_extinguish.n_w)
      event_set_1 = [l1_next,s_fail]
      prob_set_1 =  [1.0 - UAV_fire_extinguish.t_fail[i_uav], UAV_fire_extinguish.t_fail[i_uav]]

    (event_product,prob_product) = mix_distribution(event_product,prob_product,event_set_1,prob_set_1)


  ##### Transition of the fire states #####

  for i_fire in range(UAV_fire_extinguish.n_fire):

    the_fire_state = cart_product[UAV_fire_extinguish.n_uav + i_fire]

    if the_fire_state == 0: # no fire

      (event_product,prob_product) = mix_distribution(event_product,prob_product,[0],[1.0])

    else:

      l_f = UAV_fire_extinguish.l_fire[i_fire]
      l_0 = cart_product[0]
      l_1 = cart_product[1]

      if fire_has_uavs(l_f,cart_product[0:UAV_fire_extinguish.n_uav],2) == 1:

        rate_put_down = UAV_fire_extinguish.e_fire[i_fire][0]
        (event_product,prob_product) = mix_distribution(event_product,prob_product,[0,1],[rate_put_down,1.0-rate_put_down])

      elif fire_has_uavs(l_f,cart_product[0:UAV_fire_extinguish.n_uav],2) == 2:

        rate_put_down = UAV_fire_extinguish.e_fire[i_fire][1]
        (event_product,prob_product) = mix_distribution(event_product,prob_product,[0,1],[rate_put_down,1.0-rate_put_down])

      else:

        (event_product,prob_product) = mix_distribution(event_product,prob_product,[1],[1.0])


  ##### Consider the transition of time delay (Poisson Process) #####

  for i_uav in range(UAV_fire_extinguish.n_uav):

    random_p = random.random()

    if random_p < UAV_fire_extinguish.t_emit[i_uav]:

      time_delay[i_uav] = 1

    else:

      time_delay[i_uav] = time_delay[i_uav] + 1



  return (event_product,prob_product,time_delay)


def global_observation(agent,sys_state,UAV_fire_extinguish):

  s_fail = UAV_fire_extinguish.n_w * UAV_fire_extinguish.n_w

  o_length = 2 * UAV_fire_extinguish.n_uav + UAV_fire_extinguish.n_fire + 1 + 1
  #          (x,y) coordinate of each agent + fire status of each fire + agent ID + time_delay

  obs = ([agent] +
        [0] * ( 2 * UAV_fire_extinguish.n_uav) +
        sys_state[UAV_fire_extinguish.n_uav: UAV_fire_extinguish.n_uav + UAV_fire_extinguish.n_fire] +
        [sys_state[UAV_fire_extinguish.n_uav + UAV_fire_extinguish.n_fire + agent]])

  for j_agent in range(UAV_fire_extinguish.n_uav):

    [x,y] = one_dim_to_two(sys_state[j_agent],UAV_fire_extinguish.n_w)

    obs[1 + 2*j_agent] = x
    obs[2 + 2*j_agent] = y

  return obs



def local_observation(agent,sys_state,UAV_fire_extinguish):

  s_fail = UAV_fire_extinguish.n_w * UAV_fire_extinguish.n_w

  # agent = which agent is going to make the observation

  vision_depth = UAV_fire_extinguish.n_visi

  vision_area = (vision_depth * 2 + 1) ** 2

  self_location_xy = one_dim_to_two(sys_state[agent],UAV_fire_extinguish.n_w)

  # vision 1: other agents

  vision_1 = [0]*vision_area

  for other_agent in range(UAV_fire_extinguish.n_uav):

    if other_agent != agent :

        location_other_agent = sys_state[other_agent]
        location_other_xy = one_dim_to_two(location_other_agent,UAV_fire_extinguish.n_w)

        dx = location_other_xy[0] - self_location_xy[0]
        dy = location_other_xy[1] - self_location_xy[1]

        if (-1)*vision_depth <= dx <= vision_depth and (-1)*vision_depth <= dy <= vision_depth and sys_state[other_agent] != s_fail:

          relative_location = two_dim_to_one((dx + vision_depth,dy + vision_depth), vision_depth * 2 + 1)

          vision_1[relative_location] += 1

  # vision 2: big fires

  vision_2 = [0]*vision_area

  # vision 3: small fires

  vision_3 = [0]*vision_area

  for i_fire in range(UAV_fire_extinguish.n_fire):

    if sys_state[UAV_fire_extinguish.n_uav + i_fire] == 1:

      if UAV_fire_extinguish.l_fire[i_fire] in UAV_fire_extinguish.l_bigf: # it is a big fire

        big_location = one_dim_to_two(UAV_fire_extinguish.l_fire[i_fire],UAV_fire_extinguish.n_w)

        dx = big_location[0] - self_location_xy[0]
        dy = big_location[1] - self_location_xy[1]

        if (-1)*vision_depth <= dx <= vision_depth and (-1)*vision_depth <= dy <= vision_depth:

          relative_location = two_dim_to_one((dx + vision_depth,dy + vision_depth), vision_depth * 2 + 1)
          vision_2[relative_location] += 1

      else: # it is a small fire


        sml_location = one_dim_to_two(UAV_fire_extinguish.l_fire[i_fire],UAV_fire_extinguish.n_w)

        dx = sml_location[0] - self_location_xy[0]
        dy = sml_location[1] - self_location_xy[1]

        if (-1)*vision_depth <= dx <= vision_depth and (-1)*vision_depth <= dy <= vision_depth:

          relative_location = two_dim_to_one((dx + vision_depth,dy + vision_depth), vision_depth * 2 + 1)
          vision_3[relative_location] += 1


  time_delay = sys_state[UAV_fire_extinguish.n_uav + UAV_fire_extinguish.n_fire + agent]

  return (([agent] + self_location_xy + [time_delay]),(vision_1),(vision_2),(vision_3))



def transition_sample(
  current_state,
  a_joint, # tuple
  info_list, # [info_1,info_2,....]
  UAV_fire_extinguish):

  n_w = UAV_fire_extinguish.n_w

  reward = 0.0

  (event,prob,time_delay) = transition_model(current_state,a_joint,UAV_fire_extinguish)


  next_state = sampling_events(event,prob) + time_delay


  # Collect rewards

  for i_fire in range(UAV_fire_extinguish.n_fire):

    if current_state[UAV_fire_extinguish.n_uav + i_fire] == 1 and next_state[UAV_fire_extinguish.n_uav + i_fire] == 0:

      reward += UAV_fire_extinguish.r_fire[i_fire]


  # Update information if time delay is 1.0

  updated_info_list = info_list[:]

  for i_agent in range(UAV_fire_extinguish.n_uav):

    if next_state[UAV_fire_extinguish.n_uav + UAV_fire_extinguish.n_fire + i_agent] == 1:

      updated_info_list[i_agent] = global_observation(i_agent,next_state,UAV_fire_extinguish)

    else:

      #updated_info_list[i_agent][3] = updated_info_list[i_agent][3] + 1
      updated_info_list[i_agent][-1] = updated_info_list[i_agent][-1] + 1


  return [next_state,updated_info_list,reward]



######## CODE FOR SIMULATOR IS FINISHED #########
######## CODE FOR SIMULATOR IS FINISHED #########
######## CODE FOR SIMULATOR IS FINISHED #########
######## CODE FOR SIMULATOR IS FINISHED #########
######## CODE FOR SIMULATOR IS FINISHED #########
######## CODE FOR SIMULATOR IS FINISHED #########



def samples_by_random_action(n_init_pool,UAV_fire_extinguish):

  size = UAV_fire_extinguish.n_w

  input_number = 4 + 3 *(2 * UAV_fire_extinguish.n_visi + 1)**2

  o_pool = np.zeros((UAV_fire_extinguish.n_uav,n_init_pool,input_number),float)
  a_pool = np.zeros((UAV_fire_extinguish.n_uav,n_init_pool,5),float)
  r_pool = np.zeros((UAV_fire_extinguish.n_uav,n_init_pool,1),float)
  op_pool = np.zeros((UAV_fire_extinguish.n_uav,n_init_pool,input_number),float)


  s_current = UAV_fire_extinguish.s_init
  last_info_list = []

  for i_uav in range(UAV_fire_extinguish.n_uav):

    last_info_list.append(local_observation(i_uav,s_current,UAV_fire_extinguish))

    #print(last_info_list[i_uav])

  next_info_list = last_info_list[:]


  for i_event in range(n_init_pool):

    a_joint = [0] * UAV_fire_extinguish.n_uav

    for i_uav in range(UAV_fire_extinguish.n_uav):
      a_joint[i_uav] = random.randint(0,4)

    #print(s_current,a_joint)

    outcome = transition_sample(s_current,a_joint,last_info_list,UAV_fire_extinguish)

    next_state = outcome[0]
    next_info_list = outcome[1]
    reward = outcome[2]


    for i_uav in range(UAV_fire_extinguish.n_uav):

      o_pool[i_uav,i_event,:] = last_info_list[i_uav][:]
      op_pool[i_uav,i_event,:] = next_info_list[i_uav][:]
      a_pool[i_uav,i_event,a_joint[i_uav]] = 1.0
      r_pool[i_uav,i_event,0] = reward

    last_info_list = next_info_list[:]
    s_current = next_state

  return (o_pool,a_pool,r_pool,op_pool)




def samples_by_one_agent_random_action(n_init_pool,free_agent,UAV_fire_extinguish):

  size = UAV_fire_extinguish.n_w

  input_number = 4 + 3 *(2 * UAV_fire_extinguish.n_visi + 1)**2

  o_pool = np.zeros((UAV_fire_extinguish.n_uav,n_init_pool,input_number),float)
  a_pool = np.zeros((UAV_fire_extinguish.n_uav,n_init_pool,5),float)
  r_pool = np.zeros((UAV_fire_extinguish.n_uav,n_init_pool,1),float)
  op_pool = np.zeros((UAV_fire_extinguish.n_uav,n_init_pool,input_number),float)


  s_current = UAV_fire_extinguish.s_init
  last_info_list = []

  for i_uav in range(UAV_fire_extinguish.n_uav):

    last_info_list.append(local_observation(i_uav,s_current,UAV_fire_extinguish))


  next_info_list = last_info_list[:]


  for i_event in range(n_init_pool):


    a_joint = [0] * UAV_fire_extinguish.n_uav

    for i_uav in range(UAV_fire_extinguish.n_uav):
      if i_uav == free_agent:
        a_joint[i_uav] = random.randint(0,4)
      else:
        a_joint[i_uav] = es_greedy(sess.run(Q, feed_dict={last_info: [last_info_list[i_uav]]}),0.0)


    outcome = transition_sample(s_current,a_joint,last_info_list,UAV_fire_extinguish)

    next_state = outcome[0]
    next_info_list = outcome[1]
    reward = outcome[2]


    for i_uav in range(UAV_fire_extinguish.n_uav):

      o_pool[i_uav,i_event,:] = last_info_list[i_uav][:]
      op_pool[i_uav,i_event,:] = next_info_list[i_uav][:]
      a_pool[i_uav,i_event,a_joint[i_uav]] = 1.0
      r_pool[i_uav,i_event,0] = reward

    last_info_list = next_info_list[:]
    s_current = next_state

  return (o_pool,a_pool,r_pool,op_pool)


def truncate_dataset_multiagent(data_array,n_keep_size):

  n_size = len(data_array[0])

  if n_size <= n_keep_size:
    return data_array
  else:
    return data_array[:,(n_size-n_keep_size):,:]


def batch_select_multiagent(inputs,n_uav,n_batch,seeds):

  batch_set = np.zeros((n_uav,n_batch,len(inputs[0][1])))

  for i in range(n_batch):
    for i_uav in range(n_uav):
      batch_set[i_uav,i,:] = inputs[i_uav,seeds[i],:]

  return batch_set


def visualize_scenario_indp(current_state,h_print,r_explore,UAV_fire_extinguish):

  last_info_list = []

  for i_uav in range(UAV_fire_extinguish.n_uav):

    last_info_list.append(local_observation(i_uav,current_state,UAV_fire_extinguish))


  next_info_list = last_info_list[:]


  for h in range(h_print):

    a_joint = [0] * UAV_fire_extinguish.n_uav

    for i_uav in range(UAV_fire_extinguish.n_uav):

      (obs_0,obs_1,obs_2,obs_3) = last_info_list[i_uav][:]

      old_qval = final_model.predict([np.array(obs_0).reshape(1,input_size_nn_sfinfo),
                                      np.array(obs_1).reshape(1,input_size_nn_vision),
                                      np.array(obs_2).reshape(1,input_size_nn_vision),
                                      np.array(obs_3).reshape(1,input_size_nn_vision)], batch_size=1)

      a_joint[i_uav] = es_greedy(old_qval,r_explore)


    outcome = transition_sample(current_state,a_joint,last_info_list,UAV_fire_extinguish)

    (next_state,next_info_list,reward_immed) = outcome

    next_state = outcome[0]
    next_info_list = outcome[1]
    reward = outcome[2]

    print(current_state,a_joint,reward)

    current_state = next_state
    last_info_list = next_info_list




##########################################
############ Neural Network ##############
##########################################

##### functions for nerual network #####

def es_greedy(inputs,epsi):

  x_rand = np.random.random()

  if x_rand < epsi:
    return np.random.randint(0,4)
  else:
    return np.argmax(inputs)

def softmax(inputs,T):

  x_rand = np.random.random()

  e_input = np.ones(len(inputs))

  for i in range(len(inputs)):

    e_input[i] = exp(inputs[i]/float(T))

  e_input = e_input/sum(e_input)

  e_input[-1] += 0.01

  for i in range(len(inputs)):

    if x < e_input[i]:

      return x

    else:

      x = x - e_input[i]

##### target value #####

#####################################################################
#####################################################################


input_size_nn_vision = (2 * UAV_fire_extinguish.n_visi + 1)**2
input_size_nn_sfinfo = 4

self_info_branch = Sequential()
self_info_branch.add(Dense(10, init='lecun_uniform', input_shape = (input_size_nn_sfinfo,)))
self_info_branch.add(Activation('relu'))

other_vision_branch = Sequential()
other_vision_branch.add(Dense(50, init='lecun_uniform', input_shape = (input_size_nn_vision,)))
other_vision_branch.add(Activation('relu'))
other_vision_branch.add(Dense(50, init='lecun_uniform'))
other_vision_branch.add(Activation('relu'))

smalf_vision_branch = Sequential()
smalf_vision_branch.add(Dense(50, init='lecun_uniform', input_shape = (input_size_nn_vision,)))
smalf_vision_branch.add(Activation('relu'))
smalf_vision_branch.add(Dense(50, init='lecun_uniform'))
smalf_vision_branch.add(Activation('relu'))

bigf_vision_branch = Sequential()
bigf_vision_branch.add(Dense(50, init='lecun_uniform', input_shape = (input_size_nn_vision,)))
bigf_vision_branch.add(Activation('relu'))
bigf_vision_branch.add(Dense(50, init='lecun_uniform'))
bigf_vision_branch.add(Activation('relu'))

merged = Merge([self_info_branch, other_vision_branch, smalf_vision_branch, bigf_vision_branch], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Activation('relu'))
final_model.add(Dense(5,init='lecun_uniform'))
final_model.add(Activation('linear'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
final_model.compile(loss='mse', optimizer=adam)

##############################

epochs = 60000
gamma = 0.9
epsilon = 0.2
random_action_thresold = 1000
epsi = 0.2
max_pool_size = 20000
n_batch_size = 5000


current_state = UAV_fire_extinguish.s_init

last_info_list = []

for i_uav in range(UAV_fire_extinguish.n_uav):

  last_info_list.append(local_observation(i_uav,current_state,UAV_fire_extinguish))


for iteration_times in range(0): # 3 not 0

  obs_sfinfo = []
  obs_otheru = []
  obs_smallf = []
  obs_bigf   = []
  reward_list = []
  target_list = []

  if iteration_times == 0:

    pass

  elif iteration_times == 1:

    UAV_fire_extinguish.e_fire[2][0] = 0.2

  elif iteration_times == 2:

    UAV_fire_extinguish.e_fire[2][0] = 0.1

  else:

    UAV_fire_extinguish.e_fire[2][0] = 0.0

  print(UAV_fire_extinguish.e_fire[2][0])



  for ep in range(epochs):

    epsi = 0.1# - 0.2 * (ep / epochs)

    if ep % 100 == 0:

      print("iteration times = ",ep,"===============================")



    ###################################
    ########## Choose action ##########
    ###################################

    a_joint = [0] * UAV_fire_extinguish.n_uav

    if ep < random_action_thresold:

      for i_uav in range(UAV_fire_extinguish.n_uav):

        a_joint[i_uav] = random.randint(0,4)

    else:

      for i_uav in range(UAV_fire_extinguish.n_uav):

        (obs_0,obs_1,obs_2,obs_3) = last_info_list[i_uav][:]

        old_qval = final_model.predict([np.array(obs_0).reshape(1,input_size_nn_sfinfo),
                                      np.array(obs_1).reshape(1,input_size_nn_vision),
                                      np.array(obs_2).reshape(1,input_size_nn_vision),
                                      np.array(obs_3).reshape(1,input_size_nn_vision)], batch_size=1)

        a_joint[i_uav] = es_greedy(old_qval,epsi)

    #####################################
    ########## Make transition ##########
    #####################################

    outcome_transition = transition_sample(current_state,a_joint,last_info_list,UAV_fire_extinguish)

    next_state = outcome_transition[0]

    #############################################################
    ### Add observations and rewards into pool for all agents ###
    #############################################################

    for i_uav in range(UAV_fire_extinguish.n_uav):

      # add observations

      (obs_0,obs_1,obs_2,obs_3) = last_info_list[i_uav][:]

      obs_sfinfo.append(np.array(obs_0).reshape(1,input_size_nn_sfinfo))
      obs_otheru.append(np.array(obs_1).reshape(1,input_size_nn_vision))
      obs_smallf.append(np.array(obs_2).reshape(1,input_size_nn_vision))
      obs_bigf.append(np.array(obs_3).reshape(1,input_size_nn_vision))
      reward_list.append(outcome_transition[2])

      # add target value

      (obsp_0,obsp_1,obsp_2,obsp_3) = outcome_transition[1][i_uav][:]

      old_qval = final_model.predict([np.array(obs_0).reshape(1,input_size_nn_sfinfo),
                                      np.array(obs_1).reshape(1,input_size_nn_vision),
                                      np.array(obs_2).reshape(1,input_size_nn_vision),
                                      np.array(obs_3).reshape(1,input_size_nn_vision)], batch_size=1)

      new_qval = final_model.predict([np.array(obsp_0).reshape(1,input_size_nn_sfinfo),
                                      np.array(obsp_1).reshape(1,input_size_nn_vision),
                                      np.array(obsp_2).reshape(1,input_size_nn_vision),
                                      np.array(obsp_3).reshape(1,input_size_nn_vision)], batch_size=1)

      max_q_new = np.max(new_qval)

      y = np.zeros((1,5))
      y[:] = old_qval[:]
      y[0][a_joint[i_uav]] = outcome_transition[2] + gamma * max_q_new
      target_list.append(y)


    #########################################
    ### update next state and information ###
    #########################################

    current_state = next_state
    last_info_list = outcome_transition[1][:]

    ###########################################
    ### if we have too many samples in pool ###
    ###########################################

    if len(obs_sfinfo) > max_pool_size:

      obs_sfinfo.pop(0)
      obs_otheru.pop(0)
      obs_smallf.pop(0)
      obs_bigf.pop(0)
      reward_list.pop(0)

    ############################
    ### train neural network ###
    ############################

    if ep % 500 == 0 and ep > random_action_thresold:

      # create batch

      obs_0_array = np.zeros((n_batch_size,input_size_nn_sfinfo))
      obs_1_array = np.zeros((n_batch_size,input_size_nn_vision))
      obs_2_array = np.zeros((n_batch_size,input_size_nn_vision))
      obs_3_array = np.zeros((n_batch_size,input_size_nn_vision))
      targt_array = np.zeros((n_batch_size,5))

      if len(obs_sfinfo) > n_batch_size+1:

        seeds = random.sample(xrange(0,len(obs_sfinfo)),n_batch_size)

        for i_batch_sample in range(n_batch_size):

          b_number = seeds[i_batch_sample]

          obs_0_array[i_batch_sample,:] = obs_sfinfo[b_number][0][:]
          obs_1_array[i_batch_sample,:] = obs_otheru[b_number][0][:]
          obs_2_array[i_batch_sample,:] = obs_smallf[b_number][0][:]
          obs_3_array[i_batch_sample,:] = obs_bigf[b_number][0][:]
          targt_array[i_batch_sample,:] = target_list[b_number][0][:]


        # train

        final_model.fit([obs_0_array,obs_1_array,obs_2_array,obs_3_array],
                        targt_array,
                        batch_size = n_batch_size,
                        nb_epoch = 50,
                        verbose = 1)


  visualize_scenario_indp(UAV_fire_extinguish.s_init,30,0.2,UAV_fire_extinguish)
  print("=====================")
  visualize_scenario_indp(UAV_fire_extinguish.s_init,30,0.2,UAV_fire_extinguish)
  print("=====================")
  visualize_scenario_indp(UAV_fire_extinguish.s_init,30,0.2,UAV_fire_extinguish)
  print("=====================")
  visualize_scenario_indp(UAV_fire_extinguish.s_init,30,0.2,UAV_fire_extinguish)
  print("=====================")
  visualize_scenario_indp(UAV_fire_extinguish.s_init,30,0.2,UAV_fire_extinguish)
  print("=====================")




