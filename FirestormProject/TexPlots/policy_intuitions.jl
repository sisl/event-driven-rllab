#!/usr/bin/env julia

using DataFrames
using PGFPlots

df = readtable("../paths.csv")

chose_to_hold = df[ df[:Action] .== 5,:]

was_at_live_fire = df[ (df[:Dist0] .< 1e-7) & (df[:Status0] .==1),  : ];


println("Was at fire when tried to hold")
println(size(chose_to_hold[ (chose_to_hold[:Dist0] .< 1e-7), :])[1] / size(chose_to_hold)[1])

println("Tried to extinguish when at live fire")
println(size(was_at_live_fire[ was_at_live_fire[:Action] .== 5, :])[1] / size(was_at_live_fire)[1])

println("Tried to extinguish when at live fire with no one else")
was_at_live_fire_alone = was_at_live_fire[was_at_live_fire[:Interest0] .== 1, :]
println(size(was_at_live_fire_alone[ was_at_live_fire_alone[:Action] .== 5, :])[1] / size(was_at_live_fire_alone)[1])

println("Tried to extinguish when at live fire with more than one person")
was_at_live_fire_with_interest = was_at_live_fire[was_at_live_fire[:Interest0] .> 1, :]
println(size(was_at_live_fire_with_interest[ was_at_live_fire_with_interest[:Action] .== 5, :])[1] / size(was_at_live_fire_with_interest)[1])

println("Flew to already interested fire")
chose_to_fly = df[df[:Action] .!= 5, :]
len_chose_to_fly = size(chose_to_fly)[1]
went_to_already_interested = 0
for i = 1:len_chose_to_fly
  action = chose_to_fly[i,:Action]
  # println(chose_to_fly[i,:@sprintf("Interest%d",action)])
  interests = chose_to_fly[i,[:Interest0,:Interest1,:Interest2,:Interest3,:Interest4]]
  if (interests[action+1].>0)[1]
    went_to_already_interested+=1
  end
end
println(went_to_already_interested/len_chose_to_fly)



# println("Flew to live fire")
chose_to_fly = df[df[:Action] .!= 5, :]
# print(chose_to_fly[:, [:Status0,:Status1,:Status2,:Status3,:Status4,:Action]])
len_chose_to_fly = size(chose_to_fly)[1]
went_to_already_dead = 0
had_other_option = 0
fires_chosen = [0 0 0 0 0]
for i = 1:len_chose_to_fly
  action = chose_to_fly[i,:Action]
  fires_chosen[action+1] += 1
  # println(chose_to_fly[i,:@sprintf("Interest%d",action)])
  interests = chose_to_fly[i,[:Interest0,:Interest1,:Interest2,:Interest3,:Interest4]]
  statuses = chose_to_fly[i,[:Status0,:Status1,:Status2,:Status3,:Status4]]
  # println(interests)
  # println(statuses)
  # print(action)
  if ( statuses[action+1].==0 )[1]
    went_to_already_dead+=1
    other_option = false
    for i = 1:5
      other_option = other_option | ( (statuses[i].==1)[1] & (interests[i].==0)[1] )
    end

    if(other_option)
      had_other_option += 1
    end

  end
end
println("")
println("Chose a fire that was dead")
println(went_to_already_dead/len_chose_to_fly)
println("Chose a fire that was dead but there was another fire still alive")
println(had_other_option / went_to_already_dead)
println(fires_chosen / len_chose_to_fly)
