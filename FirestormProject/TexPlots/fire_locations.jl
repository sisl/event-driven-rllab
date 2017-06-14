#!/usr/bin/env julia

using DataFrames
using PGFPlots

define_color("C1", [0,    0.4470,    0.7410])
define_color("C2", [0.8500,    0.3250,    0.0980])
define_color("C3", [0.9290,    0.6940,    0.1250])
define_color("C4", [0.4940,    0.1840,    0.5560])
define_color("C5", [0.4660,    0.6740,    0.1880])
define_color("C6", [0.3010,    0.7450,    0.9330])
define_color("C7", [0.6350,    0.0780,    0.1840])


f = [-0.01 0.01 0; 1 1 1-0.02*sin(deg2rad(60))]
R = [cos(deg2rad(120)) -sin(deg2rad(120)) ; sin(deg2rad(120)) cos(deg2rad(120))]
f = [f R*f R'*f]

fire_x = f[1,:]
fire_y = f[2,:]


ax = PGFPlots.Axis(style="grid=both",
  xlabel="x [m]", ylabel="y [m]", legendStyle = "{at={(0.55,0.6)},anchor=north west}");
push!(ax, Plots.Linear(fire_x, fire_y,legendentry="Fire Locations",
mark = "x",
onlyMarks = true,
# markSize = 0.8,
style = "black"
))
#
#
# push!(ax, Plots.Linear([0,499], [optim_mean,optim_mean], legendentry=legs[end-1], mark = "")) # markSize=1 ))
# push!(ax, Plots.Linear([0,499], [nohold_mean,nohold_mean], legendentry=legs[end], mark = "")) # markSize=1 ))
#
#
save("firelocs.pdf", ax)
save("firelocs.tex", ax, include_preamble=false)
