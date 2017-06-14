#!/usr/bin/env julia

using DataFrames
using PGFPlots

# datafile = "rlpol"
# datafile = "optim"
datafile = "nohold"

titles = Dict("optim"=> "Optimized Thresholds",
          "rlpol"=> "Optimized GRU Policy",
          "nohold"=> "No Holding")

title = titles[datafile]

df = readtable("../data/"*datafile*".csv")

marks = ["o", "triangle", "asterisk", "+", "x"]


ax = PGFPlots.Axis(style="grid=both",
  xlabel="Arrival Time [mins]", ylabel="Stop",
  title=title);
for i = 1:5
  push!(ax, Plots.Linear(df[df[:Bus] .== i-1, :][:Arrival_Time]/60,
   df[df[:Bus] .== i-1, :][:Stop],
  #  mark=marks[i],
   mark = "|",
   onlyMarks=true,
   style = "black"
   ))
end

# p = Axis(ax, )

save(datafile*"_arriv.tex", ax, include_preamble=false)

ax = PGFPlots.Axis(style="grid=both",
  xlabel="Stop", ylabel="Load at Arrival [pax]",
  title=title);
for i = 1:5
  # df_sub = df[(df[:Bus] .== i-1) & (df[:Arrival_Time] .>13000), :]
  df_sub = df[(df[:Bus] .== i-1), :]
  push!(ax, Plots.Linear(df_sub[4*10:5*10,:Stop],
   df_sub[4*10:5*10,:Load],
  #  mark=marks[i],
   mark = "o",
   onlyMarks=true,
   style = "black"
   ))
end

save(datafile*"_load.tex", ax, include_preamble=false)
