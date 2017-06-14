#!/usr/bin/env julia

using DataFrames
using PGFPlots

filenames = ["experiment_2017_05_30_gae_1",
		"experiment_2017_05_30_gae_2",
		# "experiment_2017_05_30_gae_4",
		"experiment_2017_05_30_gae_8"
		# "experiment_2017_05_30_gae_16"
		]
		# "experiment_2017_06_01_gae_2_with_featnet" ]

define_color("C1", [0,    0,    0])
define_color("C2", [0.3,    0.3,    0.3])
define_color("C3", [0.6,    0.6,    0.6])
define_color("C4", [0.,    0.,    0.])
define_color("C5", [0.4,    0.4,    0.4])


nohold_mean = -4.53976e+06;
optim_mean = -1.35403e+06;

legs = [L"$\lambda=1.0$", L"$\lambda=2.0$", L"$\lambda=4.0$",
 L"$\lambda=8.0$", L"$\lambda=16.0$", # L"$\lambda=2.0+$FeatureNet",
 "Optimized Thresholds", "No Holding"]

ax = PGFPlots.Axis(style="grid=both",
  xlabel="Training Epoch", ylabel="Average Return", legendStyle = "{at={(0.55,0.75)},anchor=north west}");

define_color("linecol", [0.3,0.3,0.3])

for i = 1:length(filenames)
  fn = filenames[i]
  df = readtable("../data/"*fn*"/progress.csv")
  push!(ax, Plots.Linear(df[:Iteration], df[:AverageReturn],legendentry=legs[i],
  mark = "",
	style = @sprintf("C%d",i)
  # markSize = 0.8, style = "linecol"
  ))
end

push!(ax, Plots.Linear([0,499], [optim_mean,optim_mean], legendentry=legs[end-1], mark = "", style = "dashed, C4")) # markSize=1 ))
push!(ax, Plots.Linear([0,499], [nohold_mean,nohold_mean], legendentry=legs[end], mark = "", style = "dashed, C5")) # markSize=1 ))


save("bbtraining.pdf", ax)
save("bbtraining.tex", ax, include_preamble=false)
