#!/usr/bin/env julia

using DataFrames
using PGFPlots

filenames = [
		"experiment_2017_05_30_gae_2",
		"experiment_2017_06_02_batchsize_1200",
		"experiment_2017_06_05_batchsize_120_3"]
		# "experiment_2017_06_05_batchsize_12" ]
		# "experiment_2017_06_01_gae_2_with_featnet" ]

filenames = [
		# "experiment_2017_06_02_batchsize_120",
		"experiment_2017_05_30_gae_2",
		# "experiment_2017_06_02_batchsize_1200",
		"experiment_2017_06_05_batchsize_120_3",
		# "experiment_2017_06_05_batchsize_120_2",
		"experiment_2017_06_05_batchsize_120"
		 ]


nohold_mean = -4.53976e+06;
optim_mean = -1.35403e+06;

legs = ["BS-12000", "BS-1200", "BS-120", "BS-12", "BS", "Optimized Thresholds", "No Holding"]

ax = PGFPlots.Axis(style="grid=both",
  xlabel="Training Epoch", ylabel="Average Return", legendStyle = "{at={(0.55,0.6)},anchor=north west}");

# define_color("linecol", [0.3,0.3,0.3])
# define_color("C1", [0,    0.4470,    0.7410])
# define_color("C2", [0.8500,    0.3250,    0.0980])
# define_color("C3", [0.9290,    0.6940,    0.1250])
# define_color("C4", [0.4940,    0.1840,    0.5560])
# define_color("C5", [0.4660,    0.6740,    0.1880])
# define_color("C6", [0.3010,    0.7450,    0.9330])
# define_color("C7", [0.6350,    0.0780,    0.1840])
define_color("C1", [0,    0,    0])
define_color("C2", [0.3,    0.3,    0.3])
define_color("C3", [0.6,    0.6,    0.6])
define_color("C4", [0.,    0.,    0.])
define_color("C5", [0.4,    0.4,    0.4])



for i = 1:length(filenames)
  fn = filenames[i]
  df = readtable("../data/"*fn*"/progress.csv")
	mean_num_trajs = floor(mean(df[:NumTrajs])/10)*10
  push!(ax, Plots.Linear(df[:Iteration], df[:AverageReturn],legendentry=@sprintf("%d Episodes/Epoch",mean_num_trajs),
  mark = "",
  # markSize = 0.8,
	style = @sprintf("C%d",i)
  ))
	# println(mean(df[:NumTrajs]))
end

push!(ax, Plots.Linear([0,499], [optim_mean,optim_mean], legendentry=legs[end-1], mark = "", style="dashed, C4")) # markSize=1 ))
push!(ax, Plots.Linear([0,499], [nohold_mean,nohold_mean], legendentry=legs[end], mark = "", style="dashed, C5")) # markSize=1 ))


save("bstest.pdf", ax)
save("bbtest.tex", ax, include_preamble=false)
