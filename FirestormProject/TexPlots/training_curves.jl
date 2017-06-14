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

means_stds = [
2.77967987039 0.0136182751054;
2.77116278061 0.0179833271993;
2.74452130859 0.0153417433698;
2.78854871988 0.00882454950388;
2.21184437795 0.0820432066556;
1.8413277904 0.179313897733;
1.93681589969 0.0410598352529;
]

ax = PGFPlots.Axis(style="grid=both",
  xlabel="Training Simulator", ylabel="Average Discounted Return",)
  # legendStyle = "{at={(0.05,0.15)},anchor=north west}",
  # xtick="{1,2,3,4,5,6}",
  # xticklabels="{ED, FS-$10^{-1.0}$}, FS-$10^{-0.5}$}, FS-$10^{0}$}, FS-$10^{0.5}$}, FS-$10^{1.0}$}")

push!(ax,
  Plots.Linear([4], [means_stds[7,1]], errorBars = ErrorBars(y=[means_stds[7,2]]), style="black", mark="black"))
  # , legendentry=L"$T_{health} = 2.9999$"))


push!(ax,
  Plots.Linear([1], [means_stds[1,1]], errorBars = ErrorBars(y=[means_stds[1,2]]), style="black", mark="black"))

push!(ax, Plots.Linear( [2,3,4,5,6], means_stds[2:6,1], errorBars = ErrorBars(y=means_stds[2:6,2]),
  onlyMarks=true, style="black", mark = "black"))

push!(ax, Plots.Node(L"\scriptsize{$T_{health} = 2.9999$}",2,1.94))


save("transferperformance.pdf", ax)
save("transferperformance.tex", ax, include_preamble = false)
