

clear
close all
clc

%% Newest Env 10F20U


load ./Newest_10U20F_GRU/progress.mat

avg_mystrategy = 21.8744500769;
std_mystrategy = 3.33521673323;

avg_naive = 30.7281558151; 
std_naive = 1.12569733771;

hc = figure();
set(hc,'PaperUnits','Points');
set(hc,'PaperPosition',[650,550,350,300]);
set(hc,'Units','Points');
set(hc,'Position',[650,550,350,300]);

plot(Iteration, AverageDiscountedReturn);
hold on
C = [0.2123    0.2138    0.6270];
plot([0,max(Iteration)],[avg_mystrategy, avg_mystrategy]);
plot([0,max(Iteration)],[avg_naive, avg_naive]);
plot([0,max(Iteration)],[avg_mystrategy, avg_mystrategy]+std_mystrategy,'--','Color',C )
plot([0,max(Iteration)],[avg_mystrategy, avg_mystrategy]-std_mystrategy,'--','Color',C )

plot([0,max(Iteration)],[avg_naive, avg_naive]+std_naive,'--','Color',C )
plot([0,max(Iteration)],[avg_naive, avg_naive]-std_naive,'--','Color',C )
grid on

set(gca,'xlim',[0,500]);

xlabel('Iteration')
ylabel('Avg Discounted Return')
legend 'Learned MLP+GRU Policy' 'Largest Need Policy' 'Closest Fire Policy' 'Location' 'Best'
title '20 Fires, 10 UAVs'

print -dpng -r300 'TrainingCurve_Newest_10U20F.png'
