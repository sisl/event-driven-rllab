%% 10U20F

clear
close all
clc



load 10U20F.mat

avg_mystrategy = 68.3250010153;
std_mystrategy = 6.04530597015;

hc = figure();
set(hc,'PaperUnits','Points');
set(hc,'PaperPosition',[650,550,350,300]);
set(hc,'Units','Points');
set(hc,'Position',[650,550,350,300]);

plot(Iteration, AverageDiscountedReturn);
hold on
plot(Iteration, AverageDiscountedReturn_GRU);
C = [0.2123    0.2138    0.6270];
plot([0,499],[avg_mystrategy, avg_mystrategy]); 
% plot([0,499],[avg_mystrategy, avg_mystrategy]+std_mystrategy,'--','Color',C )
plot([0,499],[avg_mystrategy, avg_mystrategy]-std_mystrategy,'--','Color',C )
grid on

xlabel('Iteration')
ylabel('Avg Discounted Return')
legend 'Learned MLP Policy' 'Learned MLP+GRU Policy' 'Intuitive Policy' 'Location' 'Best'
title '20 Fires, 10 UAVs'

print -dpng -r300 'TrainingCurve_10U20F.png'


%% New Env 10F20U


load ./New_10U20F_GRU/progress.mat

avg_mystrategy = 56.091376214;
std_mystrategy = 9.26602384617;

hc = figure();
set(hc,'PaperUnits','Points');
set(hc,'PaperPosition',[650,550,350,300]);
set(hc,'Units','Points');
set(hc,'Position',[650,550,350,300]);

plot(Iteration, AverageDiscountedReturn);
hold on
C = [0.2123    0.2138    0.6270];
plot([0,max(Iteration)],[avg_mystrategy, avg_mystrategy]); 
plot([0,max(Iteration)],[avg_mystrategy, avg_mystrategy]+std_mystrategy,'--','Color',C )
plot([0,max(Iteration)],[avg_mystrategy, avg_mystrategy]-std_mystrategy,'--','Color',C )
grid on

xlabel('Iteration')
ylabel('Avg Discounted Return')
legend 'Learned MLP+GRU Policy' 'Intuitive Policy' 'Location' 'Best'
title '20 Fires, 10 UAVs'

print -dpng -r300 'TrainingCurve_New_10U20F.png'
