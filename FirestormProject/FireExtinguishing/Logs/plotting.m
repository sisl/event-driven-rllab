

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

avg_mystrategy = 10.9093980249;
std_mystrategy = 2.08991266819;

avg_naive = 10.4368430991; 
std_naive = 1.74412980459;

%% EVEN Newer Env 10F20U


load ./EVEN_Newer_10U20F_GRU/progress.mat

avg_mystrategy = 10.9093980249;
std_mystrategy = 2.08991266819;

avg_naive = 10.4368430991; 
std_naive = 1.74412980459;

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

print -dpng -r300 'TrainingCurve_EVEN_Newer_10U20F.png'

%% DistObs_WithGamma_3U6F_GRU


load ./DistObs_WithGamma_3U6F_GRU/progress.mat

avg_mystrategy = 3.72260280688;
std_mystrategy = 1.08297925885;

avg_naive = 2.96048767776; 
std_naive = 0.832976973466;

hc = figure();
set(hc,'PaperUnits','Points');
set(hc,'PaperPosition',[650,550,350,300]);
set(hc,'Units','Points');
set(hc,'Position',[650,550,350,300]);

plot(Iteration, AverageReturn);
hold on
C1 = [0.8500    0.3250    0.0980];
C2 = [0.9290    0.6940    0.1250];
plot([0,max(Iteration)],[avg_mystrategy, avg_mystrategy]);
plot([0,max(Iteration)],[avg_naive, avg_naive]);
plot([0,max(Iteration)],[avg_mystrategy, avg_mystrategy]+std_mystrategy,'--','Color',C1 )
plot([0,max(Iteration)],[avg_mystrategy, avg_mystrategy]-std_mystrategy,'--','Color',C1 )

plot([0,max(Iteration)],[avg_naive, avg_naive]+std_naive,'--','Color',C2 )
plot([0,max(Iteration)],[avg_naive, avg_naive]-std_naive,'--','Color',C2 )
grid on

set(gca,'xlim',[0,100]);

xlabel('Iteration')
ylabel('Avg Discounted Return')
legend 'Learned MLP+GRU Policy' 'Largest Need Policy' 'Closest Fire Policy' 'Location' 'Best'
title '6 Fires, 3 UAVs'

print -dpng -r300 'TrainingCurve_EVEN_Newer_10U20F.png'