clear
close all
clc


fires_x = [-.5, 0.5, 0.9];
fires_y = [0.5, -0.8, -0.9];
fire_rew = [1,2,5];

uav_x = [-0.9, 0.9];
uav_y = [-0.9, 0.9];

hc = figure();
set(hc,'PaperUnits','Points');
set(hc,'PaperPosition',[650,550,350,300]);
set(hc,'Units','Points');
set(hc,'Position',[650,550,350,300]);




plot(uav_x, uav_y, '*', 'LineWidth', 5);
hold on
for i = 1:length(fires_x)
    plot(fires_x(i), fires_y(i), 'rx', 'LineWidth',fire_rew(i));
end

grid on
legend 'UAV Start Location' 'Fire' 'Location' 'Best'

print -dpng -r300 'GameLayout.png'

%%
load 3_fires_2_agents.mat

avg_mystrategy = 6.38456804978; 
std_mystrategy = 0.298271833632;

hc = figure();
set(hc,'PaperUnits','Points');
set(hc,'PaperPosition',[650,550,350,300]);
set(hc,'Units','Points');
set(hc,'Position',[650,550,350,300]);

plot(Iteration, AverageDiscountedReturn);
hold on
C = [0.2123    0.2138    0.6270];
plot([0,74],[avg_mystrategy, avg_mystrategy]); 
plot([0,74],[avg_mystrategy, avg_mystrategy]+std_mystrategy,'--','Color',C )
plot([0,74],[avg_mystrategy, avg_mystrategy]-std_mystrategy,'--','Color',C )

grid on

xlabel('Iteration')
ylabel('Avg Discounted Return')
legend 'Learned Policy' 'Intuitive Policy' 'Location' 'Best'
title '3 Fires, 2 UAVs'

print -dpng -r300 'TrainingCurve.png'

%%

fires_x = [-0.8, -0.8, 0.8, 0.8,-0.5, 0.5, 0];
fires_y = [-0.8, 0.8, -0.8, 0.8, 0, 0, 0 ];
fire_rew = [1,1,1,1,5,5,20];

uav_x = [-0.9, -0.9, 0.9, 0.9];
uav_y = [-0.9, 0.9, -0.9, 0.9];

hc = figure();
set(hc,'PaperUnits','Points');
set(hc,'PaperPosition',[650,550,350,300]);
set(hc,'Units','Points');
set(hc,'Position',[650,550,350,300]);




plot(uav_x, uav_y, '*', 'LineWidth', 5);
hold on
for i = 1:length(fires_x)
    plot(fires_x(i), fires_y(i), 'rx', 'LineWidth',fire_rew(i));
end

grid on
legend 'UAV Start Location' 'Fire' 'Location' 'Best'

print -dpng -r300 'GameLayout_7F4U.png'

%%
load 7fires_4agents.mat

avg_mystrategy = 28.8299560584;
std_mystrategy = 0.946267293438;

hc = figure();
set(hc,'PaperUnits','Points');
set(hc,'PaperPosition',[650,550,350,300]);
set(hc,'Units','Points');
set(hc,'Position',[650,550,350,300]);

plot(Iteration, AverageDiscountedReturn);
hold on
C = [0.2123    0.2138    0.6270];
plot([0,149],[avg_mystrategy, avg_mystrategy]); 
plot([0,149],[avg_mystrategy, avg_mystrategy]+std_mystrategy,'--','Color',C )
plot([0,149],[avg_mystrategy, avg_mystrategy]-std_mystrategy,'--','Color',C )

clear Iteration Average*
load 7fires_4agents_2.mat

plot(Iteration(1:150), AverageDiscountedReturn(1:150), 'Color', [0.2    0.6470    0.9410]);

clear Iteration Average*
load 7fires_4agents_3.mat

plot(Iteration, AverageDiscountedReturn, 'Color', [0    0.2470    0.5410]);


grid on

xlabel('Iteration')
ylabel('Avg Discounted Return')
legend 'Learned Policy' 'Intuitive Policy' 'Location' 'Best'
title '7 Fires, 4 UAVs'

print -dpng -r300 'TrainingCurve_7F4U.png'
