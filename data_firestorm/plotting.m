clear
close all
clc

load test_policy_results.mat
% S = dataloader('experiment_2017_04_10_11_21_38_simpy_rollout/progress.csv');

C = {'Event-Driven', '10^{-1}', '10^{-0.5}', '10^0',...
    '10^{0.5}', '10^1'};

%% ADR

hc = figure();
set(hc,'PaperUnits','Points');
set(hc,'PaperPosition',[650,550,350,300]);
set(hc,'Units','Points');
set(hc,'Position',[650,550,350,300]);

errorbar([1], [ADR(6)], [StdADR(6)], 'o');
hold on
errorbar(2:6, ADR(1:5), StdADR(1:5), 'o')
grid on
set(gca,'XLim',[1-0.5 6+0.5],'XTickLabel', C)
ylabel 'Average Discounted Return'
title 'Test Policy Performance Over 5000 Rollouts'
rotateXLabels(gca(),45)

% print -dpng -r300 'ADR.png'
%% RolloutTime

hc = figure();
set(hc,'PaperUnits','Points');
set(hc,'PaperPosition',[650,550,350,300]);
set(hc,'Units','Points');
set(hc,'Position',[650,550,350,300]);

errorbar([1], [RolloutTime(6)], [StdRolloutTime(6)], 'o');
hold on
errorbar(2:6, RolloutTime(1:5), StdRolloutTime(1:5), 'o')
grid on
set(gca,'XLim',[1-0.5 6+0.5],'XTickLabel', C)
ylabel 'Average Simulation Time [s]'
title 'Rollout Simulation Time Over 5000 Rollouts'
rotateXLabels(gca(),45)

% print -dpng -r300 'SimTime.png'

%% RolloutTimeNoED

hc = figure();
set(hc,'PaperUnits','Points');
set(hc,'PaperPosition',[650,550,350,300]);
set(hc,'Units','Points');
set(hc,'Position',[650,550,350,300]);

% errorbar([1], [RolloutTime(6)], [StdRolloutTime(6)], 'o');
% hold on
errorbar(1:5, RolloutTime(1:5), StdRolloutTime(1:5), 'o')
grid on
set(gca,'XLim',[1-0.5 5+0.5],'XTickLabel', C(2:6))
ylabel 'Average Simulation Time [s]'
title 'Rollout Simulation Time Over 5000 Rollouts'
xlabel 'Simualtion Timestep'
% rotateXLabels(gca(),45)

print -dpng -r300 'SimTimeNoED.png'

%% Learned Policy Comparison

hc = figure();
set(hc,'PaperUnits','Points');
set(hc,'PaperPosition',[650,550,350,300]);
set(hc,'Units','Points');
set(hc,'Position',[650,550,350,300]);

plot(S.Iteration, S.AverageDiscountedReturn);
hold on
plot([0,max(S.Iteration)],[ADR(6),ADR(6)]);
clr = [0.8500    0.3250    0.0980];
stdadr_500 = StdADR(6) * sqrt(5000) / sqrt(mean(S.NumTrajs(50:end)));
plot([0,max(S.Iteration)],[ADR(6),ADR(6)]+2*stdadr_500,'--', 'Color',clr);
plot([0,max(S.Iteration)],[ADR(6),ADR(6)]-2*stdadr_500,'--', 'Color',clr);
grid on
xlabel 'Iteration'
ylabel 'Average Discounted Return'
title 'Learning Curve of GRU Policy'
legend 'GRU Policy' 'Test Policy Avg' 'Test Policy \pm 2\sigma' 'Location' 'Best'

% print -dpng -r300 'LearningCurve.png'