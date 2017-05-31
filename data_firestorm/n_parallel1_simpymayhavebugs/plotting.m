clear
close all
clc

filename = './experiment_2017_04_07_18_44_35_fixed_10e1/progress.csv';
Fixed2 = dataloader(filename);

filename = './experiment_2017_04_07_20_42_19_fixed_10e0p5/progress.csv';
Fixed1 = dataloader(filename);

filename = './experiment_2017_04_07_21_44_55_fixed_10e0/progress.csv';
Fixed0 = dataloader(filename);

filename = './experiment_2017_04_07_22_13_22_fixed_10e-0p5/progress.csv';
Fixedm1 = dataloader(filename);

filename = './experiment_2017_04_07_23_27_23_fixed_10e-1/progress.csv';
Fixedm2 = dataloader(filename);

filename = './experiment_2017_04_07_19_09_27_simpy/progress.csv';
Simpy = dataloader(filename);

filename = './experiment_2017_04_07_20_00_40_simpy_rollout/progress.csv';
SimpyRollout = dataloader(filename);

%%
plot(SimpyRollout.Iteration, SimpyRollout.AverageDiscountedReturn);
hold on
plot(Simpy.Iteration, Simpy.AverageDiscountedReturn);
plot(Fixedm2.Iteration, Fixedm2.AverageDiscountedReturn);
plot(Fixedm1.Iteration, Fixedm1.AverageDiscountedReturn);
plot(Fixed0.Iteration, Fixed0.AverageDiscountedReturn);
plot(Fixed1.Iteration, Fixed1.AverageDiscountedReturn);
plot(Fixed2.Iteration, Fixed2.AverageDiscountedReturn);
legend 'SimpyRollout' 'Simpy' 'Fixed 10^{-1}' 'Fixed 10e-.5' 'Fixed 10e0' 'Fixed 10e0.5' 'Fixed 10e1' 

%%
figure();
plot(SimpyRollout.Iteration, SimpyRollout.ItrTime./SimpyRollout.NumTrajs);
hold on
plot(Simpy.Iteration, Simpy.ItrTime./Simpy.NumTrajs);
plot(Fixedm2.Iteration, Fixedm2.ItrTime./Fixedm2.NumTrajs);
plot(Fixedm1.Iteration, Fixedm1.ItrTime./Fixedm1.NumTrajs);
plot(Fixed0.Iteration, Fixed0.ItrTime./Fixed0.NumTrajs);
plot(Fixed1.Iteration, Fixed1.ItrTime./Fixed1.NumTrajs);
plot(Fixed2.Iteration, Fixed2.ItrTime./Fixed2.NumTrajs);

legend 'SimpyRollout' 'Simpy' 'Fixed 10e-1' 'Fixed 10e-.5' 'Fixed 10e0' 'Fixed 10e0.5' 'Fixed 10e1' 

set(gca, 'yscale', 'log')

%%

C = categorical({ 'SimpyRollout' 'Simpy' 'Fixed 10e-1' 'Fixed 10e-.5' 'Fixed 10e0' 'Fixed 10e0.5' 'Fixed 10e1' });

ymean = @(s) mean(s.ItrTime(end-50:end)./s.NumTrajs(end-50:end));
ymin = @(s) min(s.ItrTime(end-50:end)./s.NumTrajs(end-50:end));
ymax = @(s) max(s.ItrTime(end-50:end)./s.NumTrajs(end-50:end));

y = [ymean(SimpyRollout), ymean(Simpy), ymean(Fixedm2), ymean(Fixedm1), ymean(Fixed0), ymean(Fixed1), ymean(Fixed2)];
ymin = [ymin(SimpyRollout), ymin(Simpy), ymin(Fixedm2), ymin(Fixedm1), ymin(Fixed0), ymin(Fixed1), ymin(Fixed2)];
ymax = [ymax(SimpyRollout), ymax(Simpy), ymax(Fixedm2), ymax(Fixedm1), ymax(Fixed0), ymax(Fixed1), ymax(Fixed2)];

errorbar(1:7, y, y-ymin, ymax-y, 'o');

% set(gca, 'yscale', 'log');

%%

ymean = @(s) mean(s.AverageDiscountedReturn(end-50:end));
ymin = @(s) min(s.AverageDiscountedReturn(end-50:end));
ymax = @(s) max(s.AverageDiscountedReturn(end-50:end));
y = [ymean(SimpyRollout), ymean(Simpy), ymean(Fixedm2), ymean(Fixedm1), ymean(Fixed0), ymean(Fixed1), ymean(Fixed2)];
ymin = [ymin(SimpyRollout), ymin(Simpy), ymin(Fixedm2), ymin(Fixedm1), ymin(Fixed0), ymin(Fixed1), ymin(Fixed2)];
ymax = [ymax(SimpyRollout), ymax(Simpy), ymax(Fixedm2), ymax(Fixedm1), ymax(Fixed0), ymax(Fixed1), ymax(Fixed2)];

errorbar(1:7, y, y-ymin, ymax-y, 'o');