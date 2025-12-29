% Load CSV
T = readtable('synthetic_weather_635samples_daylight.csv');

irr = T.IRRADIATION_SYN;     % หรือ IRRADIATION_SYN_kWm2
N = length(irr);

% Create synthetic timestamp
time = (0:N-1)' * 0.01;      % เพิ่มทีละ 0.01

% Create timeseries
irr_ts = timeseries(irr, time);
irr_ts.Name = 'irr_signal';

% Put into Dataset
irr_ds = Simulink.SimulationData.Dataset;
irr_ds = irr_ds.addElement(irr_ts);

save('irr_dataset2.mat','irr_ds');
temp = T.MODULE_TEMPERATURE_SYN;
N = length(temp);

time = (0:N-1)' * 0.01;

temp_ts = timeseries(temp, time);
temp_ts.Name = 'temp_signal';

temp_ds = Simulink.SimulationData.Dataset;
temp_ds = temp_ds.addElement(temp_ts);

save('temp_dataset2.mat','temp_ds');
