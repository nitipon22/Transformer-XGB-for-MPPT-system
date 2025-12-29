function [irr_array, temp_array, time_irr, time_temp] = archivedatafromsumulink()
% Load IRR & TEMP from Simulink Dataset (.mat)
% Returns numeric arrays and corresponding time vectors

% โหลดไฟล์
irrData = load('irr_dataset3.mat');   % object ในไฟล์คือ irr_ds
tempData = load('temp_dataset3.mat'); % object ในไฟล์คือ temp_ds

% ดึง signal ออกมา (Dataset method)
irr_ts = irrData.irr_ds.getElement(1);   % timeseries object
temp_ts = tempData.temp_ds.getElement(1); % timeseries object

% ดึงค่า numeric array
irr_array = irr_ts.Data;    % ใช้ .Data สำหรับ timeseries
temp_array = temp_ts.Data;

% ดึงเวลา
time_irr = irr_ts.Time;
time_temp = temp_ts.Time;

end
