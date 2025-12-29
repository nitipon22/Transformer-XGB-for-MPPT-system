% ===== 1. โDownload Simulink Dataset =====
[irr_array, temp_array, time_irr, time_temp] = archivedatafromsumulink();
num_samples = length(irr_array);

% ===== 2. seq_length =====
seq_length = min(10, num_samples);  % ปรับให้ไม่เกินจำนวน sample

disp('Example before forecast:');
disp([irr_array(1:seq_length), temp_array(1:seq_length)]);

% ===== 3. Prepare storage =====
num_forecast = max(num_samples-seq_length, 1); % อย่างน้อย 1
irr_forecast2 = zeros(num_forecast,1);
temp_forecast2 = zeros(num_forecast,1);

% ===== 4. forecast loop =====
for t = 1:num_forecast
    seq = [irr_array(t:t+seq_length-1), temp_array(t:t+seq_length-1)];
    
    % ===== transformer_predict =====
    try
        seq_next = transformer_predict(seq);  % forecast n+1
    catch ME
        warning('transformer_predict failed: %s', ME.message);
        seq_next = seq(end,:);  % ถ้า fail ใช้ค่า last timestep
    end
    
    irr_forecast2(t) = seq_next(1);
    temp_forecast2(t) = seq_next(2);
end

% ===== 5.Time of forecast =====
time_forecast2 = time_irr(seq_length+1 : seq_length+num_forecast);

% ===== 6. Save forecast data =====
save_path = 'C:\Users\Asus\OneDrive - KASETSART UNIVERSITY\Documents\MATLAB\Examples\MPPT\forecast2.mat';
save(save_path,'irr_forecast2','temp_forecast2','time_forecast2');
disp(['Forecast saved to: ', save_path]);

% ===== 7. Check file =====
if exist(save_path,'file') == 2
    disp('forecast2.mat exists!');
else
    disp('forecast2.mat NOT found!');
end

% ===== 8. Download and check =====
data = load(save_path);
disp('Variables in forecast.mat:');
disp(fieldnames(data));

% ===== 9. Plot preview =====
figure;
plot(time_forecast2, irr_forecast2,'-o');
xlabel('Time'); ylabel('Irradiation Forecast');
title('Forecasted Irradiation');
grid on;
