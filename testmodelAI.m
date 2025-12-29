load('forecast3.mat'); % irr_forecast, temp_forecast
P_pred4 = xgb_predict_wrapper([irr_forecast3, temp_forecast3]);
Ts = 0.01;                         % sample time 
t = (0:length(P_pred4)-1)'*Ts;  % 3249×1

P_pred4 = [t, P_pred4];      % 3249×2

save('P_pred4.mat','P_pred4');
