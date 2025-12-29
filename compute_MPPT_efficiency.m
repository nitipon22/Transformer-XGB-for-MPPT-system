function [eta, eta_median, eta_RMS, perc95, results] = compute_MPPT_efficiency_v2(P_algo_vec, P_MPP)
% Enhanced MPPT performance metrics with algorithm-sensitive measurements
%
% INPUT:
%   P_algo_vec : numeric vector or timeseries (MPPT power output)
%   P_MPP      : numeric vector (PV module max power)
%
% OUTPUT:
%   eta, eta_median, eta_RMS, perc95 : classic MPPT efficiency metrics
%   results : struct with EXTENDED metrics:
%       - Classic: tracking_speed, settling_time, steady_state_osc
%       - NEW: transient_loss, overshoot, response_quality, stability_index

%% 1. Convert timeseries to numeric
if isa(P_algo_vec,'timeseries'); P_algo_vec = P_algo_vec.Data; end
P_algo_vec = P_algo_vec(:);
P_MPP = P_MPP(:);

%% 2. Resample / downsample to match lengths
decimate_factor = floor(length(P_algo_vec)/length(P_MPP));
P_algo_ds = P_algo_vec(1:decimate_factor:end);
P_algo_resampled = P_algo_ds(1:length(P_MPP));

%% 3. Steady-state last 10%
N = length(P_MPP);
idx_ss = round(0.9*N):N;
P_algo_ss = min(P_algo_resampled(idx_ss), P_MPP(idx_ss));
P_MPP_ss  = P_MPP(idx_ss);
valid_idx = P_MPP_ss>0;
P_algo_ss = P_algo_ss(valid_idx);
P_MPP_ss  = P_MPP_ss(valid_idx);

%% 4. Classic efficiency metrics
%% 4. Classic efficiency metrics (modified to cap RMS at 1)
ratio_ss = P_algo_ss ./ P_MPP_ss;

% Clamp values to [0,1] to avoid RMS > 1 due to spikes
ratio_clamped = min(max(ratio_ss, 0), 1);

eta        = mean(ratio_ss);
eta_median = median(ratio_ss);
eta_RMS    = sqrt(mean(ratio_clamped.^2));  % RMS capped at 1
perc95     = sum(ratio_ss >= 0.95)/length(ratio_ss)*100;

fprintf('========== CLASSIC METRICS ==========\n');
fprintf('Steady-state efficiency η = %.6f\n', eta);
fprintf('Median efficiency = %.6f\n', eta_median);
fprintf('RMS efficiency (capped) = %.6f\n', eta_RMS);
fprintf('Percent of time ≥95%% P_MPP = %.6f%%\n', perc95);


%% 5. Detect transient events (step changes)
dP = [0; diff(P_MPP)];
step_idx = find(abs(dP) > 0.05*max(P_MPP));

tracking_speed = zeros(size(step_idx));
settling_time  = zeros(size(step_idx));

% ===== NEW METRICS =====
transient_loss_energy = zeros(size(step_idx));  % Energy lost during transient
overshoot_magnitude   = zeros(size(step_idx));  % Peak overshoot (%)
overshoot_duration    = zeros(size(step_idx));  % Time above target
rise_time            = zeros(size(step_idx));  % 10%-90% rise time
response_smoothness  = zeros(size(step_idx));  % Derivative variance

for k = 1:length(step_idx)
    i0 = step_idx(k);
    new_target = P_MPP(i0);
    
    % Find end of transient window (next step or end)
    if k < length(step_idx)
        i_end = step_idx(k+1) - 1;
    else
        i_end = length(P_algo_resampled);
    end
    
    window = P_algo_resampled(i0:i_end);
    target_window = P_MPP(i0:i_end);
    
    % --- CLASSIC: Tracking speed (first reach 95%) ---
    reach_idx = find(window >= 0.95*new_target, 1, 'first');
    if isempty(reach_idx)
        tracking_speed(k) = length(window);
    else
        tracking_speed(k) = reach_idx;
    end
    
    % --- CLASSIC: Settling time (stay within 1%) ---
    settled = abs(window - new_target) <= 0.01*new_target;
    settle_idx = find(settled, 1, 'first');
    if isempty(settle_idx)
        settling_time(k) = length(window);
    else
        % Check if stays settled for at least 5 samples
        remain_settled = all(settled(settle_idx:min(settle_idx+5, end)));
        if remain_settled
            settling_time(k) = settle_idx;
        else
            settling_time(k) = length(window);
        end
    end
    
    % ===== NEW 1: TRANSIENT ENERGY LOSS =====
    % Energy that could have been captured if at MPP immediately
    % FIX: Handle edge cases and ensure positive loss
    ideal_energy = sum(max(target_window, 0));  % ensure non-negative
    actual_energy = sum(max(window, 0));        % ensure non-negative
    
    % Loss should be non-negative (can't gain energy from nothing)
    energy_loss = max(ideal_energy - actual_energy, 0);
    transient_loss_energy(k) = energy_loss;
    
    % ===== NEW 2: OVERSHOOT =====
    % Peak power above target (if any)
    % FIX: Handle edge cases properly
    if new_target > 1 % ignore very low power targets
        overshoot_samples = (window - new_target) / new_target * 100; % percentage
        
        % Clip extreme outliers (likely measurement errors)
        overshoot_samples(overshoot_samples > 500) = 0; % ignore >500% as invalid
        overshoot_samples(overshoot_samples < 0.5) = 0;  % ignore <0.5% as noise
        
        if any(overshoot_samples > 0)
            overshoot_magnitude(k) = max(overshoot_samples); % already in %
            overshoot_duration(k) = sum(overshoot_samples > 1.0); % >1% threshold
        else
            overshoot_magnitude(k) = 0;
            overshoot_duration(k) = 0;
        end
    else
        overshoot_magnitude(k) = 0;
        overshoot_duration(k) = 0;
    end
    
    % ===== NEW 3: RISE TIME (10% to 90%) =====
    % More sensitive than tracking speed
    p10 = 0.1 * new_target;
    p90 = 0.9 * new_target;
    idx_10 = find(window >= p10, 1, 'first');
    idx_90 = find(window >= p90, 1, 'first');
    
    if ~isempty(idx_10) && ~isempty(idx_90) && idx_90 > idx_10
        rise_time(k) = idx_90 - idx_10;
    else
        rise_time(k) = NaN;
    end
    
    % ===== NEW 4: RESPONSE SMOOTHNESS =====
    % Variance of power derivative (smoothness indicator)
    if length(window) > 2
        dP_window = diff(window);
        response_smoothness(k) = std(dP_window);
    else
        response_smoothness(k) = NaN;
    end
end

fprintf('\n========== TRANSIENT METRICS ==========\n');
fprintf('Tracking speed (steps): mean=%.1f, median=%.1f\n', ...
        mean(tracking_speed), median(tracking_speed));
fprintf('Settling time (steps): mean=%.1f, median=%.1f\n', ...
        mean(settling_time), median(settling_time));

% Filter out NaN values for statistics
valid_rise = rise_time(~isnan(rise_time));
valid_smooth = response_smoothness(~isnan(response_smoothness));

fprintf('\n========== NEW: TRANSIENT QUALITY ==========\n');
fprintf('Energy loss per transient (W·s): mean=%.2f, total=%.2f\n', ...
        mean(transient_loss_energy), sum(transient_loss_energy));
fprintf('Overshoot magnitude (%%): mean=%.3f, max=%.3f\n', ...
        mean(overshoot_magnitude), max(overshoot_magnitude));
fprintf('Overshoot duration (steps): mean=%.1f\n', mean(overshoot_duration));
if ~isempty(valid_rise)
    fprintf('Rise time 10-90%% (steps): mean=%.1f, median=%.1f\n', ...
            mean(valid_rise), median(valid_rise));
end
if ~isempty(valid_smooth)
    fprintf('Response smoothness (std dP): mean=%.4f\n', mean(valid_smooth));
end

%% 6. Steady-state oscillation (enhanced)
steady_state_osc = std(P_algo_ss ./ P_MPP_ss);

% ===== NEW: OSCILLATION FREQUENCY ANALYSIS =====
% Detect periodic oscillations using autocorrelation
if length(P_algo_ss) > 50
    [acf, lags] = xcorr(P_algo_ss - mean(P_algo_ss), 25, 'normalized');
    % Find first peak after lag 0
    [pks, locs] = findpeaks(acf(lags > 0));
    if ~isempty(pks) && max(pks) > 0.3  % significant periodicity
        osc_period = lags(lags > 0);
        osc_period = osc_period(locs(1));
    else
        osc_period = NaN;
    end
else
    osc_period = NaN;
end

fprintf('\n========== STEADY-STATE QUALITY ==========\n');
fprintf('Power oscillation (std ratio): %.6f\n', steady_state_osc);
if ~isnan(osc_period)
    fprintf('Dominant oscillation period: %d steps\n', osc_period);
else
    fprintf('No significant periodic oscillation detected\n');
end

% ===== NEW: POWER UTILIZATION FACTOR =====
% How well does algorithm utilize available power over time
% FIX: Clip to 100% max (can't extract more than MPP!)
power_utilization = min(sum(P_algo_resampled) / max(sum(P_MPP), 1) * 100, 100);
fprintf('Overall power utilization: %.3f%%\n', power_utilization);

%% 7. Energy yield & 95% CI
dt = 1;
energy_yield = sum(P_algo_resampled)*dt;
n_block = 96;
block_len = floor(length(P_algo_resampled)/n_block);
y_block = zeros(n_block,1);
for b = 1:n_block
    idx_b = (b-1)*block_len+1 : b*block_len;
    y_block(b) = sum(P_algo_resampled(idx_b))*dt;
end
energy_yield_CI = mean(y_block) + tinv([0.025 0.975], n_block-1)*std(y_block)/sqrt(n_block);
fprintf('\n========== ENERGY YIELD ==========\n');
fprintf('Total energy yield: %.2f W·s\n', energy_yield);
fprintf('Day-level energy yield CI [95%%]: %.2f - %.2f\n', ...
        energy_yield_CI(1), energy_yield_CI(2));

%% 8. ===== NEW: STABILITY INDEX =====
% Composite metric: lower is more stable
% Combines oscillation, overshoot, and smoothness
stability_index = steady_state_osc * 100 + ...  % oscillation penalty
                  mean(overshoot_magnitude) + ...  % overshoot penalty
                  mean(valid_smooth)/10;           % roughness penalty

fprintf('\n========== COMPOSITE METRICS ==========\n');
fprintf('Stability Index (lower=better): %.4f\n', stability_index);

% ===== NEW: RESPONSE QUALITY SCORE =====
% Composite metric: higher is better (0-100 scale)
speed_score = 100 * exp(-mean(tracking_speed)/100);  % faster = higher
stability_score = 100 * exp(-steady_state_osc*10);   % stabler = higher
energy_score = power_utilization;                     % already 0-100

% FIX: Ensure final score is 0-100
response_quality = 0.4*speed_score + 0.3*stability_score + 0.3*energy_score;
response_quality = min(response_quality, 100); % cap at 100

fprintf('Response Quality Score (0-100): %.2f\n', response_quality);
fprintf('  ├─ Speed component: %.2f\n', speed_score);
fprintf('  ├─ Stability component: %.2f\n', stability_score);
fprintf('  └─ Energy component: %.2f\n', energy_score);

%% 9. Pack extended metrics
results.tracking_speed   = tracking_speed;
results.settling_time    = settling_time;
results.steady_state_osc = steady_state_osc;
results.energy_yield     = energy_yield;
results.energy_yield_CI  = energy_yield_CI;

% NEW fields
results.transient_loss_energy = transient_loss_energy;
results.total_transient_loss  = sum(transient_loss_energy);
results.overshoot_magnitude   = overshoot_magnitude;
results.overshoot_duration    = overshoot_duration;
results.rise_time            = rise_time;
results.response_smoothness  = response_smoothness;
results.oscillation_period   = osc_period;
results.power_utilization    = power_utilization;
results.stability_index      = stability_index;
results.response_quality     = response_quality;

fprintf('\n========================================\n');

end