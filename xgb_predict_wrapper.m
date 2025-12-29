function y = xgb_predict_wrapper(seq)

    mu_X = [0.232024, 31.242466];
    sigma_X = [0.301290, 12.295958];

    mu_y = 0.51234;
    sigma_y = 0.18452;

    seq_scaled = (seq - mu_X) ./ sigma_X;

    N = size(seq_scaled,1);
    y_scaled = zeros(N,1);

    for i = 1:N
        y_scaled(i) = xgb_model(seq_scaled(i,:));
    end

    y = y_scaled .* sigma_y + mu_y;
end
