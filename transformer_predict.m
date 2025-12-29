function y = transformer_predict(seq)
    % scale
    x_min = [0.0, 18.14041547];
    x_max = [1.22165185, 65.54571367];
    y_min = [0.0, 18.14041547];
    y_max = [1.22165185, 65.54571367];

    % ===== scale input =====
    eps_val = 1e-8;
    seq_scaled = (seq - x_min) ./ (x_max - x_min + eps_val);
    seq_scaled = single(seq_scaled);

    % ===== เพิ่ม Python module path =====
    folder = 'C:\Users\Asus\OneDrive - KASETSART UNIVERSITY\Documents\MATLAB\Examples\MPPT';
    if count(py.sys.path, folder) == 0
        insert(py.sys.path,int32(0),folder);
    end

    % ===== import Python module และ reload =====
    mod = py.importlib.import_module('transformer_predict_py');
    py.importlib.reload(mod);

    % ===== Python inference =====
    y_scaled_py = mod.predict(seq_scaled);  % เรียก function จาก module
    y_scaled = double(y_scaled_py);         % แปลงเป็น double สำหรับ MATLAB

    % ===== inverse scale =====
    y = y_scaled .* (y_max - y_min) + y_min;
end
