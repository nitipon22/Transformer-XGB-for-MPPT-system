function P_MPP = pv_mpp_from_GT(G, T)
% G, T : numeric vectors (W/m^2, °C)

    Vmpp_stc = 54.7;
    Impp_stc = 5.58;
    G_stc    = 1000;
    T_stc    = 25;

    alpha_V = -0.177/100;
    alpha_I =  0.003516/100;

    Vmpp = Vmpp_stc .* (1 + alpha_V .* (T - T_stc));
    Impp = Impp_stc .* (G ./ G_stc) .* (1 + alpha_I .* (T - T_stc));

    P_MPP = Vmpp .* Impp;
end
