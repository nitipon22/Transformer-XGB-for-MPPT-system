/*
 * Extra Trees Model for DC Power Prediction
 * Generated using m2cgen
 *
 * Model Parameters:
 *   n_estimators: 100
 *   max_depth: 3
 *
 * Input Features (must be scaled):
 *   - IRRADIATION (mean: 0.232024, std: 0.301290)
 *   - MODULE_TEMPERATURE (mean: 31.242466, std: 12.295958)
 *
 * Scaling Formula: scaled_value = (original_value - mean) / std
 *
 * Performance Metrics:
 *   MAE:  159.8737
 *   RMSE: 339.4975
 *   R²:   0.9934
 */

double score(double * input) {
    double var0;
    if (input[0] < 0.3092404) {
        if (input[0] < -0.4036506) {
            if (input[0] < -0.6426407) {
                var0 = -971.55133;
            } else {
                var0 = -652.10394;
            }
        } else {
            if (input[0] < -0.059336174) {
                var0 = -283.232;
            } else {
                var0 = 212.5348;
            }
        }
    } else {
        if (input[0] < 1.3692976) {
            if (input[0] < 0.83622116) {
                var0 = 817.41266;
            } else {
                var0 = 1438.6625;
            }
        } else {
            if (input[0] < 1.9643731) {
                var0 = 2038.6354;
            } else {
                var0 = 2714.5964;
            }
        }
    }
    double var1;
    if (input[0] < 0.41982147) {
        if (input[0] < -0.34753308) {
            if (input[0] < -0.5955983) {
                var1 = -677.29095;
            } else {
                var1 = -400.44806;
            }
        } else {
            if (input[0] < 0.07555466) {
                var1 = -108.56453;
            } else {
                var1 = 242.22064;
            }
        }
    } else {
        if (input[0] < 1.5001159) {
            if (input[0] < 0.9549586) {
                var1 = 679.69794;
            } else {
                var1 = 1094.9202;
            }
        } else {
            if (input[0] < 2.1525705) {
                var1 = 1537.2963;
            } else {
                var1 = 2001.1453;
            }
        }
    }
    double var2;
    if (input[0] < 0.50222623) {
        if (input[0] < -0.23435031) {
            if (input[0] < -0.6952748) {
                var2 = -481.52792;
            } else {
                var2 = -296.1006;
            }
        } else {
            if (input[0] < 0.17048691) {
                var2 = 7.750913;
            } else {
                var2 = 206.188;
            }
        }
    } else {
        if (input[0] < 1.6180478) {
            if (input[0] < 1.1666988) {
                var2 = 554.72205;
            } else {
                var2 = 866.07404;
            }
        } else {
            if (input[0] < 2.3585827) {
                var2 = 1164.7466;
            } else {
                var2 = 1488.8081;
            }
        }
    }
    double var3;
    if (input[0] < 0.60947174) {
        if (input[0] < -0.16914791) {
            if (input[0] < -0.53352755) {
                var3 = -332.28745;
            } else {
                var3 = -150.11543;
            }
        } else {
            if (input[0] < 0.2144261) {
                var3 = 34.96648;
            } else {
                var3 = 154.97089;
            }
        }
    } else {
        if (input[0] < 1.76388) {
            if (input[0] < 1.0918447) {
                var3 = 410.18872;
            } else {
                var3 = 606.0084;
            }
        } else {
            if (input[0] < 2.4122026) {
                var3 = 871.48663;
            } else {
                var3 = 1082.5677;
            }
        }
    }
    double var4;
    if (input[0] < 0.5795773) {
        if (input[0] < -0.27622887) {
            if (input[0] < -0.72587824) {
                var4 = -238.61072;
            } else {
                var4 = -146.64221;
            }
        } else {
            if (input[0] < -0.014805929) {
                var4 = -45.314796;
            } else {
                var4 = 87.87202;
            }
        }
    } else {
        if (input[0] < 1.7830465) {
            if (input[0] < 1.2247891) {
                var4 = 294.0959;
            } else {
                var4 = 452.18222;
            }
        } else {
            if (input[0] < 2.4122026) {
                var4 = 616.0131;
            } else {
                var4 = 762.9525;
            }
        }
    }
    double var5;
    if (input[0] < 0.6510046) {
        if (input[0] < -0.4722229) {
            if (input[0] < -0.7503656) {
                var5 = -168.16687;
            } else {
                var5 = -129.86343;
            }
        } else {
            if (input[0] < -0.014805929) {
                var5 = -43.128445;
            } else {
                var5 = 62.77603;
            }
        }
    } else {
        if (input[0] < 1.8932048) {
            if (input[0] < 1.269781) {
                var5 = 223.93822;
            } else {
                var5 = 323.3267;
            }
        } else {
            if (input[0] < 2.516053) {
                var5 = 457.79404;
            } else {
                var5 = 572.37506;
            }
        }
    }
    double var6;
    if (input[0] < 0.69700843) {
        if (input[0] < -0.16914791) {
            if (input[0] < -0.5059475) {
                var6 = -114.766975;
            } else {
                var6 = -47.54285;
            }
        } else {
            if (input[0] < 0.22389412) {
                var6 = 8.960936;
            } else {
                var6 = 58.844337;
            }
        }
    } else {
        if (input[0] < 1.8932048) {
            if (input[1] < 2.298426) {
                var6 = 199.00842;
            } else {
                var6 = -93.586624;
            }
        } else {
            if (input[1] < 1.8910224) {
                var6 = 382.71964;
            } else {
                var6 = 305.99268;
            }
        }
    }
    double var7;
    if (input[0] < 0.7465692) {
        if (input[0] < -0.16914791) {
            if (input[0] < -0.7430259) {
                var7 = -83.21306;
            } else {
                var7 = -47.291447;
            }
        } else {
            if (input[0] < 0.12286835) {
                var7 = 0.52278036;
            } else {
                var7 = 38.81131;
            }
        }
    } else {
        if (input[0] < 1.5896733) {
            if (input[1] < 1.9836276) {
                var7 = 131.21596;
            } else {
                var7 = -79.06775;
            }
        } else {
            if (input[0] < 2.1525705) {
                var7 = 192.70027;
            } else {
                var7 = 266.53165;
            }
        }
    }
    double var8;
    if (input[0] < 0.7465692) {
        if (input[0] < -0.48537967) {
            if (input[0] < -0.75992775) {
                var8 = -58.495415;
            } else {
                var8 = -45.579815;
            }
        } else {
            if (input[0] < 0.12286835) {
                var8 = -9.936868;
            } else {
                var8 = 27.203573;
            }
        }
    } else {
        if (input[0] < 1.3444357) {
            if (input[1] < 1.9446343) {
                var8 = 72.47058;
            } else {
                var8 = 410.1219;
            }
        } else {
            if (input[1] < 1.026505) {
                var8 = 267.9241;
            } else {
                var8 = 140.74623;
            }
        }
    }
    double var9;
    if (input[0] < 0.77664536) {
        if (input[0] < -0.14205119) {
            if (input[0] < -0.49476722) {
                var9 = -39.738163;
            } else {
                var9 = -15.174232;
            }
        } else {
            if (input[1] < 1.2423424) {
                var9 = 19.896748;
            } else {
                var9 = -65.84257;
            }
        }
    } else {
        if (input[0] < 2.0581074) {
            if (input[1] < 2.2472205) {
                var9 = 73.848625;
            } else {
                var9 = -128.2043;
            }
        } else {
            if (input[1] < 1.8547814) {
                var9 = 212.71791;
            } else {
                var9 = 114.675354;
            }
        }
    }
    double var10;
    if (input[0] < 1.0093727) {
        if (input[1] < 1.9143627) {
            if (input[0] < -0.12363764) {
                var10 = -25.925406;
            } else {
                var10 = 9.671036;
            }
        } else {
            var10 = 895.07495;
        }
    } else {
        if (input[1] < 0.6373045) {
            if (input[0] < 1.1293308) {
                var10 = 187.81216;
            } else {
                var10 = 358.6543;
            }
        } else {
            if (input[0] < 2.4122026) {
                var10 = 55.819492;
            } else {
                var10 = 131.14833;
            }
        }
    }
    double var11;
    if (input[0] < 0.22389412) {
        if (input[1] < 0.98096067) {
            if (input[0] < -0.7557853) {
                var11 = -21.258003;
            } else {
                var11 = -7.820851;
            }
        } else {
            if (input[0] < 0.15154985) {
                var11 = 167.17702;
            } else {
                var11 = -7.9886723;
            }
        }
    } else {
        if (input[0] < 1.5551221) {
            if (input[1] < 1.1597792) {
                var11 = 35.92434;
            } else {
                var11 = -9.018124;
            }
        } else {
            if (input[1] < 1.6690246) {
                var11 = 99.18785;
            } else {
                var11 = 41.526257;
            }
        }
    }
    double var12;
    if (input[0] < 1.0383079) {
        if (input[1] < 1.9143627) {
            if (input[0] < -0.49476722) {
                var12 = -14.2438135;
            } else {
                var12 = 0.7870086;
            }
        } else {
            var12 = 762.16644;
        }
    } else {
        if (input[1] < 1.050449) {
            if (input[1] < 0.98096067) {
                var12 = 65.59209;
            } else {
                var12 = 158.18459;
            }
        } else {
            if (input[0] < 1.5551221) {
                var12 = -1.5256207;
            } else {
                var12 = 43.2547;
            }
        }
    }
    double var13;
    if (input[0] < 1.0383079) {
        if (input[1] < 1.9143627) {
            if (input[1] < 1.4386871) {
                var13 = -7.0086856;
            } else {
                var13 = 53.41435;
            }
        } else {
            var13 = 647.84155;
        }
    } else {
        if (input[1] < 1.322288) {
            if (input[1] < 0.4408364) {
                var13 = 258.6996;
            } else {
                var13 = 50.68172;
            }
        } else {
            if (input[0] < 1.4661658) {
                var13 = -50.301754;
            } else {
                var13 = 28.801533;
            }
        }
    }
    double var14;
    if (input[0] < 0.22389412) {
        if (input[1] < 0.98096067) {
            if (input[1] < 0.21954711) {
                var14 = -5.772857;
            } else {
                var14 = -25.014036;
            }
        } else {
            if (input[0] < 0.15154985) {
                var14 = 130.96207;
            } else {
                var14 = -5.857105;
            }
        }
    } else {
        if (input[0] < 2.516053) {
            if (input[1] < 1.7308077) {
                var14 = 17.218359;
            } else {
                var14 = -14.259417;
            }
        } else {
            if (input[1] < 2.2118182) {
                var14 = 144.09094;
            } else {
                var14 = 21.116693;
            }
        }
    }
    double var15;
    if (input[0] < 2.101897) {
        if (input[1] < 2.298426) {
            if (input[0] < 1.194018) {
                var15 = -3.2600994;
            } else {
                var15 = 14.924539;
            }
        } else {
            if (input[0] < 1.7509601) {
                var15 = -369.47925;
            } else {
                var15 = 54.301174;
            }
        }
    } else {
        if (input[1] < 1.8547814) {
            if (input[0] < 2.1764362) {
                var15 = 197.6617;
            } else {
                var15 = 52.79644;
            }
        } else {
            if (input[1] < 1.9836276) {
                var15 = -53.69183;
            } else {
                var15 = 32.32158;
            }
        }
    }
    double var16;
    if (input[0] < 2.3210783) {
        if (input[1] < 1.7308077) {
            if (input[1] < 1.5290937) {
                var16 = -1.5117841;
            } else {
                var16 = 63.489037;
            }
        } else {
            if (input[0] < 0.87619853) {
                var16 = 213.09344;
            } else {
                var16 = -31.39345;
            }
        }
    } else {
        if (input[1] < 1.9143627) {
            if (input[1] < 1.8110256) {
                var16 = 7.7162995;
            } else {
                var16 = 212.79927;
            }
        } else {
            if (input[0] < 2.4122026) {
                var16 = -18.96097;
            } else {
                var16 = 37.14729;
            }
        }
    }
    double var17;
    if (input[0] < -0.49476722) {
        if (input[0] < -0.76361305) {
            var17 = -5.380774;
        } else {
            if (input[0] < -0.6426407) {
                var17 = 6.762713;
            } else {
                var17 = -10.7896385;
            }
        }
    } else {
        if (input[0] < 2.1525705) {
            if (input[1] < 2.298426) {
                var17 = 5.174144;
            } else {
                var17 = -88.80028;
            }
        } else {
            if (input[1] < 1.6690246) {
                var17 = 95.29222;
            } else {
                var17 = 15.377105;
            }
        }
    }
    double var18;
    if (input[0] < 1.194018) {
        if (input[1] < 1.9143627) {
            if (input[1] < 1.1597792) {
                var18 = -0.25580215;
            } else {
                var18 = -40.418007;
            }
        } else {
            if (input[0] < 0.87619853) {
                var18 = 520.55304;
            } else {
                var18 = 96.606155;
            }
        }
    } else {
        if (input[1] < 0.6373045) {
            var18 = 220.99864;
        } else {
            if (input[1] < 1.7308077) {
                var18 = 20.637638;
            } else {
                var18 = -9.363586;
            }
        }
    }
    double var19;
    if (input[0] < 2.0581074) {
        if (input[0] < 1.9367478) {
            if (input[0] < 1.8548867) {
                var19 = -0.85646933;
            } else {
                var19 = 66.35125;
            }
        } else {
            if (input[1] < 1.8910224) {
                var19 = -17.511673;
            } else {
                var19 = -132.82379;
            }
        }
    } else {
        if (input[1] < 1.2294728) {
            var19 = -108.090096;
        } else {
            if (input[1] < 1.8547814) {
                var19 = 52.69897;
            } else {
                var19 = 10.513758;
            }
        }
    }
    double var20;
    if (input[0] < -0.7408543) {
        if (input[0] < -0.7557853) {
            var20 = -3.4385307;
        } else {
            if (input[1] < -0.75587535) {
                var20 = -5.028318;
            } else {
                var20 = 0.21950291;
            }
        }
    } else {
        if (input[1] < 0.7387563) {
            if (input[0] < 0.9390912) {
                var20 = 5.5767484;
            } else {
                var20 = 90.19362;
            }
        } else {
            if (input[1] < 0.98096067) {
                var20 = -24.50547;
            } else {
                var20 = 4.240661;
            }
        }
    }
    double var21;
    if (input[0] < 2.516053) {
        if (input[0] < 2.4665205) {
            if (input[1] < 2.4669218) {
                var21 = -0.24596006;
            } else {
                var21 = 65.16349;
            }
        } else {
            if (input[1] < 2.1376953) {
                var21 = -163.68362;
            } else {
                var21 = -15.733888;
            }
        }
    } else {
        if (input[1] < 2.2118182) {
            if (input[0] < 2.561257) {
                var21 = 137.73404;
            } else {
                var21 = 41.253933;
            }
        } else {
            if (input[1] < 2.298426) {
                var21 = -163.47972;
            } else {
                var21 = 28.364521;
            }
        }
    }
    double var22;
    if (input[1] < 2.0423226) {
        if (input[1] < 1.7308077) {
            if (input[1] < 1.5290937) {
                var22 = -0.435818;
            } else {
                var22 = 33.702454;
            }
        } else {
            if (input[0] < 1.194018) {
                var22 = 99.20548;
            } else {
                var22 = -37.925262;
            }
        }
    } else {
        if (input[0] < 1.6180478) {
            if (input[0] < 1.4525768) {
                var22 = 52.535526;
            } else {
                var22 = -175.16542;
            }
        } else {
            if (input[1] < 2.0785747) {
                var22 = 147.12625;
            } else {
                var22 = 9.985013;
            }
        }
    }
    double var23;
    if (input[1] < 1.3579777) {
        if (input[1] < 1.3417464) {
            if (input[1] < 1.322288) {
                var23 = 0.9441734;
            } else {
                var23 = -111.436874;
            }
        } else {
            if (input[0] < 0.7566076) {
                var23 = -1.9961914;
            } else {
                var23 = 164.80652;
            }
        }
    } else {
        if (input[1] < 1.5290937) {
            if (input[0] < 1.1424932) {
                var23 = 3.0606987;
            } else {
                var23 = -51.770386;
            }
        } else {
            if (input[1] < 1.5866061) {
                var23 = 111.71547;
            } else {
                var23 = -5.881516;
            }
        }
    }
    double var24;
    if (input[0] < 2.2813783) {
        if (input[0] < 2.2603693) {
            if (input[1] < 2.298426) {
                var24 = 0.29510286;
            } else {
                var24 = -78.03307;
            }
        } else {
            if (input[1] < 2.0423226) {
                var24 = -292.73636;
            } else {
                var24 = 55.618946;
            }
        }
    } else {
        if (input[1] < 1.4976379) {
            var24 = 140.2062;
        } else {
            if (input[1] < 1.8110256) {
                var24 = -40.63734;
            } else {
                var24 = 18.578081;
            }
        }
    }
    double var25;
    if (input[0] < -0.4722229) {
        if (input[0] < -0.6426407) {
            if (input[0] < -0.67342806) {
                var25 = -2.4501176;
            } else {
                var25 = 23.977392;
            }
        } else {
            if (input[0] < -0.6187279) {
                var25 = -29.002296;
            } else {
                var25 = -3.8052766;
            }
        }
    } else {
        if (input[1] < 0.45447338) {
            if (input[0] < 0.92342216) {
                var25 = 8.050907;
            } else {
                var25 = 207.83192;
            }
        } else {
            if (input[0] < 0.69700843) {
                var25 = -11.072799;
            } else {
                var25 = 2.364787;
            }
        }
    }
    double var26;
    if (input[1] < -0.050636075) {
        if (input[0] < -0.03185337) {
            if (input[0] < -0.09668803) {
                var26 = -1.1823635;
            } else {
                var26 = 29.966003;
            }
        } else {
            if (input[1] < -0.3586931) {
                var26 = -91.92473;
            } else {
                var26 = -20.037918;
            }
        }
    } else {
        if (input[1] < 0.019112622) {
            if (input[0] < -0.0411059) {
                var26 = -14.279527;
            } else {
                var26 = 116.84524;
            }
        } else {
            if (input[1] < 1.3579777) {
                var26 = 5.03479;
            } else {
                var26 = -4.5906253;
            }
        }
    }
    double var27;
    if (input[0] < 2.004078) {
        if (input[0] < 1.9367478) {
            if (input[1] < 1.8910224) {
                var27 = -0.7124135;
            } else {
                var27 = 45.320652;
            }
        } else {
            if (input[1] < 1.8910224) {
                var27 = -25.450417;
            } else {
                var27 = -204.49303;
            }
        }
    } else {
        if (input[1] < 1.2294728) {
            if (input[0] < 2.101897) {
                var27 = -25.698633;
            } else {
                var27 = -98.22022;
            }
        } else {
            if (input[1] < 1.4510678) {
                var27 = 86.47764;
            } else {
                var27 = 9.097627;
            }
        }
    }
    double var28;
    if (input[0] < 1.8548867) {
        if (input[1] < 1.5866061) {
            if (input[1] < 1.5290937) {
                var28 = -0.4190725;
            } else {
                var28 = 130.50777;
            }
        } else {
            if (input[0] < 1.0625091) {
                var28 = 48.2289;
            } else {
                var28 = -34.20902;
            }
        }
    } else {
        if (input[0] < 1.927341) {
            if (input[1] < 1.8910224) {
                var28 = 38.500435;
            } else {
                var28 = 139.831;
            }
        } else {
            if (input[0] < 2.004078) {
                var28 = -52.40054;
            } else {
                var28 = 6.725982;
            }
        }
    }
    double var29;
    if (input[1] < 2.4003484) {
        if (input[1] < 2.3471203) {
            if (input[1] < 1.3579777) {
                var29 = 0.9123809;
            } else {
                var29 = -6.0356817;
            }
        } else {
            if (input[0] < 1.5001159) {
                var29 = -49.6158;
            } else {
                var29 = 68.86954;
            }
        }
    } else {
        if (input[0] < 1.7509601) {
            var29 = -179.77405;
        } else {
            if (input[0] < 2.3585827) {
                var29 = 43.893826;
            } else {
                var29 = -55.77846;
            }
        }
    }
    double var30;
    if (input[0] < 2.516053) {
        if (input[0] < 2.4665205) {
            if (input[0] < 2.4122026) {
                var30 = -0.24552211;
            } else {
                var30 = 32.586903;
            }
        } else {
            if (input[1] < 2.4003484) {
                var30 = -31.97101;
            } else {
                var30 = -191.16608;
            }
        }
    } else {
        if (input[1] < 2.2118182) {
            if (input[0] < 2.561257) {
                var30 = 109.596214;
            } else {
                var30 = 30.418068;
            }
        } else {
            if (input[1] < 2.2472205) {
                var30 = -201.18765;
            } else {
                var30 = 5.3062177;
            }
        }
    }
    double var31;
    if (input[1] < 1.3579777) {
        if (input[1] < 1.3417464) {
            if (input[1] < 1.322288) {
                var31 = 0.6473094;
            } else {
                var31 = -82.57973;
            }
        } else {
            if (input[0] < 0.7566076) {
                var31 = -1.6992189;
            } else {
                var31 = 115.31455;
            }
        }
    } else {
        if (input[0] < 1.4661658) {
            if (input[0] < 1.0918447) {
                var31 = 13.270036;
            } else {
                var31 = -52.224186;
            }
        } else {
            if (input[0] < 1.5317043) {
                var31 = 55.35561;
            } else {
                var31 = -0.6718203;
            }
        }
    }
    double var32;
    if (input[0] < 1.0093727) {
        if (input[1] < 1.9143627) {
            if (input[0] < 0.97267365) {
                var32 = -0.6998625;
            } else {
                var32 = -83.98396;
            }
        } else {
            var32 = 413.20987;
        }
    } else {
        if (input[1] < 0.47941405) {
            if (input[1] < 0.30494073) {
                var32 = 12.073243;
            } else {
                var32 = 162.99141;
            }
        } else {
            if (input[1] < 1.4386871) {
                var32 = 15.492208;
            } else {
                var32 = -5.456686;
            }
        }
    }
    double var33;
    if (input[1] < 2.0423226) {
        if (input[0] < 2.3830833) {
            if (input[0] < 2.2603693) {
                var33 = -0.2083468;
            } else {
                var33 = -116.78492;
            }
        } else {
            if (input[1] < 1.8110256) {
                var33 = -26.273767;
            } else {
                var33 = 98.40286;
            }
        }
    } else {
        if (input[0] < 1.6180478) {
            if (input[1] < 2.298426) {
                var33 = 11.701046;
            } else {
                var33 = -173.37291;
            }
        } else {
            if (input[0] < 1.8932048) {
                var33 = 103.380394;
            } else {
                var33 = 4.2949934;
            }
        }
    }
    double var34;
    if (input[1] < 2.4003484) {
        if (input[1] < 2.2118182) {
            if (input[0] < 2.6601155) {
                var34 = -0.2976923;
            } else {
                var34 = 53.321415;
            }
        } else {
            if (input[0] < 2.516053) {
                var34 = 36.55935;
            } else {
                var34 = -23.108276;
            }
        }
    } else {
        if (input[0] < 1.7509601) {
            var34 = -167.35892;
        } else {
            if (input[0] < 1.805211) {
                var34 = 105.294876;
            } else {
                var34 = -19.545433;
            }
        }
    }
    double var35;
    if (input[1] < 2.4003484) {
        if (input[1] < 2.3471203) {
            if (input[1] < 2.298426) {
                var35 = 0.1108632;
            } else {
                var35 = -22.774582;
            }
        } else {
            if (input[0] < 2.3210783) {
                var35 = 65.33133;
            } else {
                var35 = 13.241342;
            }
        }
    } else {
        if (input[0] < 1.7509601) {
            var35 = -142.255;
        } else {
            if (input[0] < 2.3585827) {
                var35 = 27.217505;
            } else {
                var35 = -33.03428;
            }
        }
    }
    double var36;
    if (input[1] < 0.98096067) {
        if (input[1] < 0.87054884) {
            if (input[0] < 0.7379302) {
                var36 = -0.84356874;
            } else {
                var36 = 28.910103;
            }
        } else {
            if (input[0] < 0.25309572) {
                var36 = 79.4001;
            } else {
                var36 = -38.16805;
            }
        }
    } else {
        if (input[1] < 1.026505) {
            if (input[0] < 0.7379302) {
                var36 = -18.725195;
            } else {
                var36 = 125.14111;
            }
        } else {
            if (input[0] < 0.34305537) {
                var36 = 96.2954;
            } else {
                var36 = -3.2555714;
            }
        }
    }
    double var37;
    if (input[0] < 1.8548867) {
        if (input[0] < 1.8240484) {
            if (input[1] < 2.298426) {
                var37 = 0.05906444;
            } else {
                var37 = -107.06807;
            }
        } else {
            if (input[1] < 1.7504286) {
                var37 = -189.36334;
            } else {
                var37 = 45.08965;
            }
        }
    } else {
        if (input[0] < 1.927341) {
            if (input[0] < 1.9155196) {
                var37 = 16.884083;
            } else {
                var37 = 122.2577;
            }
        } else {
            if (input[0] < 2.004078) {
                var37 = -34.327724;
            } else {
                var37 = 5.502265;
            }
        }
    }
    double var38;
    if (input[0] < 1.3692976) {
        if (input[1] < 1.9143627) {
            if (input[1] < 1.322288) {
                var38 = 0.19698702;
            } else {
                var38 = -33.793427;
            }
        } else {
            if (input[0] < 0.87619853) {
                var38 = 351.76715;
            } else {
                var38 = 77.89901;
            }
        }
    } else {
        if (input[0] < 1.4250009) {
            if (input[0] < 1.4028634) {
                var38 = 46.94979;
            } else {
                var38 = 231.33363;
            }
        } else {
            if (input[0] < 1.4661658) {
                var38 = -58.216473;
            } else {
                var38 = 1.9810548;
            }
        }
    }
    double var39;
    if (input[1] < 1.7308077) {
        if (input[1] < 1.6925088) {
            if (input[1] < 1.6690246) {
                var39 = 0.41141173;
            } else {
                var39 = -73.842064;
            }
        } else {
            if (input[0] < 1.4525768) {
                var39 = 168.10034;
            } else {
                var39 = 43.35589;
            }
        }
    } else {
        if (input[1] < 1.7504286) {
            if (input[0] < 1.8845633) {
                var39 = -146.39214;
            } else {
                var39 = 26.967676;
            }
        } else {
            if (input[0] < 0.87619853) {
                var39 = 127.864365;
            } else {
                var39 = -3.2760563;
            }
        }
    }
    double var40;
    if (input[0] < 1.5896733) {
        if (input[1] < 1.8110256) {
            if (input[0] < 1.3883452) {
                var40 = -0.9082563;
            } else {
                var40 = 34.808964;
            }
        } else {
            if (input[0] < 1.5317043) {
                var40 = 2.1357598;
            } else {
                var40 = -403.4644;
            }
        }
    } else {
        if (input[0] < 1.602237) {
            if (input[1] < 1.4510678) {
                var40 = -1.4931641;
            } else {
                var40 = 194.98586;
            }
        } else {
            if (input[1] < 0.6373045) {
                var40 = 137.92589;
            } else {
                var40 = 2.074264;
            }
        }
    }
    double var41;
    if (input[0] < 1.7830465) {
        if (input[1] < 2.298426) {
            if (input[0] < 1.7509601) {
                var41 = 0.26867905;
            } else {
                var41 = -73.03147;
            }
        } else {
            if (input[0] < 1.5001159) {
                var41 = -38.874367;
            } else {
                var41 = -197.62126;
            }
        }
    } else {
        if (input[0] < 1.8240484) {
            if (input[1] < 1.4100294) {
                var41 = -82.24556;
            } else {
                var41 = 100.642525;
            }
        } else {
            if (input[0] < 1.8548867) {
                var41 = -67.727936;
            } else {
                var41 = 4.056995;
            }
        }
    }
    double var42;
    if (input[0] < 1.5896733) {
        if (input[1] < 1.7708623) {
            if (input[1] < 1.6925088) {
                var42 = -0.33296275;
            } else {
                var42 = 90.205894;
            }
        } else {
            if (input[0] < 1.5317043) {
                var42 = -2.8333008;
            } else {
                var42 = -312.7452;
            }
        }
    } else {
        if (input[1] < 0.6373045) {
            var42 = 117.19673;
        } else {
            if (input[1] < 0.89338964) {
                var42 = -171.55069;
            } else {
                var42 = 4.5521836;
            }
        }
    }
    double var43;
    if (input[0] < 2.4391267) {
        if (input[0] < 2.4254553) {
            if (input[1] < 2.298426) {
                var43 = 0.1693788;
            } else {
                var43 = -20.623247;
            }
        } else {
            if (input[1] < 1.9836276) {
                var43 = 150.9986;
            } else {
                var43 = 27.823536;
            }
        }
    } else {
        if (input[0] < 2.516053) {
            if (input[1] < 2.1376953) {
                var43 = -96.17466;
            } else {
                var43 = -19.23003;
            }
        } else {
            if (input[1] < 2.4669218) {
                var43 = 0.25890198;
            } else {
                var43 = 49.588013;
            }
        }
    }
    double var44;
    if (input[1] < 2.2118182) {
        if (input[1] < 2.1376953) {
            if (input[1] < 2.0423226) {
                var44 = -0.21514335;
            } else {
                var44 = 27.348774;
            }
        } else {
            if (input[0] < 1.602237) {
                var44 = 72.154106;
            } else {
                var44 = -37.283497;
            }
        }
    } else {
        if (input[0] < 2.3830833) {
            if (input[1] < 2.298426) {
                var44 = 66.56757;
            } else {
                var44 = -2.692881;
            }
        } else {
            if (input[1] < 2.298426) {
                var44 = -53.96977;
            } else {
                var44 = 1.2805882;
            }
        }
    }
    double var45;
    if (input[1] < 0.5321654) {
        if (input[0] < 0.23453309) {
            if (input[1] < 0.4408364) {
                var45 = -0.2819945;
            } else {
                var45 = -58.363125;
            }
        } else {
            if (input[0] < 0.29995042) {
                var45 = 69.67121;
            } else {
                var45 = 16.081913;
            }
        }
    } else {
        if (input[1] < 0.92111415) {
            if (input[0] < 1.6608067) {
                var45 = -13.639228;
            } else {
                var45 = 120.66094;
            }
        } else {
            if (input[0] < 0.23453309) {
                var45 = 74.12892;
            } else {
                var45 = 0.18972348;
            }
        }
    }
    double var46;
    if (input[1] < 1.5866061) {
        if (input[1] < 1.5290937) {
            if (input[1] < 1.4386871) {
                var46 = 0.36344802;
            } else {
                var46 = -22.232847;
            }
        } else {
            if (input[0] < 1.6901685) {
                var46 = 136.54372;
            } else {
                var46 = -68.22161;
            }
        }
    } else {
        if (input[0] < 1.5707138) {
            if (input[0] < 1.4250009) {
                var46 = 4.5344276;
            } else {
                var46 = -82.623856;
            }
        } else {
            if (input[0] < 1.602237) {
                var46 = 163.1526;
            } else {
                var46 = -0.63992;
            }
        }
    }
    double var47;
    if (input[1] < 2.4003484) {
        if (input[1] < 2.3471203) {
            if (input[0] < 2.561257) {
                var47 = 0.089823976;
            } else {
                var47 = -14.009807;
            }
        } else {
            if (input[0] < 2.3210783) {
                var47 = 69.18396;
            } else {
                var47 = 10.148893;
            }
        }
    } else {
        if (input[0] < 2.70753) {
            if (input[0] < 2.3585827) {
                var47 = 10.067432;
            } else {
                var47 = -51.44173;
            }
        } else {
            if (input[1] < 2.4669218) {
                var47 = -6.9246097;
            } else {
                var47 = 113.56978;
            }
        }
    }
    double var48;
    if (input[0] < 2.2813783) {
        if (input[0] < 2.2603693) {
            if (input[1] < 2.298426) {
                var48 = 0.33655292;
            } else {
                var48 = -56.03509;
            }
        } else {
            if (input[1] < 2.0423226) {
                var48 = -203.34544;
            } else {
                var48 = 26.209497;
            }
        }
    } else {
        if (input[1] < 1.4976379) {
            var48 = 137.54459;
        } else {
            if (input[1] < 2.2472205) {
                var48 = -11.026055;
            } else {
                var48 = 23.511444;
            }
        }
    }
    double var49;
    if (input[0] < 1.6901685) {
        if (input[1] < 1.9836276) {
            if (input[1] < 1.8910224) {
                var49 = -0.5199108;
            } else {
                var49 = 68.20166;
            }
        } else {
            if (input[0] < 1.3413136) {
                var49 = 86.43219;
            } else {
                var49 = -102.520905;
            }
        }
    } else {
        if (input[0] < 1.7509601) {
            if (input[1] < 1.1180828) {
                var49 = -117.46758;
            } else {
                var49 = 60.448475;
            }
        } else {
            if (input[0] < 1.76388) {
                var49 = -102.736275;
            } else {
                var49 = 4.528277;
            }
        }
    }
    double var50;
    if (input[1] < 2.4669218) {
        if (input[1] < 2.4003484) {
            if (input[1] < 2.1831675) {
                var50 = -0.15946332;
            } else {
                var50 = 11.842246;
            }
        } else {
            if (input[0] < 2.1525705) {
                var50 = -104.026726;
            } else {
                var50 = -7.3758698;
            }
        }
    } else {
        if (input[0] < 2.6601155) {
            if (input[0] < 2.4665205) {
                var50 = 37.044876;
            } else {
                var50 = -97.06751;
            }
        } else {
            var50 = 84.893074;
        }
    }
    double var51;
    if (input[1] < 1.1597792) {
        if (input[1] < 0.98096067) {
            if (input[1] < 0.7921925) {
                var51 = 0.5663412;
            } else {
                var51 = -17.342737;
            }
        } else {
            if (input[0] < 0.69700843) {
                var51 = -39.782223;
            } else {
                var51 = 31.161383;
            }
        }
    } else {
        if (input[1] < 1.1793569) {
            if (input[0] < 1.5063825) {
                var51 = -135.9347;
            } else {
                var51 = 76.70176;
            }
        } else {
            if (input[1] < 1.202559) {
                var51 = 51.357098;
            } else {
                var51 = -1.2261847;
            }
        }
    }
    double var52;
    if (input[1] < 1.026505) {
        if (input[1] < 0.98096067) {
            if (input[0] < 1.6608067) {
                var52 = -0.361569;
            } else {
                var52 = 99.47022;
            }
        } else {
            if (input[0] < 0.7379302) {
                var52 = -9.9406;
            } else {
                var52 = 77.145584;
            }
        }
    } else {
        if (input[1] < 1.0697067) {
            if (input[0] < 1.3883452) {
                var52 = -17.977764;
            } else {
                var52 = -105.95899;
            }
        } else {
            if (input[0] < 0.34305537) {
                var52 = 69.68485;
            } else {
                var52 = -1.4617715;
            }
        }
    }
    double var53;
    if (input[0] < 1.5896733) {
        if (input[0] < 1.4250009) {
            if (input[0] < 1.4028634) {
                var53 = -0.22454526;
            } else {
                var53 = 152.10895;
            }
        } else {
            if (input[1] < 1.8110256) {
                var53 = -1.7362794;
            } else {
                var53 = -112.825356;
            }
        }
    } else {
        if (input[1] < 0.8499698) {
            var53 = 93.44199;
        } else {
            if (input[1] < 0.89338964) {
                var53 = -235.01764;
            } else {
                var53 = 3.5244734;
            }
        }
    }
    double var54;
    if (input[1] < 0.7387563) {
        if (input[0] < 0.7227904) {
            if (input[0] < 0.7120771) {
                var54 = 0.10915211;
            } else {
                var54 = -125.95909;
            }
        } else {
            if (input[1] < -0.031561047) {
                var54 = 159.99632;
            } else {
                var54 = 20.682014;
            }
        }
    } else {
        if (input[0] < 0.25309572) {
            if (input[1] < 1.1597792) {
                var54 = 43.482956;
            } else {
                var54 = -32.109303;
            }
        } else {
            if (input[0] < 0.5627772) {
                var54 = -44.590714;
            } else {
                var54 = -0.44742563;
            }
        }
    }
    double var55;
    if (input[1] < -0.0061724996) {
        if (input[0] < 0.34305537) {
            if (input[0] < -0.03185337) {
                var55 = -0.104882255;
            } else {
                var55 = -29.143711;
            }
        } else {
            var55 = 121.32872;
        }
    } else {
        if (input[1] < 0.019112622) {
            if (input[0] < -0.09668803) {
                var55 = 3.1499865;
            } else {
                var55 = 217.25294;
            }
        } else {
            if (input[0] < 0.10617138) {
                var55 = -9.636255;
            } else {
                var55 = 1.1108465;
            }
        }
    }
    double var56;
    if (input[0] < 0.7120771) {
        if (input[1] < 1.4100294) {
            if (input[1] < 1.2423424) {
                var56 = 0.8512711;
            } else {
                var56 = -128.30896;
            }
        } else {
            if (input[1] < 1.5178847) {
                var56 = 326.171;
            } else {
                var56 = -73.18526;
            }
        }
    } else {
        if (input[0] < 0.7227904) {
            if (input[1] < 0.87054884) {
                var56 = -163.50381;
            } else {
                var56 = 181.36502;
            }
        } else {
            if (input[1] < 0.47941405) {
                var56 = 57.260258;
            } else {
                var56 = -1.5989974;
            }
        }
    }
    double var57;
    if (input[0] < -0.5059475) {
        if (input[0] < -0.5955983) {
            if (input[0] < -0.6028376) {
                var57 = -0.826639;
            } else {
                var57 = 22.29519;
            }
        } else {
            if (input[0] < -0.568849) {
                var57 = -30.164042;
            } else {
                var57 = -2.51801;
            }
        }
    } else {
        if (input[0] < -0.4036506) {
            if (input[0] < -0.44780684) {
                var57 = 0.8574767;
            } else {
                var57 = 36.676952;
            }
        } else {
            if (input[1] < -0.5861996) {
                var57 = -40.485588;
            } else {
                var57 = 1.0072745;
            }
        }
    }
    double var58;
    if (input[0] < 0.7120771) {
        if (input[0] < 0.69700843) {
            if (input[1] < 1.2423424) {
                var58 = 0.31480953;
            } else {
                var58 = -49.30218;
            }
        } else {
            if (input[1] < 1.3905287) {
                var58 = 64.46991;
            } else {
                var58 = 383.92398;
            }
        }
    } else {
        if (input[0] < 0.7227904) {
            if (input[1] < 0.87054884) {
                var58 = -121.71898;
            } else {
                var58 = 154.00914;
            }
        } else {
            if (input[1] < 0.87054884) {
                var58 = 16.193756;
            } else {
                var58 = -3.1857965;
            }
        }
    }
    double var59;
    if (input[1] < 1.2294728) {
        if (input[1] < 1.1597792) {
            if (input[0] < 1.5551221) {
                var59 = 0.33663902;
            } else {
                var59 = -43.94773;
            }
        } else {
            if (input[0] < 0.46703342) {
                var59 = 124.32041;
            } else {
                var59 = -38.38625;
            }
        }
    } else {
        if (input[0] < 0.4041916) {
            if (input[1] < 1.3739798) {
                var59 = -209.87566;
            } else {
                var59 = 35.045776;
            }
        } else {
            if (input[1] < 1.322288) {
                var59 = 29.679169;
            } else {
                var59 = 0.050388724;
            }
        }
    }
    double var60;
    if (input[0] < 1.4661658) {
        if (input[0] < 1.4250009) {
            if (input[0] < 1.4028634) {
                var60 = -0.3163552;
            } else {
                var60 = 113.303764;
            }
        } else {
            if (input[1] < 1.5866061) {
                var60 = -15.483029;
            } else {
                var60 = -104.30719;
            }
        }
    } else {
        if (input[1] < 1.1180828) {
            if (input[1] < 0.7921925) {
                var60 = 81.38017;
            } else {
                var60 = -69.49932;
            }
        } else {
            if (input[1] < 1.1310265) {
                var60 = 135.12666;
            } else {
                var60 = 3.7994068;
            }
        }
    }
    double var61;
    if (input[0] < 1.3692976) {
        if (input[1] < 1.9143627) {
            if (input[1] < 1.4698071) {
                var61 = -0.00981305;
            } else {
                var61 = -33.09302;
            }
        } else {
            if (input[0] < 0.87619853) {
                var61 = 269.85852;
            } else {
                var61 = 40.48529;
            }
        }
    } else {
        if (input[0] < 1.4250009) {
            if (input[1] < 1.7308077) {
                var61 = 66.19056;
            } else {
                var61 = -57.781277;
            }
        } else {
            if (input[0] < 1.4417175) {
                var61 = -56.41984;
            } else {
                var61 = 2.1069658;
            }
        }
    }
    double var62;
    if (input[1] < -0.0061724996) {
        if (input[0] < 0.34305537) {
            if (input[1] < -0.12081121) {
                var62 = -0.17046869;
            } else {
                var62 = -20.53421;
            }
        } else {
            if (input[0] < 0.35332263) {
                var62 = 27.46941;
            } else {
                var62 = 106.62672;
            }
        }
    } else {
        if (input[1] < 0.019112622) {
            if (input[0] < -0.09668803) {
                var62 = 1.7554917;
            } else {
                var62 = 162.39377;
            }
        } else {
            if (input[1] < 0.1469382) {
                var62 = 12.500379;
            } else {
                var62 = -0.29512614;
            }
        }
    }
    double var63;
    if (input[0] < 2.4391267) {
        if (input[0] < 2.4122026) {
            if (input[1] < 2.4669218) {
                var63 = -0.09479701;
            } else {
                var63 = 34.942642;
            }
        } else {
            if (input[1] < 2.0423226) {
                var63 = 123.83548;
            } else {
                var63 = 4.1312857;
            }
        }
    } else {
        if (input[0] < 2.516053) {
            if (input[1] < 2.1376953) {
                var63 = -75.30381;
            } else {
                var63 = -9.236251;
            }
        } else {
            if (input[1] < 2.2118182) {
                var63 = 25.118639;
            } else {
                var63 = -13.043894;
            }
        }
    }
    double var64;
    if (input[1] < 2.4003484) {
        if (input[1] < 2.3471203) {
            if (input[0] < 2.561257) {
                var64 = 0.15625003;
            } else {
                var64 = -16.860538;
            }
        } else {
            if (input[0] < 2.3210783) {
                var64 = 69.94084;
            } else {
                var64 = -2.6113281;
            }
        }
    } else {
        if (input[0] < 2.70753) {
            if (input[0] < 2.3585827) {
                var64 = 9.66798;
            } else {
                var64 = -41.912834;
            }
        } else {
            if (input[1] < 2.4669218) {
                var64 = -7.536768;
            } else {
                var64 = 81.04307;
            }
        }
    }
    double var65;
    if (input[0] < 2.516053) {
        if (input[0] < 2.4665205) {
            if (input[0] < 2.4122026) {
                var65 = -0.07102422;
            } else {
                var65 = 17.654297;
            }
        } else {
            if (input[1] < 2.4003484) {
                var65 = -7.5791855;
            } else {
                var65 = -128.07921;
            }
        }
    } else {
        if (input[0] < 2.561257) {
            if (input[1] < 1.8249304) {
                var65 = 115.20714;
            } else {
                var65 = -18.599474;
            }
        } else {
            if (input[1] < 1.7504286) {
                var65 = -76.012405;
            } else {
                var65 = 9.687504;
            }
        }
    }
    double var66;
    if (input[1] < 2.4669218) {
        if (input[1] < 2.4003484) {
            if (input[1] < 2.3471203) {
                var66 = -0.002079908;
            } else {
                var66 = 18.879627;
            }
        } else {
            if (input[0] < 2.1525705) {
                var66 = -83.52052;
            } else {
                var66 = -0.9821192;
            }
        }
    } else {
        if (input[0] < 2.561257) {
            if (input[0] < 2.4665205) {
                var66 = 19.931515;
            } else {
                var66 = -86.876076;
            }
        } else {
            var66 = 59.518654;
        }
    }
    double var67;
    if (input[0] < 2.1764362) {
        if (input[0] < 2.1525705) {
            if (input[1] < 2.298426) {
                var67 = 0.11029102;
            } else {
                var67 = -27.421259;
            }
        } else {
            if (input[1] < 2.1376953) {
                var67 = 94.197495;
            } else {
                var67 = -18.982471;
            }
        }
    } else {
        if (input[0] < 2.1896114) {
            if (input[1] < 2.2472205) {
                var67 = -21.934326;
            } else {
                var67 = -141.17345;
            }
        } else {
            if (input[1] < 1.8910224) {
                var67 = 24.694536;
            } else {
                var67 = -7.879488;
            }
        }
    }
    double var68;
    if (input[1] < 2.0423226) {
        if (input[1] < 1.9836276) {
            if (input[0] < 2.2603693) {
                var68 = 0.12678972;
            } else {
                var68 = -24.164064;
            }
        } else {
            if (input[0] < 2.101897) {
                var68 = -109.30116;
            } else {
                var68 = 36.04964;
            }
        }
    } else {
        if (input[0] < 1.8932048) {
            if (input[1] < 2.0785747) {
                var68 = -52.31802;
            } else {
                var68 = 63.315746;
            }
        } else {
            if (input[0] < 1.9643731) {
                var68 = -161.42389;
            } else {
                var68 = 6.1526165;
            }
        }
    }
    double var69;
    if (input[0] < -0.7464813) {
        if (input[1] < -0.6146139) {
            if (input[0] < -0.76361305) {
                var69 = -0.9170509;
            } else {
                var69 = -2.7006383;
            }
        } else {
            if (input[0] < -0.76361305) {
                var69 = -0.890757;
            } else {
                var69 = 0.75675863;
            }
        }
    } else {
        if (input[0] < 0.29995042) {
            if (input[0] < 0.25309572) {
                var69 = 0.96670216;
            } else {
                var69 = 65.43226;
            }
        } else {
            if (input[0] < 0.39144477) {
                var69 = -30.455614;
            } else {
                var69 = 0.7017096;
            }
        }
    }
    double var70;
    if (input[0] < 2.1764362) {
        if (input[0] < 2.1525705) {
            if (input[1] < 2.298426) {
                var70 = 0.14322953;
            } else {
                var70 = -31.4914;
            }
        } else {
            if (input[1] < 2.1376953) {
                var70 = 69.50391;
            } else {
                var70 = -17.163282;
            }
        }
    } else {
        if (input[0] < 2.1896114) {
            if (input[1] < 2.2472205) {
                var70 = -16.959229;
            } else {
                var70 = -121.02554;
            }
        } else {
            if (input[1] < 1.9143627) {
                var70 = 20.753887;
            } else {
                var70 = -7.605084;
            }
        }
    }
    double var71;
    if (input[1] < 2.0423226) {
        if (input[0] < 2.3830833) {
            if (input[0] < 2.2603693) {
                var71 = -0.12059664;
            } else {
                var71 = -77.32869;
            }
        } else {
            if (input[0] < 2.4391267) {
                var71 = 90.31455;
            } else {
                var71 = -6.11294;
            }
        }
    } else {
        if (input[1] < 2.0785747) {
            if (input[0] < 1.6180478) {
                var71 = -130.50381;
            } else {
                var71 = 84.88858;
            }
        } else {
            if (input[0] < 1.8932048) {
                var71 = 49.665737;
            } else {
                var71 = -6.4624763;
            }
        }
    }
    double var72;
    if (input[1] < 2.1831675) {
        if (input[1] < 2.1376953) {
            if (input[1] < 2.0423226) {
                var72 = -0.19317345;
            } else {
                var72 = 20.139845;
            }
        } else {
            if (input[0] < 1.927341) {
                var72 = -136.76514;
            } else {
                var72 = -1.3522106;
            }
        }
    } else {
        if (input[0] < 1.602237) {
            if (input[1] < 2.298426) {
                var72 = 95.17688;
            } else {
                var72 = -26.490187;
            }
        } else {
            if (input[0] < 1.7509601) {
                var72 = -134.86534;
            } else {
                var72 = 5.818385;
            }
        }
    }
    double var73;
    if (input[0] < 0.7120771) {
        if (input[0] < 0.69700843) {
            if (input[1] < 1.2423424) {
                var73 = 0.24729292;
            } else {
                var73 = -25.849913;
            }
        } else {
            if (input[1] < 1.3905287) {
                var73 = 46.52528;
            } else {
                var73 = 326.3075;
            }
        }
    } else {
        if (input[0] < 0.89766264) {
            if (input[1] < 1.8547814) {
                var73 = -21.30072;
            } else {
                var73 = 215.96007;
            }
        } else {
            if (input[0] < 0.97267365) {
                var73 = 31.274948;
            } else {
                var73 = -1.4152306;
            }
        }
    }
    double var74;
    if (input[1] < 1.6690246) {
        if (input[0] < 2.1205378) {
            if (input[0] < 2.043128) {
                var74 = 0.24505629;
            } else {
                var74 = -71.5355;
            }
        } else {
            if (input[1] < 1.2294728) {
                var74 = -74.12862;
            } else {
                var74 = 70.50027;
            }
        }
    } else {
        if (input[1] < 1.6925088) {
            if (input[0] < 2.2390106) {
                var74 = -83.042114;
            } else {
                var74 = 57.903225;
            }
        } else {
            if (input[1] < 1.7308077) {
                var74 = 37.425316;
            } else {
                var74 = -1.8360609;
            }
        }
    }
    double var75;
    if (input[1] < 2.2118182) {
        if (input[0] < 2.2603693) {
            if (input[0] < 2.2390106) {
                var75 = 0.010838417;
            } else {
                var75 = 54.833416;
            }
        } else {
            if (input[0] < 2.3210783) {
                var75 = -91.5248;
            } else {
                var75 = 2.0953615;
            }
        }
    } else {
        if (input[0] < 2.6601155) {
            if (input[0] < 2.6071699) {
                var75 = 7.090787;
            } else {
                var75 = 66.32862;
            }
        } else {
            if (input[1] < 2.4669218) {
                var75 = -64.50528;
            } else {
                var75 = 54.597073;
            }
        }
    }
    double var76;
    if (input[0] < 2.516053) {
        if (input[0] < 2.4391267) {
            if (input[0] < 2.4254553) {
                var76 = -0.05357934;
            } else {
                var76 = 41.415565;
            }
        } else {
            if (input[1] < 2.1376953) {
                var76 = -63.748634;
            } else {
                var76 = 0.90849614;
            }
        }
    } else {
        if (input[0] < 2.561257) {
            if (input[1] < 1.8249304) {
                var76 = 89.26124;
            } else {
                var76 = -5.5626564;
            }
        } else {
            if (input[1] < 2.298426) {
                var76 = -19.917946;
            } else {
                var76 = 25.378231;
            }
        }
    }
    double var77;
    if (input[1] < 0.92111415) {
        if (input[0] < 1.1989113) {
            if (input[0] < 1.0918447) {
                var77 = -0.39976758;
            } else {
                var77 = 64.42874;
            }
        } else {
            if (input[0] < 1.3444357) {
                var77 = -143.61487;
            } else {
                var77 = 24.643946;
            }
        }
    } else {
        if (input[1] < 1.026505) {
            if (input[0] < 1.1666988) {
                var77 = -4.422101;
            } else {
                var77 = 72.4375;
            }
        } else {
            if (input[0] < 0.7227904) {
                var77 = 27.445515;
            } else {
                var77 = -2.7298193;
            }
        }
    }
    double var78;
    if (input[1] < 1.5290937) {
        if (input[1] < 1.4386871) {
            if (input[1] < 1.4100294) {
                var78 = -0.018967437;
            } else {
                var78 = 39.072433;
            }
        } else {
            if (input[0] < 0.7120771) {
                var78 = 209.26137;
            } else {
                var78 = -36.891262;
            }
        }
    } else {
        if (input[1] < 1.5866061) {
            if (input[0] < 1.6901685) {
                var78 = 105.82643;
            } else {
                var78 = -42.40603;
            }
        } else {
            if (input[0] < 0.8178123) {
                var78 = -59.743824;
            } else {
                var78 = 0.075119324;
            }
        }
    }
    double var79;
    if (input[0] < 1.5896733) {
        if (input[1] < 2.0785747) {
            if (input[1] < 1.9446343) {
                var79 = -0.2743106;
            } else {
                var79 = -70.13285;
            }
        } else {
            if (input[0] < 1.2247891) {
                var79 = -42.63391;
            } else {
                var79 = 67.18735;
            }
        }
    } else {
        if (input[1] < 1.6420649) {
            if (input[1] < 1.6169676) {
                var79 = 8.507182;
            } else {
                var79 = 81.5005;
            }
        } else {
            if (input[1] < 1.7504286) {
                var79 = -49.61803;
            } else {
                var79 = 6.293259;
            }
        }
    }
    double var80;
    if (input[0] < 0.83622116) {
        if (input[1] < 1.2423424) {
            if (input[1] < 0.98096067) {
                var80 = 0.17400154;
            } else {
                var80 = 34.180847;
            }
        } else {
            if (input[1] < 1.3058977) {
                var80 = -161.18628;
            } else {
                var80 = 8.385759;
            }
        }
    } else {
        if (input[0] < 0.89766264) {
            if (input[1] < 1.4386871) {
                var80 = -63.131733;
            } else {
                var80 = 208.11623;
            }
        } else {
            if (input[1] < 1.2294728) {
                var80 = -9.127654;
            } else {
                var80 = 4.4597306;
            }
        }
    }
    double var81;
    if (input[1] < 1.8547814) {
        if (input[0] < 1.8548867) {
            if (input[1] < 1.8110256) {
                var81 = 0.16971381;
            } else {
                var81 = -74.81216;
            }
        } else {
            if (input[1] < 1.8110256) {
                var81 = 4.5644784;
            } else {
                var81 = 77.68208;
            }
        }
    } else {
        if (input[0] < 0.87619853) {
            var81 = 165.04718;
        } else {
            if (input[0] < 1.1666988) {
                var81 = -81.318504;
            } else {
                var81 = -5.715631;
            }
        }
    }
    double var82;
    if (input[0] < 1.9367478) {
        if (input[0] < 1.9155196) {
            if (input[1] < 2.0785747) {
                var82 = -0.19084938;
            } else {
                var82 = 31.383942;
            }
        } else {
            if (input[1] < 1.6690246) {
                var82 = -6.6390305;
            } else {
                var82 = 125.640114;
            }
        }
    } else {
        if (input[1] < 1.8910224) {
            if (input[1] < 1.8249304) {
                var82 = 1.1245877;
            } else {
                var82 = 50.12434;
            }
        } else {
            if (input[0] < 2.004078) {
                var82 = -129.85704;
            } else {
                var82 = -5.031274;
            }
        }
    }
    double var83;
    if (input[1] < 1.3579777) {
        if (input[1] < 1.3417464) {
            if (input[1] < 1.322288) {
                var83 = 0.38629588;
            } else {
                var83 = -65.58304;
            }
        } else {
            if (input[0] < 1.2954786) {
                var83 = 114.638725;
            } else {
                var83 = 23.638477;
            }
        }
    } else {
        if (input[0] < 1.4661658) {
            if (input[0] < 1.0918447) {
                var83 = 11.6585455;
            } else {
                var83 = -27.76903;
            }
        } else {
            if (input[0] < 1.5317043) {
                var83 = 44.631283;
            } else {
                var83 = -1.9801427;
            }
        }
    }
    double var84;
    if (input[0] < 1.0093727) {
        if (input[0] < 0.97267365) {
            if (input[0] < 0.9549586) {
                var84 = -0.18260276;
            } else {
                var84 = 109.94669;
            }
        } else {
            if (input[1] < 0.82323366) {
                var84 = 53.15271;
            } else {
                var84 = -95.40811;
            }
        }
    } else {
        if (input[1] < 0.47941405) {
            if (input[1] < 0.30494073) {
                var84 = -19.021729;
            } else {
                var84 = 97.89493;
            }
        } else {
            if (input[1] < 0.5119418) {
                var84 = -101.64532;
            } else {
                var84 = 1.5191879;
            }
        }
    }
    double var85;
    if (input[0] < 1.927341) {
        if (input[0] < 1.9155196) {
            if (input[1] < 2.298426) {
                var85 = 0.12838653;
            } else {
                var85 = -43.640156;
            }
        } else {
            if (input[1] < 1.9836276) {
                var85 = 145.41504;
            } else {
                var85 = -170.21455;
            }
        }
    } else {
        if (input[1] < 1.4698071) {
            if (input[0] < 1.9643731) {
                var85 = -135.87006;
            } else {
                var85 = 7.5435195;
            }
        } else {
            if (input[1] < 1.5660595) {
                var85 = 41.73351;
            } else {
                var85 = -2.594171;
            }
        }
    }
    double var86;
    if (input[1] < 2.0423226) {
        if (input[0] < 1.7509601) {
            if (input[0] < 1.7247764) {
                var86 = 0.1392728;
            } else {
                var86 = 74.90108;
            }
        } else {
            if (input[0] < 1.76388) {
                var86 = -81.501236;
            } else {
                var86 = -4.72348;
            }
        }
    } else {
        if (input[0] < 1.6180478) {
            if (input[0] < 1.4525768) {
                var86 = 36.816727;
            } else {
                var86 = -70.350334;
            }
        } else {
            if (input[1] < 2.0785747) {
                var86 = 61.132195;
            } else {
                var86 = 1.472048;
            }
        }
    }
    double var87;
    if (input[0] < 2.1764362) {
        if (input[0] < 2.1525705) {
            if (input[1] < 2.2472205) {
                var87 = 0.20139341;
            } else {
                var87 = -23.536362;
            }
        } else {
            if (input[1] < 1.9143627) {
                var87 = 15.273145;
            } else {
                var87 = 74.790726;
            }
        }
    } else {
        if (input[0] < 2.1896114) {
            if (input[1] < 2.2472205) {
                var87 = -23.76714;
            } else {
                var87 = -102.70826;
            }
        } else {
            if (input[1] < 2.2472205) {
                var87 = -10.0240555;
            } else {
                var87 = 14.496754;
            }
        }
    }
    double var88;
    if (input[0] < 1.0093727) {
        if (input[1] < 1.9143627) {
            if (input[0] < 0.97267365) {
                var88 = -0.12708333;
            } else {
                var88 = -45.94337;
            }
        } else {
            var88 = 150.37866;
        }
    } else {
        if (input[0] < 1.0625091) {
            if (input[1] < 1.2850909) {
                var88 = -4.125615;
            } else {
                var88 = 81.18581;
            }
        } else {
            if (input[0] < 1.0799716) {
                var88 = -60.475765;
            } else {
                var88 = 1.1118081;
            }
        }
    }
    double var89;
    if (input[1] < 1.3579777) {
        if (input[1] < 1.3417464) {
            if (input[1] < 1.322288) {
                var89 = 0.30543262;
            } else {
                var89 = -50.059956;
            }
        } else {
            if (input[0] < 0.788789) {
                var89 = 109.02668;
            } else {
                var89 = 27.672394;
            }
        }
    } else {
        if (input[0] < 0.867625) {
            if (input[1] < 1.3905287) {
                var89 = -90.89377;
            } else {
                var89 = -3.2756026;
            }
        } else {
            if (input[0] < 0.9390912) {
                var89 = 98.48443;
            } else {
                var89 = -2.0089445;
            }
        }
    }
    double var90;
    if (input[0] < 1.1369029) {
        if (input[1] < 1.9143627) {
            if (input[0] < 1.1058204) {
                var90 = -0.2610263;
            } else {
                var90 = -34.28972;
            }
        } else {
            var90 = 113.04917;
        }
    } else {
        if (input[1] < 0.92111415) {
            if (input[1] < 0.87054884) {
                var90 = 12.931943;
            } else {
                var90 = -102.13789;
            }
        } else {
            if (input[1] < 1.050449) {
                var90 = 47.927387;
            } else {
                var90 = 0.041512795;
            }
        }
    }
    double var91;
    if (input[0] < -0.19160807) {
        if (input[0] < -0.23435031) {
            if (input[1] < -0.072672546) {
                var91 = 0.027295588;
            } else {
                var91 = -9.437049;
            }
        } else {
            if (input[1] < -0.3086844) {
                var91 = -8.419423;
            } else {
                var91 = -43.544895;
            }
        }
    } else {
        if (input[1] < -0.5861996) {
            var91 = -54.86553;
        } else {
            if (input[1] < 0.1469382) {
                var91 = 12.138352;
            } else {
                var91 = -0.14588337;
            }
        }
    }
    double var92;
    if (input[0] < 0.5627772) {
        if (input[0] < 0.5194549) {
            if (input[0] < 0.4463097) {
                var92 = -0.47322944;
            } else {
                var92 = 30.222788;
            }
        } else {
            if (input[1] < 0.7921925) {
                var92 = -20.167164;
            } else {
                var92 = -97.40884;
            }
        }
    } else {
        if (input[0] < 0.56852037) {
            if (input[1] < 0.4408364) {
                var92 = 32.731495;
            } else {
                var92 = 187.71115;
            }
        } else {
            if (input[1] < -0.031561047) {
                var92 = 89.13985;
            } else {
                var92 = 0.14708143;
            }
        }
    }
    double var93;
    if (input[1] < 2.1831675) {
        if (input[1] < 2.1376953) {
            if (input[0] < 2.3830833) {
                var93 = -0.1506758;
            } else {
                var93 = 17.869835;
            }
        } else {
            if (input[0] < 2.043128) {
                var93 = -92.37481;
            } else {
                var93 = 15.395671;
            }
        }
    } else {
        if (input[0] < 2.3585827) {
            if (input[0] < 2.2813783) {
                var93 = 7.864218;
            } else {
                var93 = 97.844406;
            }
        } else {
            if (input[0] < 2.4254553) {
                var93 = -53.649025;
            } else {
                var93 = 0.51940775;
            }
        }
    }
    double var94;
    if (input[0] < 2.1764362) {
        if (input[1] < 2.0423226) {
            if (input[1] < 1.9836276) {
                var94 = 0.07254081;
            } else {
                var94 = -65.4082;
            }
        } else {
            if (input[1] < 2.2472205) {
                var94 = 43.476307;
            } else {
                var94 = -18.839611;
            }
        }
    } else {
        if (input[0] < 2.2390106) {
            if (input[1] < 2.0423226) {
                var94 = 2.4433594;
            } else {
                var94 = -59.738934;
            }
        } else {
            if (input[1] < 1.9143627) {
                var94 = 25.726313;
            } else {
                var94 = -6.6903214;
            }
        }
    }
    double var95;
    if (input[1] < 2.1831675) {
        if (input[1] < 2.1376953) {
            if (input[0] < 2.6601155) {
                var95 = -0.099790454;
            } else {
                var95 = 28.704885;
            }
        } else {
            if (input[0] < 2.043128) {
                var95 = -80.63918;
            } else {
                var95 = 9.8259115;
            }
        }
    } else {
        if (input[0] < 2.3585827) {
            if (input[0] < 2.2813783) {
                var95 = 7.7773685;
            } else {
                var95 = 74.40473;
            }
        } else {
            if (input[0] < 2.4122026) {
                var95 = -43.77207;
            } else {
                var95 = 0.75766474;
            }
        }
    }
    double var96;
    if (input[1] < 1.4386871) {
        if (input[0] < 1.5896733) {
            if (input[0] < 1.4301412) {
                var96 = 0.4782753;
            } else {
                var96 = -41.884563;
            }
        } else {
            if (input[0] < 2.004078) {
                var96 = 25.638384;
            } else {
                var96 = -28.398598;
            }
        }
    } else {
        if (input[0] < 0.7120771) {
            if (input[0] < 0.6330219) {
                var96 = 13.674287;
            } else {
                var96 = 239.39583;
            }
        } else {
            if (input[1] < 1.4698071) {
                var96 = -56.08656;
            } else {
                var96 = -0.2709607;
            }
        }
    }
    double var97;
    if (input[0] < 1.0383079) {
        if (input[1] < 1.0935616) {
            if (input[1] < 1.0697067) {
                var97 = -0.34076673;
            } else {
                var97 = 140.43295;
            }
        } else {
            if (input[0] < 0.89766264) {
                var97 = -32.081505;
            } else {
                var97 = 17.902414;
            }
        }
    } else {
        if (input[0] < 1.0625091) {
            if (input[1] < 1.202559) {
                var97 = 1.9748292;
            } else {
                var97 = 87.130035;
            }
        } else {
            if (input[0] < 1.0799716) {
                var97 = -44.576954;
            } else {
                var97 = 1.4382812;
            }
        }
    }
    double var98;
    if (input[0] < 2.1764362) {
        if (input[1] < 1.8910224) {
            if (input[1] < 1.8547814) {
                var98 = 0.20336647;
            } else {
                var98 = -78.94321;
            }
        } else {
            if (input[0] < 1.9367478) {
                var98 = 28.482233;
            } else {
                var98 = -14.0494175;
            }
        }
    } else {
        if (input[1] < 1.2294728) {
            if (input[0] < 2.2076535) {
                var98 = -70.156204;
            } else {
                var98 = -10.900196;
            }
        } else {
            if (input[1] < 1.6690246) {
                var98 = 34.29738;
            } else {
                var98 = -5.3094473;
            }
        }
    }
    double var99;
    if (input[0] < 1.0466032) {
        if (input[1] < 1.9143627) {
            if (input[0] < 0.83622116) {
                var99 = 0.16124438;
            } else {
                var99 = -12.90641;
            }
        } else {
            var99 = 96.698586;
        }
    } else {
        if (input[0] < 1.0625091) {
            if (input[1] < 1.050449) {
                var99 = 71.44105;
            } else {
                var99 = 29.987953;
            }
        } else {
            if (input[1] < 1.322288) {
                var99 = 7.256185;
            } else {
                var99 = -2.1228728;
            }
        }
    }
    return 0 + (var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12 + var13 + var14 + var15 + var16 + var17 + var18 + var19 + var20 + var21 + var22 + var23 + var24 + var25 + var26 + var27 + var28 + var29 + var30 + var31 + var32 + var33 + var34 + var35 + var36 + var37 + var38 + var39 + var40 + var41 + var42 + var43 + var44 + var45 + var46 + var47 + var48 + var49 + var50 + var51 + var52 + var53 + var54 + var55 + var56 + var57 + var58 + var59 + var60 + var61 + var62 + var63 + var64 + var65 + var66 + var67 + var68 + var69 + var70 + var71 + var72 + var73 + var74 + var75 + var76 + var77 + var78 + var79 + var80 + var81 + var82 + var83 + var84 + var85 + var86 + var87 + var88 + var89 + var90 + var91 + var92 + var93 + var94 + var95 + var96 + var97 + var98 + var99);
}
