import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
import time
from collections import Counter
import shap

def mase(y_true, y_pred, y_train, m=1, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)
    naive_errors = np.abs(y_train[m:] - y_train[:-m])
    scale = np.mean(naive_errors) + eps
    return np.mean(np.abs(y_true - y_pred)) / scale

def smape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(
        2.0 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + eps)
    ) * 100

def remove_outliers_zscore(X, y, threshold=3):
    """Remove outliers based on z-score threshold"""
    z_scores = np.abs(zscore(X))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], y[mask]

# 1. อ่านข้อมูล
data = pd.read_csv("solar_single_source_15min.csv")
print(data.head())

# 2. Feature Engineering (เตรียมข้อมูลเบื้องต้นเท่านั้น)
X = data[['IRRADIATION', 'MODULE_TEMPERATURE']].values
y = data['DC_POWER'].values

# เช็ค missing values เบื้องต้น
print(f"\nMissing values in X: {pd.DataFrame(X).isnull().sum().sum()}")
print(f"Missing values in y: {pd.Series(y).isnull().sum()}")

# 3. Parameter Grids
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
}

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
}

param_grid_et = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
}

param_grid_svr = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1],
    'gamma': ['scale', 'auto']
}

param_grid_lr = {
    'fit_intercept': [True, False],
    'positive': [False, True]
}

# 4. Nested Cross-Validation with Proper Preprocessing
print("\n" + "="*70)
print("Nested Cross-Validation: Outer 5-Fold (Test), Inner 3-Fold (Tuning)")
print("="*70)

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = {
    'rf': {'mae_folds': [], 'predictions': [], 'train_times': [], 'best_params': [], 'models': [], 'data': []},
    'xgb': {'mae_folds': [], 'predictions': [], 'train_times': [], 'best_params': [], 'models': [], 'data': []},
    'et': {'mae_folds': [], 'predictions': [], 'train_times': [], 'best_params': [], 'models': [], 'data': []},
    'svr': {'mae_folds': [], 'predictions': [], 'train_times': [], 'best_params': [], 'models': [], 'data': []},
    'lr': {'mae_folds': [], 'predictions': [], 'train_times': [], 'best_params': [], 'models': [], 'data': []}
}

fold_num = 0
for train_idx, test_idx in outer_cv.split(X):
    fold_num += 1
    print(f"\n{'='*70}")
    print(f"Outer Fold {fold_num}/5")
    print(f"{'='*70}")
    
    # แบ่งข้อมูล
    X_train_raw, X_test_raw = X[train_idx].copy(), X[test_idx].copy()
    y_train_raw, y_test_raw = y[train_idx].copy(), y[test_idx].copy()
    
    # === PREPROCESSING PIPELINE (Fit บน Training เท่านั้น) ===
    
    # 1. Imputation (fit บน training)
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)
    
    # 2. Outlier Removal (เฉพาะ training set)
    X_train_clean, y_train_clean = remove_outliers_zscore(X_train_imputed, y_train_raw, threshold=3)
    print(f"  Removed {len(y_train_raw) - len(y_train_clean)} outliers from training set")
    
    # 3. Scaling (fit บน training ที่ clean แล้ว)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_imputed)  # ใช้ scaler จาก training
    
    print(f"  Training samples: {len(X_train_scaled)}, Test samples: {len(X_test_scaled)}")
    
    # === TRAINING MODELS ===
    
    # Random Forest
    print(f"\n[Fold {fold_num}] Training Random Forest with Inner 3-Fold CV...")
    start_time = time.time()
    rf_model = RandomForestRegressor(random_state=42)
    grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=3, n_jobs=-1, 
                                   scoring='neg_mean_absolute_error', verbose=0)
    grid_search_rf.fit(X_train_scaled, y_train_clean)
    rf_best = grid_search_rf.best_estimator_
    rf_pred = rf_best.predict(X_test_scaled)
    rf_mae = mean_absolute_error(y_test_raw, rf_pred)
    rf_time = time.time() - start_time
    results['rf']['mae_folds'].append(rf_mae)
    results['rf']['predictions'].append({'y_true': y_test_raw, 'y_pred': rf_pred})
    results['rf']['train_times'].append(rf_time)
    results['rf']['best_params'].append(grid_search_rf.best_params_)
    results['rf']['models'].append(rf_best)
    results['rf']['data'].append({
        'X_train': X_train_scaled, 'X_test': X_test_scaled, 
        'y_train': y_train_clean, 'y_test': y_test_raw,
        'scaler': scaler, 'imputer': imputer
    })
    print(f"  Best params: {grid_search_rf.best_params_}")
    print(f"  Validation MAE: {rf_mae:.4f}")
    print(f"  Training time: {rf_time:.4f} seconds")
    
    # XGBoost
    print(f"\n[Fold {fold_num}] Training XGBoost with Inner 3-Fold CV...")
    start_time = time.time()
    xgb_model = XGBRegressor(random_state=42, booster='gbtree', enable_categorical=False)
    grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, n_jobs=-1, 
                                    scoring='neg_mean_absolute_error', verbose=0)
    grid_search_xgb.fit(X_train_scaled, y_train_clean)
    xgb_best = grid_search_xgb.best_estimator_
    xgb_pred = xgb_best.predict(X_test_scaled)
    xgb_mae = mean_absolute_error(y_test_raw, xgb_pred)
    xgb_time = time.time() - start_time
    results['xgb']['mae_folds'].append(xgb_mae)
    results['xgb']['predictions'].append({'y_true': y_test_raw, 'y_pred': xgb_pred})
    results['xgb']['train_times'].append(xgb_time)
    results['xgb']['best_params'].append(grid_search_xgb.best_params_)
    results['xgb']['models'].append(xgb_best)
    results['xgb']['data'].append({
        'X_train': X_train_scaled, 'X_test': X_test_scaled, 
        'y_train': y_train_clean, 'y_test': y_test_raw,
        'scaler': scaler, 'imputer': imputer
    })
    print(f"  Best params: {grid_search_xgb.best_params_}")
    print(f"  Validation MAE: {xgb_mae:.4f}")
    print(f"  Training time: {xgb_time:.4f} seconds")
    
    # Extra Trees
    print(f"\n[Fold {fold_num}] Training Extra Trees with Inner 3-Fold CV...")
    start_time = time.time()
    et_model = ExtraTreesRegressor(random_state=42)
    grid_search_et = GridSearchCV(et_model, param_grid_et, cv=3, n_jobs=-1, 
                                   scoring='neg_mean_absolute_error', verbose=0)
    grid_search_et.fit(X_train_scaled, y_train_clean)
    et_best = grid_search_et.best_estimator_
    et_pred = et_best.predict(X_test_scaled)
    et_mae = mean_absolute_error(y_test_raw, et_pred)
    et_time = time.time() - start_time
    results['et']['mae_folds'].append(et_mae)
    results['et']['predictions'].append({'y_true': y_test_raw, 'y_pred': et_pred})
    results['et']['train_times'].append(et_time)
    results['et']['best_params'].append(grid_search_et.best_params_)
    results['et']['models'].append(et_best)
    results['et']['data'].append({
        'X_train': X_train_scaled, 'X_test': X_test_scaled, 
        'y_train': y_train_clean, 'y_test': y_test_raw,
        'scaler': scaler, 'imputer': imputer
    })
    print(f"  Best params: {grid_search_et.best_params_}")
    print(f"  Validation MAE: {et_mae:.4f}")
    print(f"  Training time: {et_time:.4f} seconds")
    
    # SVR
    print(f"\n[Fold {fold_num}] Training SVR with Inner 3-Fold CV...")
    start_time = time.time()
    svr_model = SVR()
    grid_search_svr = GridSearchCV(svr_model, param_grid_svr, cv=3, n_jobs=-1, 
                                    scoring='neg_mean_absolute_error', verbose=0)
    grid_search_svr.fit(X_train_scaled, y_train_clean)
    svr_best = grid_search_svr.best_estimator_
    svr_pred = svr_best.predict(X_test_scaled)
    svr_mae = mean_absolute_error(y_test_raw, svr_pred)
    svr_time = time.time() - start_time
    results['svr']['mae_folds'].append(svr_mae)
    results['svr']['predictions'].append({'y_true': y_test_raw, 'y_pred': svr_pred})
    results['svr']['train_times'].append(svr_time)
    results['svr']['best_params'].append(grid_search_svr.best_params_)
    results['svr']['models'].append(svr_best)
    results['svr']['data'].append({
        'X_train': X_train_scaled, 'X_test': X_test_scaled, 
        'y_train': y_train_clean, 'y_test': y_test_raw,
        'scaler': scaler, 'imputer': imputer
    })
    print(f"  Best params: {grid_search_svr.best_params_}")
    print(f"  Validation MAE: {svr_mae:.4f}")
    print(f"  Training time: {svr_time:.4f} seconds")
    
    # Linear Regression
    print(f"\n[Fold {fold_num}] Training Linear Regression with Inner 3-Fold CV...")
    start_time = time.time()
    lr_model = LinearRegression()
    grid_search_lr = GridSearchCV(lr_model, param_grid_lr, cv=3, n_jobs=-1, 
                                   scoring='neg_mean_absolute_error', verbose=0)
    grid_search_lr.fit(X_train_scaled, y_train_clean)
    lr_best = grid_search_lr.best_estimator_
    lr_pred = lr_best.predict(X_test_scaled)
    lr_mae = mean_absolute_error(y_test_raw, lr_pred)
    lr_time = time.time() - start_time
    results['lr']['mae_folds'].append(lr_mae)
    results['lr']['predictions'].append({'y_true': y_test_raw, 'y_pred': lr_pred})
    results['lr']['train_times'].append(lr_time)
    results['lr']['best_params'].append(grid_search_lr.best_params_)
    results['lr']['models'].append(lr_best)
    results['lr']['data'].append({
        'X_train': X_train_scaled, 'X_test': X_test_scaled, 
        'y_train': y_train_clean, 'y_test': y_test_raw,
        'scaler': scaler, 'imputer': imputer
    })
    print(f"  Best params: {grid_search_lr.best_params_}")
    print(f"  Validation MAE: {lr_mae:.4f}")
    print(f"  Training time: {lr_time:.4f} seconds")

# 5. สรุปผล
print("\n" + "="*70)
print("Summary: MAE from All 5 Folds")
print("="*70)

models_info = [
    ('Random Forest', 'rf'),
    ('XGBoost', 'xgb'),
    ('Extra Trees', 'et'),
    ('SVR', 'svr'),
    ('Linear Regression', 'lr')
]

print("\n{:<25} {:<50} {:<20}".format("Model", "MAE per Fold", "Mean ± Std"))
print("-" * 95)

for model_name, model_key in models_info:
    mae_list = results[model_key]['mae_folds']
    mae_str = ", ".join([f"{mae:.4f}" for mae in mae_list])
    mean_std = f"{np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}"
    print("{:<25} {:<50} {:<20}".format(model_name, mae_str, mean_std))

# Training Time Summary
print("\n" + "="*70)
print("Training Time Summary")
print("="*70)
print("\n{:<25} {:<20}".format("Model", "Mean Train Time (sec)"))
print("-" * 45)

for model_name, model_key in models_info:
    train_times = results[model_key]['train_times']
    mean_time = np.mean(train_times)
    print("{:<25} {:<20.4f}".format(model_name, mean_time))

# 6. Evaluation Function
def evaluate_model_all_folds(model_name, predictions_list, mae_folds):
    print(f"\n{model_name}:")
    
    rmse_list = []
    r2_list = []
    mape_list = []
    accuracy_list = []
    threshold = 10
    smape_list = []
    
    for pred_dict in predictions_list:
        y_true = pred_dict['y_true']
        y_pred = pred_dict['y_pred']

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        rmse_list.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2_list.append(r2_score(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
        smape_list.append(smape(y_true, y_pred))
        mape_list.append(mape)
        accuracy_list.append(np.mean(np.abs(y_true - y_pred) <= threshold) * 100)

    print(f"  MAE: {np.mean(mae_folds):.4f} ± {np.std(mae_folds):.4f}")
    print(f"  RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
    print(f"  R²: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
    print(f"  MAPE: {np.mean(mape_list):.2f}% ± {np.std(mape_list):.2f}%")
    print(f"  sMAPE: {np.mean(smape_list):.2f}% ± {np.std(smape_list):.2f}%")
    print(f"  Accuracy (±{threshold}): {np.mean(accuracy_list):.2f}% ± {np.std(accuracy_list):.2f}%")

# 7. Overall Performance
print("\n" + "="*70)
print("Overall Performance Metrics (Mean ± Std from All 5 Folds)")
print("="*70)

for model_name, model_key in models_info:
    predictions_list = results[model_key]['predictions']
    mae_folds = results[model_key]['mae_folds']
    evaluate_model_all_folds(model_name, predictions_list, mae_folds)

print("\n" + "="*70)
print("Nested Cross-Validation Complete!")
print("="*70)
# เพิ่มส่วนนี้หลังจาก section "Nested Cross-Validation Complete!"
# ============================================================================
# VISUALIZATION: Plot All Models Across All Folds
# ============================================================================

print("\n" + "="*70)
print("Plotting Predictions: All Models Across All 5 Folds")
print("="*70)

# Models to plot
plot_models = [
    ('Random Forest', 'rf'),
    ('XGBoost', 'xgb'),
    ('Extra Trees', 'et'),
    ('SVR', 'svr'),
    ('Linear Regression', 'lr')
]

# Plot each model across all folds
for model_name, model_key in plot_models:
    print(f"\n{'-'*70}")
    print(f"Plotting {model_name}: Folds 1-5")
    print(f"{'-'*70}")
    
    for fold_idx in range(5):
        y_true = results[model_key]['predictions'][fold_idx]['y_true']
        y_pred = results[model_key]['predictions'][fold_idx]['y_pred']
        mae = results[model_key]['mae_folds'][fold_idx]
        
        # Calculate additional metrics for this fold
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        plt.figure(figsize=(10, 7))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, color='darkblue', alpha=0.6, s=40, 
                   edgecolors='black', linewidth=0.5, label='Predictions')
        
        # Ideal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                color='red', linestyle='--', linewidth=2.5, label='Perfect Prediction')
        plt.xticks(fontsize=14)  # ปรับตามที่ต้องการ
        plt.yticks(fontsize=14)
        # Labels and title
        plt.xlabel('Actual DC_POWER (kW)', fontsize=13, fontweight='bold')
        plt.ylabel('Predicted DC_POWER (kW)', fontsize=13, fontweight='bold')
        plt.title(f'{model_name} - Fold {fold_idx + 1}/5',
                 fontsize=12, fontweight='bold', pad=15)
        
        # Grid and legend
        plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        plt.legend(loc='upper left', fontsize=13, framealpha=0.9)
        
        # Equal aspect ratio
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

# ============================================================================
# SHAP ANALYSIS - สำหรับทุกโมเดลจาก Fold ที่มี MAE ต่ำที่สุด
# ============================================================================
# ============================================================================
# SHAP ANALYSIS - สำหรับทุกโมเดลจาก Fold ที่มี MAE ต่ำที่สุด
# ============================================================================

print("\n" + "="*70)
print("SHAP Analysis: Explaining Models from Best Fold")
print("="*70)

feature_names = ['IRRADIATION', 'MODULE_TEMPERATURE']

for model_name, model_key in models_info:
    print(f"\n{'='*70}")
    print(f"SHAP Analysis: {model_name}")
    print(f"{'='*70}")
    
    # หา fold ที่มี MAE ต่ำที่สุด
    mae_folds = results[model_key]['mae_folds']
    best_fold_idx = np.argmin(mae_folds)
    best_mae = mae_folds[best_fold_idx]
    
    print(f"\nBest Fold: {best_fold_idx + 1}/5 (MAE: {best_mae:.4f})")
    
    # ดึงโมเดลและข้อมูลจาก fold ที่ดีที่สุด
    best_model = results[model_key]['models'][best_fold_idx]
    best_data = results[model_key]['data'][best_fold_idx]
    X_train_best = best_data['X_train']
    X_test_best = best_data['X_test']
    
    # สุ่มตัวอย่างถ้าข้อมูลเยอะเกินไป (สำหรับ SHAP ที่รวดเร็ว)
    max_samples = 1000
    if len(X_test_best) > max_samples:
        sample_idx = np.random.choice(len(X_test_best), max_samples, replace=False)
        X_test_sample = X_test_best[sample_idx]
    else:
        X_test_sample = X_test_best
    
    try:
        # เลือก explainer ตามประเภทโมเดล
        if model_key in ['rf', 'et', 'xgb']:
            # Tree-based models: ใช้ TreeExplainer
            print(f"Using TreeExplainer for {model_name}...")
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test_sample)
        elif model_key == 'lr':
            # Linear model: ใช้ LinearExplainer
            print(f"Using LinearExplainer for {model_name}...")
            explainer = shap.LinearExplainer(best_model, X_train_best)
            shap_values = explainer.shap_values(X_test_sample)
        else:  # SVR
            # Kernel-based models: ใช้ KernelExplainer
            print(f"Using KernelExplainer for {model_name} (this may take a while)...")
            background_size = min(100, len(X_train_best))
            background = X_train_best[np.random.choice(len(X_train_best), background_size, replace=False)]
            explainer = shap.KernelExplainer(best_model.predict, background)
            shap_values = explainer.shap_values(X_test_sample)
        
        # 1. Summary Plot (Bar) - Feature Importance
        print(f"\nGenerating SHAP Summary Plot (Feature Importance)...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.title(f'{model_name} - SHAP Feature Importance (Best Fold {best_fold_idx + 1})', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # 2. Summary Plot (Beeswarm) - Detailed Impact
        print(f"Generating SHAP Summary Plot (Beeswarm)...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
        plt.title(f'{model_name} - SHAP Summary Plot (Best Fold {best_fold_idx + 1})', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # 3. Dependence Plots - แสดงความสัมพันธ์ของแต่ละ feature
        for i, feature_name in enumerate(feature_names):
            print(f"Generating SHAP Dependence Plot for {feature_name}...")
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(i, shap_values, X_test_sample, 
                               feature_names=feature_names, show=False)
            plt.title(f'{model_name} - SHAP Dependence: {feature_name} (Best Fold {best_fold_idx + 1})', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        # 4. Force Plot - แสดงตัวอย่างการทำนาย (5 ตัวอย่างแรก)
        print(f"Generating SHAP Force Plots for first 5 predictions...")
        num_examples = min(5, len(X_test_sample))
        
        # Get expected value
        if hasattr(explainer, 'expected_value'):
            expected_value = explainer.expected_value
            # For tree models, expected_value might be an array
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[0] if len(expected_value) > 0 else expected_value
        else:
            expected_value = np.mean(shap_values)
        
        for idx in range(num_examples):
            plt.figure(figsize=(12, 3))
            shap.force_plot(expected_value, shap_values[idx], X_test_sample[idx], 
                          feature_names=feature_names, matplotlib=True, show=False)
            plt.title(f'{model_name} - SHAP Force Plot: Sample {idx + 1} (Best Fold {best_fold_idx + 1})', 
                     fontsize=12, fontweight='bold', pad=10)
            plt.tight_layout()
            plt.show()
        
        # 5. Mean Absolute SHAP Values
        mean_shap = np.abs(shap_values).mean(axis=0)
        print(f"\nMean Absolute SHAP Values for {model_name} (Best Fold {best_fold_idx + 1}):")
        for i, feature_name in enumerate(feature_names):
            print(f"  {feature_name}: {mean_shap[i]:.6f}")
        
        # 6. Feature Importance Summary
        print(f"\nFeature Importance Ranking for {model_name}:")
        importance_pairs = [(feature_names[i], mean_shap[i]) for i in range(len(feature_names))]
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        for rank, (feat, imp) in enumerate(importance_pairs, 1):
            print(f"  {rank}. {feat}: {imp:.6f}")
        
    except Exception as e:
        print(f"Error generating SHAP plots for {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*70)
print("SHAP Analysis Complete!")
print("="*70)

# ============================================================================
# SHAP COMPARISON: Feature Importance Across All Models
# ============================================================================

print("\n" + "="*70)
print("SHAP Comparison: Feature Importance Across All Models (Best Folds)")
print("="*70)

# Collect SHAP values from best fold of each model
shap_importance_dict = {}

for model_name, model_key in models_info:
    try:
        mae_folds = results[model_key]['mae_folds']
        best_fold_idx = np.argmin(mae_folds)
        best_model = results[model_key]['models'][best_fold_idx]
        best_data = results[model_key]['data'][best_fold_idx]
        X_test_best = best_data['X_test']
        
        # Sample data
        max_samples = 100
        if len(X_test_best) > max_samples:
            sample_idx = np.random.choice(len(X_test_best), max_samples, replace=False)
            X_test_sample = X_test_best[sample_idx]
        else:
            X_test_sample = X_test_best
        
        # Calculate SHAP values
        if model_key in ['rf', 'et', 'xgb']:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test_sample)
        elif model_key == 'lr':
            explainer = shap.LinearExplainer(best_model, best_data['X_train'])
            shap_values = explainer.shap_values(X_test_sample)
        else:  # SVR
            background = best_data['X_train'][np.random.choice(len(best_data['X_train']), 
                                                                min(100, len(best_data['X_train'])), 
                                                                replace=False)]
            explainer = shap.KernelExplainer(best_model.predict, background)
            shap_values = explainer.shap_values(X_test_sample)
        
        # Store mean absolute SHAP values
        shap_importance_dict[model_name] = np.abs(shap_values).mean(axis=0)
        
    except Exception as e:
        print(f"Warning: Could not compute SHAP for {model_name}: {str(e)}")
        continue

# Plot comparison
if shap_importance_dict:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(feature_names))
    width = 0.15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (model_name, importance) in enumerate(shap_importance_dict.items()):
        offset = width * (idx - len(shap_importance_dict)/2 + 0.5)
        ax.bar(x + offset, importance, width, label=model_name, 
               color=colors[idx % len(colors)], alpha=0.8, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Features', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean |SHAP Value|', fontsize=13, fontweight='bold')
    ax.set_title('Feature Importance Comparison (SHAP) - All Models (Best Folds)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\nFeature Importance Summary (Mean |SHAP Value|):")
    print(f"{'Model':<25} {'IRRADIATION':<20} {'MODULE_TEMPERATURE':<20}")
    print("-" * 65)
    for model_name, importance in shap_importance_dict.items():
        print(f"{model_name:<25} {importance[0]:<20.6f} {importance[1]:<20.6f}")

print("\n" + "="*70)
print("All SHAP Analyses Complete!")
print("="*70)

# ============================================================================
# COMPARISON: All Models in Fold 5 (Side by Side Summary)
# ============================================================================

print("\n" + "="*70)
print("Comparison: All Models in Fold 5")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (model_name, model_key) in enumerate(plot_models):
    ax = axes[idx]
    
    y_true = results[model_key]['predictions'][4]['y_true']  # Fold 5
    y_pred = results[model_key]['predictions'][4]['y_pred']
    mae = results[model_key]['mae_folds'][4]
    r2 = r2_score(y_true, y_pred)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, color='darkgreen', alpha=0.5, s=30, 
              edgecolors='black', linewidth=0.3)
    
    # Ideal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Actual DC_POWER (kW)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted DC_POWER (kW)', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name}\nMAE: {mae:.4f} | R²: {r2:.4f}', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

# Hide the 6th subplot (since we only have 5 models)
axes[5].axis('off')

plt.suptitle('All Models Comparison - Fold 5', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# ============================================================================
# MAE COMPARISON: Bar Chart Across All Folds
# ============================================================================

print("\n" + "="*70)
print("MAE Comparison: All Models Across All Folds")
print("="*70)

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(5)  # 5 folds
width = 0.15  # bar width

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for idx, (model_name, model_key) in enumerate(plot_models):
    mae_values = results[model_key]['mae_folds']
    offset = width * (idx - 2)  # Center the bars
    ax.bar(x + offset, mae_values, width, label=model_name, 
          color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.8)

ax.set_xlabel('Fold Number', fontsize=13, fontweight='bold')
ax.set_ylabel('MAE', fontsize=13, fontweight='bold')
ax.set_title('MAE Comparison: All Models Across 5 Folds', 
            fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.show()

# ============================================================================
# BOX PLOT: MAE Distribution Across Folds
# ============================================================================

print("\n" + "="*70)
print("MAE Distribution: Box Plot Across All Folds")
print("="*70)

fig, ax = plt.subplots(figsize=(12, 7))

mae_data = []
labels = []

for model_name, model_key in plot_models:
    mae_data.append(results[model_key]['mae_folds'])
    labels.append(model_name)

bp = ax.boxplot(mae_data, labels=labels, patch_artist=True, 
               notch=True, widths=0.6)

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Model', fontsize=13, fontweight='bold')
ax.set_ylabel('MAE', fontsize=13, fontweight='bold')
ax.set_title('MAE Distribution Across 5 Folds (All Models)', 
            fontsize=15, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("All Visualizations Complete!")
print("="*70)