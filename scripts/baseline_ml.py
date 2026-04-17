"""传统机器学习基线：XGBoost + 临床特征 + 简单ECG特征"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy import stats
import xml.etree.ElementTree as ET
import base64
import warnings
warnings.filterwarnings('ignore')


def extract_ecg_features(xml_path):
    """提取简单的ECG特征：心率、QRS时长、PR间期、QT间期"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 从RestingECGMeasurements提取
        measurements = root.find('.//RestingECGMeasurements')
        if measurements is None:
            return {}

        features = {}

        # 心率
        hr = measurements.findtext('VentricularRate')
        if hr:
            features['heart_rate'] = float(hr)

        # QRS时长
        qrs = measurements.findtext('QRSDuration')
        if qrs:
            features['qrs_duration'] = float(qrs)

        # PR间期
        pr = measurements.findtext('PRInterval')
        if pr:
            features['pr_interval'] = float(pr)

        # QT间期
        qt = measurements.findtext('QTInterval')
        if qt:
            features['qt_interval'] = float(qt)

        # QTc (校正QT)
        qtc = measurements.findtext('QTCorrected')
        if qtc:
            features['qtc'] = float(qtc)

        return features
    except Exception as e:
        return {}


def load_data(manifest_path):
    """加载数据并提取特征"""
    with open(manifest_path) as f:
        manifest = json.load(f)

    data = []
    for entry in manifest:
        # 临床特征
        meta = entry.get('meta', {})
        features = {
            'age': meta.get('age', 65),
            'sex': meta.get('sex', 1),
            'atrial_fibrillation': meta.get('atrial_fibrillation', 0),
            'pacemaker': meta.get('pacemaker', 0),
            'arrhythmia': meta.get('arrhythmia', 0),
        }

        # ECG特征
        xml_path = entry.get('xml_path')
        if xml_path:
            ecg_features = extract_ecg_features(xml_path)
            features.update(ecg_features)

        # 标签
        features['time'] = entry['time_to_event']
        features['event'] = entry['event']
        features['patient_id'] = entry['patient_id']

        data.append(features)

    df = pd.DataFrame(data)

    # 填充缺失值
    for col in df.columns:
        if col not in ['time', 'event', 'patient_id']:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)

    return df


def train_xgboost_baseline(df, time_window_days=1095, n_folds=5):
    """训练XGBoost基线模型"""

    # 创建二分类标签：time_window内发生事件
    # event=1且time<=window：正样本
    # event=0且time>=window：负样本（随访足够长但未发生）
    # event=0且time<window：排除（随访不够长，不知道会不会发生）
    # event=1且time>window：排除（事件发生在窗口外）

    df_filtered = df[
        ((df['event'] == 1) & (df['time'] <= time_window_days)) |  # 窗口内事件
        ((df['event'] == 0) & (df['time'] >= time_window_days))    # 随访足够长的非事件
    ].copy()

    df_filtered['label'] = df_filtered['event'].astype(int)

    # 特征列
    feature_cols = [c for c in df_filtered.columns if c not in ['time', 'event', 'label', 'patient_id']]
    X = df_filtered[feature_cols].values
    y = df_filtered['label'].values

    print(f"\n{'='*60}")
    print(f"XGBoost基线 - {time_window_days/365:.1f}年预测窗口")
    print(f"{'='*60}")
    print(f"原始样本数: {len(df)}")
    print(f"过滤后样本数: {len(df_filtered)}")
    print(f"事件数: {y.sum()} ({y.mean():.2%})")
    print(f"特征数: {len(feature_cols)}")
    print(f"特征: {feature_cols}")

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    aucs = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # 训练XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train, verbose=False)

        # 预测
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold+1}: AUROC = {auc:.4f}")

    # 总体指标
    overall_auc = roc_auc_score(all_y_true, all_y_pred)

    print(f"\n{'='*60}")
    print(f"交叉验证结果:")
    print(f"  平均AUROC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  总体AUROC: {overall_auc:.4f}")
    print(f"{'='*60}\n")

    # 特征重要性
    model.fit(scaler.fit_transform(X), y)
    importance = model.feature_importances_
    feature_importance = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)

    print("特征重要性 Top 5:")
    for feat, imp in feature_importance[:5]:
        print(f"  {feat}: {imp:.4f}")

    return {
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'overall_auc': overall_auc,
        'feature_importance': feature_importance,
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred
    }


if __name__ == '__main__':
    manifest_path = 'data/stroke_1113/manifest.json'

    print("加载数据...")
    df = load_data(manifest_path)

    print(f"数据加载完成: {len(df)} 样本")
    print(f"可用特征: {[c for c in df.columns if c not in ['time', 'event', 'patient_id']]}")

    # 测试不同时间窗口
    for years in [1, 3, 5]:
        days = years * 365

        results = train_xgboost_baseline(df, time_window_days=days, n_folds=5)

        # 保存结果
        output_dir = Path('outputs/baseline_ml')
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / f'results_{years}year.json', 'w') as f:
            json.dump({
                'mean_auc': float(results['mean_auc']),
                'std_auc': float(results['std_auc']),
                'overall_auc': float(results['overall_auc']),
                'feature_importance': [(k, float(v)) for k, v in results['feature_importance'][:10]]
            }, f, indent=2)
