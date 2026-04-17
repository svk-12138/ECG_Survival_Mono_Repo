"""模型复杂度匹配规则应用工具

改进的模型选型规则，基于三维数据复杂度评估：
1. PCA有效维度（Shannon熵估计）
2. 频域复杂度（归一化频谱熵）
3. 样本稀疏度（k近邻平均距离）

应用场景：
- 根据数据规模和复杂度推荐合适的模型架构
- 避免过拟合（模型过复杂）或欠拟合（模型过简单）
- 输出1200样本和10000样本两种场景的推荐模型列表
"""

import numpy as np
import json
import base64
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.signal import welch, resample


def load_ecg_waveform(xml_path, target_len=4096):
    """加载ECG波形，返回(8, 4096)数组"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        leads_order = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        for wf in root.findall('.//Waveform'):
            if wf.findtext('WaveformType') != 'Rhythm':
                continue
            leads = {}
            for lead in wf.findall('LeadData'):
                lead_id = lead.findtext('LeadID', '')
                data = lead.findtext('WaveFormData', '')
                if data and lead_id in leads_order:
                    try:
                        decoded = base64.b64decode(data)
                        arr = np.frombuffer(decoded, dtype=np.int16).astype(np.float32)
                        leads[lead_id] = arr
                    except:
                        pass

            if len(leads) == 8:
                signals = []
                for lead_id in leads_order:
                    sig = leads[lead_id]
                    if len(sig) != target_len:
                        sig = resample(sig, target_len)
                    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
                    signals.append(sig)
                return np.stack(signals)
    except:
        pass
    return None


def analyze_data_complexity_improved(manifest_path, max_samples=300):
    """改进的三维数据复杂度评估

    Args:
        manifest_path: manifest.json路径
        max_samples: 最多加载多少个样本做分析（避免太慢）

    Returns:
        dict: 包含三维复杂度指标
    """
    print(f"加载ECG波形（最多{max_samples}个样本）...")

    with open(manifest_path) as f:
        manifest = json.load(f)

    waveforms = []
    for i, entry in enumerate(manifest[:max_samples]):
        if i % 50 == 0:
            print(f"  进度: {i}/{min(len(manifest), max_samples)}")
        xml_path = entry.get('xml_path')
        if xml_path:
            wf = load_ecg_waveform(xml_path)
            if wf is not None:
                waveforms.append(wf)

    if len(waveforms) == 0:
        raise ValueError("未能加载任何ECG波形")

    X_raw = np.array(waveforms)  # (N, 8, 4096)
    print(f"成功加载: {len(X_raw)} 个样本\n")

    # === D1: PCA有效维度 ===
    X_flat = X_raw.reshape(len(X_raw), -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    pca = PCA(n_components=min(150, len(X_raw)-1))
    pca.fit(X_scaled)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n95 = np.argmax(cumsum >= 0.95) + 1
    n99 = np.argmax(cumsum >= 0.99) + 1

    var_ratios = pca.explained_variance_ratio_
    var_ratios = var_ratios[var_ratios > 1e-10]
    entropy = -np.sum(var_ratios * np.log(var_ratios + 1e-10))
    pca_intrinsic_dim = np.exp(entropy)

    # === D2: 频域复杂度 ===
    spectral_entropies = []
    for i in range(len(X_raw)):
        for ch in range(8):
            f, psd = welch(X_raw[i, ch], fs=500, nperseg=256)
            psd_norm = psd / (psd.sum() + 1e-10)
            se = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            spectral_entropies.append(se)

    mean_spectral_entropy = np.mean(spectral_entropies)
    max_spectral_entropy = np.log(len(f))
    spectral_complexity = mean_spectral_entropy / max_spectral_entropy

    # === D3: 样本稀疏度 ===
    X_pca50 = pca.transform(X_scaled)[:, :50]
    nbrs = NearestNeighbors(n_neighbors=min(10, len(X_raw)-1)).fit(X_pca50)
    distances, _ = nbrs.kneighbors(X_pca50)
    mean_nn_dist = distances[:, 1:].mean()
    max_dist = np.sqrt(50) * 2
    density_score = mean_nn_dist / max_dist

    # === 综合有效复杂度 ===
    effective_complexity = pca_intrinsic_dim * spectral_complexity * (1 + density_score)

    return {
        'n_samples_analyzed': len(X_raw),
        'pca_intrinsic_dim': float(pca_intrinsic_dim),
        'n_components_95': int(n95),
        'n_components_99': int(n99),
        'spectral_complexity': float(spectral_complexity),
        'density_score': float(density_score),
        'effective_complexity': float(effective_complexity),
        'explained_variance_ratio_top10': pca.explained_variance_ratio_[:10].tolist(),
    }


def calculate_param_constraints(effective_complexity, n_samples):
    """计算参数量约束"""
    max_params = effective_complexity * n_samples / 10
    min_params = effective_complexity * 100
    return int(max_params), int(min_params)


def model_score(n_params, n_layers):
    """计算模型复杂度得分：log10(params) / log10(layers)"""
    if n_layers <= 1:
        n_layers = 2
    return np.log10(n_params) / np.log10(n_layers)


def get_candidate_models():
    """定义候选模型库（ECG时序场景）"""
    return [
        # 传统ML
        {
            'name': 'Cox PH + ECG特征',
            'params': 200,
            'layers': 1,
            'category': '传统ML',
            'description': '统计学基线，医生最认可'
        },
        {
            'name': 'XGBoost (浅层, depth=3)',
            'params': 5000,
            'layers': 6,
            'category': '传统ML',
            'description': '传统ML，特征工程+临床特征'
        },
        {
            'name': 'XGBoost (标准, depth=6)',
            'params': 50000,
            'layers': 6,
            'category': '传统ML',
            'description': '传统ML，特征工程+临床特征'
        },
        {
            'name': 'LightGBM',
            'params': 30000,
            'layers': 8,
            'category': '传统ML',
            'description': '传统ML，速度快，小样本友好'
        },

        # 轻量深度学习
        {
            'name': 'MiniCNN (2层, ch=16/32)',
            'params': 12000,
            'layers': 4,
            'category': '轻量深度学习',
            'description': '极轻量CNN，适合小样本'
        },
        {
            'name': 'MiniCNN (3层, ch=32/64/128)',
            'params': 28000,
            'layers': 5,
            'category': '轻量深度学习',
            'description': '轻量CNN，接近参数上限'
        },
        {
            'name': 'TCN (时序卷积, 轻量)',
            'params': 25000,
            'layers': 6,
            'category': '轻量深度学习',
            'description': '因果卷积，适合时序，无信息泄露'
        },
        {
            'name': 'S4/Mamba (轻量版)',
            'params': 20000,
            'layers': 4,
            'category': '轻量深度学习',
            'description': '状态空间模型，长序列高效'
        },

        # 中等深度学习
        {
            'name': 'ResNet1d (小版, 4块)',
            'params': 120000,
            'layers': 9,
            'category': '中等深度学习',
            'description': '残差CNN，论文同款架构'
        },
        {
            'name': 'CNN + GRU',
            'params': 80000,
            'layers': 6,
            'category': '中等深度学习',
            'description': 'CNN提特征+GRU时序建模'
        },
        {
            'name': 'TCN (标准)',
            'params': 150000,
            'layers': 8,
            'category': '中等深度学习',
            'description': '时序卷积网络，标准版'
        },
        {
            'name': 'MiniTransformer (2层)',
            'params': 100000,
            'layers': 5,
            'category': '中等深度学习',
            'description': '轻量Transformer，需要足够样本'
        },

        # 过重（参考用）
        {
            'name': 'CNN-Transformer (实验性)',
            'params': 692992,
            'layers': 7,
            'category': '过重模型',
            'description': '实验性模型，参数量过大'
        },
        {
            'name': 'ResNet1d (论文原版)',
            'params': 2944000,
            'layers': 9,
            'category': '过重模型',
            'description': '葛均波论文模型，需要40万样本'
        },
    ]


def classify_model(n_params, max_params, min_params):
    """分类模型状态"""
    if n_params < min_params:
        return '欠拟合风险', '⚠'
    elif n_params <= max_params:
        return '推荐', '✓'
    else:
        overfitting_ratio = n_params / max_params
        if overfitting_ratio < 2:
            return '过拟合风险', '✗'
        else:
            return f'超标{overfitting_ratio:.1f}倍', '✗✗'


def generate_recommendations(data_complexity, n_samples):
    """生成模型推荐列表"""
    E = data_complexity['effective_complexity']
    max_params, min_params = calculate_param_constraints(E, n_samples)

    candidates = get_candidate_models()
    results = []

    for model in candidates:
        n_params = model['params']
        n_layers = model['layers']
        score = model_score(n_params, n_layers)
        status, mark = classify_model(n_params, max_params, min_params)

        results.append({
            'name': model['name'],
            'params': n_params,
            'layers': n_layers,
            'score': float(score),
            'category': model['category'],
            'description': model['description'],
            'status': status,
            'mark': mark,
            'recommended': status == '推荐'
        })

    # 按得分排序
    results.sort(key=lambda x: x['score'])

    return {
        'n_samples': n_samples,
        'effective_complexity': E,
        'max_params': max_params,
        'min_params': min_params,
        'models': results
    }


def print_report(data_complexity, recommendations_1200, recommendations_10000):
    """打印可读性报告"""
    print("=" * 80)
    print("模型复杂度匹配规则应用报告")
    print("=" * 80)

    print("\n【数据复杂度分析】")
    print(f"  分析样本数: {data_complexity['n_samples_analyzed']}")
    print(f"  D1 (PCA有效维度):    {data_complexity['pca_intrinsic_dim']:.1f}")
    print(f"  D2 (频域复杂度):     {data_complexity['spectral_complexity']:.3f}")
    print(f"  D3 (样本稀疏度):     {data_complexity['density_score']:.3f}")
    print(f"  E  (综合有效复杂度): {data_complexity['effective_complexity']:.1f}")

    for scenario_name, rec in [('1200样本', recommendations_1200), ('10000样本', recommendations_10000)]:
        print(f"\n{'=' * 80}")
        print(f"模型选型列表 — {scenario_name}")
        print(f"{'=' * 80}")
        print(f"推荐参数量范围: {rec['min_params']:,} ~ {rec['max_params']:,}")
        print()
        print(f"{'模型':<30} {'参数量':>10} {'层数':>5} {'得分':>6}  {'状态':<12} 说明")
        print("-" * 80)

        for m in rec['models']:
            print(f"{m['name']:<30} {m['params']:>10,} {m['layers']:>5} {m['score']:>6.2f}  "
                  f"{m['mark']} {m['status']:<10}  {m['description']}")

    print(f"\n{'=' * 80}")
    print("关键结论")
    print("=" * 80)

    rec_1200 = [m['name'] for m in recommendations_1200['models'] if m['recommended']]
    rec_10000 = [m['name'] for m in recommendations_10000['models'] if m['recommended']]

    print(f"\n1200样本推荐模型:")
    for name in rec_1200[:5]:
        print(f"  ✓ {name}")

    print(f"\n10000样本推荐模型:")
    for name in rec_10000[:5]:
        print(f"  ✓ {name}")

    print(f"\n模型得分（log10(params)/log10(layers)）越小越轻量：")
    print(f"  得分 < 4.0 → 传统ML / 极轻量模型")
    print(f"  得分 4-6   → 轻量深度学习（1200样本可用区间）")
    print(f"  得分 6-7   → 中等深度学习（10000样本可用区间）")
    print(f"  得分 > 7   → 重型模型（需要10万+样本）")


def save_results(output_dir, data_complexity, recommendations_1200, recommendations_10000):
    """保存结果到文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存数据复杂度
    with open(output_dir / 'data_complexity.json', 'w') as f:
        json.dump(data_complexity, f, indent=2)

    # 保存推荐结果
    with open(output_dir / 'model_recommendations_1200.json', 'w') as f:
        json.dump(recommendations_1200, f, indent=2)

    with open(output_dir / 'model_recommendations_10000.json', 'w') as f:
        json.dump(recommendations_10000, f, indent=2)

    # 保存可读性报告
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = report_buffer = StringIO()

    print_report(data_complexity, recommendations_1200, recommendations_10000)

    sys.stdout = old_stdout
    report_text = report_buffer.getvalue()

    with open(output_dir / 'selection_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n结果已保存到: {output_dir}/")
    print(f"  - data_complexity.json")
    print(f"  - model_recommendations_1200.json")
    print(f"  - model_recommendations_10000.json")
    print(f"  - selection_report.txt")


if __name__ == '__main__':
    manifest_path = 'data/stroke_1113/manifest.json'
    output_dir = 'outputs/model_selection'

    # 分析数据复杂度
    data_complexity = analyze_data_complexity_improved(manifest_path, max_samples=300)

    # 生成推荐
    recommendations_1200 = generate_recommendations(data_complexity, n_samples=1200)
    recommendations_10000 = generate_recommendations(data_complexity, n_samples=10000)

    # 打印报告
    print_report(data_complexity, recommendations_1200, recommendations_10000)

    # 保存结果
    save_results(output_dir, data_complexity, recommendations_1200, recommendations_10000)
