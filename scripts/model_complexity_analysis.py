"""模型复杂度匹配分析工具

基于你提出的规则：
1. 对数据做PCA，判断样本复杂程度
2. 找模型内部复杂度和数据空间复杂度相匹配的模型
3. 模型参数取log再除以log(模型层数)，计算得分
4. 得分最小的模型即为比较合适的模型（得分<3）
"""

import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def analyze_data_complexity(X, n_components=None):
    """分析数据复杂度

    Args:
        X: 特征矩阵 (n_samples, n_features)
        n_components: PCA保留的成分数，None则自动选择

    Returns:
        dict: 包含数据复杂度指标
    """
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA分析
    pca = PCA()
    pca.fit(X_scaled)

    # 计算累积方差解释率
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)

    # 找到解释95%方差需要的成分数
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    n_components_99 = np.argmax(cumsum_var >= 0.99) + 1

    # 有效维度（intrinsic dimensionality）
    # 使用Shannon熵估计
    var_ratios = pca.explained_variance_ratio_
    var_ratios = var_ratios[var_ratios > 1e-10]  # 过滤接近0的
    entropy = -np.sum(var_ratios * np.log(var_ratios + 1e-10))
    intrinsic_dim = np.exp(entropy)

    # 数据复杂度得分（归一化到0-10）
    # 考虑：有效维度、样本数、特征数的比例
    n_samples, n_features = X.shape
    sample_feature_ratio = n_samples / n_features

    # 复杂度得分：维度越高、样本越少，复杂度越高
    complexity_score = (intrinsic_dim / n_features) * 10

    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_components_95': int(n_components_95),
        'n_components_99': int(n_components_99),
        'intrinsic_dimensionality': float(intrinsic_dim),
        'sample_feature_ratio': float(sample_feature_ratio),
        'complexity_score': float(complexity_score),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist()[:10],
        'cumsum_variance': cumsum_var.tolist()[:10]
    }


def calculate_model_complexity(n_params, n_layers):
    """计算模型复杂度得分

    按你的规则：log(params) / log(layers)

    Args:
        n_params: 模型参数量
        n_layers: 模型层数

    Returns:
        float: 复杂度得分
    """
    if n_layers <= 1:
        n_layers = 2  # 避免log(1)=0

    score = np.log10(n_params) / np.log10(n_layers)
    return score


def estimate_model_params(model_config):
    """估算模型参数量

    Args:
        model_config: dict, 包含模型配置
            - type: 'cnn', 'transformer', 'resnet', 'lstm'
            - input_dim: (n_channels, seq_len)
            - hidden_dims: list of hidden dimensions
            - n_layers: number of layers

    Returns:
        tuple: (n_params, n_layers)
    """
    model_type = model_config['type']
    input_dim = model_config.get('input_dim', (8, 4096))
    n_channels, seq_len = input_dim

    if model_type == 'cnn':
        # 简单CNN：3层卷积 + 全连接
        conv_channels = model_config.get('conv_channels', [64, 128, 256])
        n_intervals = model_config.get('n_intervals', 20)

        n_params = 0
        in_ch = n_channels
        for out_ch in conv_channels:
            # Conv1d: kernel_size=7
            n_params += in_ch * out_ch * 7
            in_ch = out_ch

        # 全连接层
        n_params += conv_channels[-1] * 128 + 128 * n_intervals

        n_layers = len(conv_channels) + 2

    elif model_type == 'transformer':
        # Transformer: CNN特征提取 + Transformer编码器
        conv_channels = model_config.get('conv_channels', [64, 128, 256])
        d_model = model_config.get('d_model', 128)
        n_heads = model_config.get('n_heads', 4)
        n_transformer_layers = model_config.get('n_transformer_layers', 2)
        n_intervals = model_config.get('n_intervals', 20)

        # CNN部分
        n_params = 0
        in_ch = n_channels
        for out_ch in conv_channels:
            n_params += in_ch * out_ch * 7
            in_ch = out_ch

        # Transformer部分
        # Self-attention: Q,K,V投影 + 输出投影
        n_params += n_transformer_layers * (4 * d_model * d_model)
        # FFN: 2层MLP
        n_params += n_transformer_layers * (d_model * d_model * 4 + d_model * 4 * d_model)

        # 输出头
        n_params += d_model * 64 + 64 * n_intervals

        n_layers = len(conv_channels) + n_transformer_layers + 2

    elif model_type == 'resnet':
        # ResNet1d
        blocks_dim = model_config.get('blocks_dim', [(64, 1024), (128, 256), (196, 64), (256, 16)])
        n_intervals = model_config.get('n_intervals', 20)

        n_params = 0
        in_ch = n_channels
        for out_ch, _ in blocks_dim:
            # 残差块：2个卷积 + shortcut
            n_params += in_ch * out_ch * 17 * 2
            n_params += in_ch * out_ch  # shortcut
            in_ch = out_ch

        # 全连接
        n_params += blocks_dim[-1][0] * n_intervals

        n_layers = len(blocks_dim) * 2 + 1

    elif model_type == 'lstm':
        # LSTM
        hidden_size = model_config.get('hidden_size', 128)
        n_lstm_layers = model_config.get('n_lstm_layers', 2)
        n_intervals = model_config.get('n_intervals', 20)

        # LSTM参数：4个门，每个门有input和hidden的权重
        n_params = n_lstm_layers * (4 * hidden_size * (n_channels + hidden_size))

        # 全连接
        n_params += hidden_size * n_intervals

        n_layers = n_lstm_layers + 1

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return n_params, n_layers


def match_model_to_data(data_complexity, candidate_models):
    """根据数据复杂度匹配模型

    Args:
        data_complexity: dict, 数据复杂度分析结果
        candidate_models: list of dict, 候选模型配置

    Returns:
        list: 排序后的模型列表，按匹配度从高到低
    """
    results = []

    data_score = data_complexity['complexity_score']
    intrinsic_dim = data_complexity['intrinsic_dimensionality']

    for model_config in candidate_models:
        n_params, n_layers = estimate_model_params(model_config)
        model_score = calculate_model_complexity(n_params, n_layers)

        # 匹配度：模型复杂度与数据复杂度的差异
        # 理想情况：model_score ≈ data_score
        mismatch = abs(model_score - data_score)

        # 参数效率：参数量 vs 有效维度
        param_efficiency = n_params / (intrinsic_dim * 1000)  # 归一化

        # 综合得分：越小越好
        final_score = model_score - data_score / 10

        results.append({
            'model_name': model_config.get('name', model_config['type']),
            'model_type': model_config['type'],
            'n_params': int(n_params),
            'n_layers': int(n_layers),
            'model_complexity_score': float(model_score),
            'mismatch': float(mismatch),
            'param_efficiency': float(param_efficiency),
            'final_score': float(final_score),
            'recommended': final_score < 3.0
        })

    # 按final_score排序
    results.sort(key=lambda x: x['final_score'])

    return results


if __name__ == '__main__':
    # 加载基线数据
    import pandas as pd
    from baseline_ml import load_data

    print("=" * 60)
    print("数据复杂度与模型匹配分析")
    print("=" * 60)

    # 加载数据
    manifest_path = 'data/stroke_1113/manifest.json'
    df = load_data(manifest_path)

    # 提取特征
    feature_cols = [c for c in df.columns if c not in ['time', 'event', 'patient_id']]
    X = df[feature_cols].values

    # 分析数据复杂度
    print("\n【1. 数据复杂度分析】")
    data_complexity = analyze_data_complexity(X)

    print(f"  样本数: {data_complexity['n_samples']}")
    print(f"  特征数: {data_complexity['n_features']}")
    print(f"  样本/特征比: {data_complexity['sample_feature_ratio']:.1f}")
    print(f"  95%方差需要的成分数: {data_complexity['n_components_95']}")
    print(f"  99%方差需要的成分数: {data_complexity['n_components_99']}")
    print(f"  有效维度(intrinsic dim): {data_complexity['intrinsic_dimensionality']:.2f}")
    print(f"  数据复杂度得分: {data_complexity['complexity_score']:.2f}")

    # 定义候选模型
    candidate_models = [
        {
            'name': 'SimpleCNN',
            'type': 'cnn',
            'input_dim': (8, 4096),
            'conv_channels': [32, 64],
            'n_intervals': 20
        },
        {
            'name': 'CNN-Transformer (当前)',
            'type': 'transformer',
            'input_dim': (8, 4096),
            'conv_channels': [64, 128, 256],
            'd_model': 128,
            'n_heads': 4,
            'n_transformer_layers': 2,
            'n_intervals': 20
        },
        {
            'name': 'ResNet1d (论文)',
            'type': 'resnet',
            'input_dim': (8, 4096),
            'blocks_dim': [(64, 1024), (128, 256), (196, 64), (256, 16)],
            'n_intervals': 20
        },
        {
            'name': 'LSTM',
            'type': 'lstm',
            'input_dim': (8, 4096),
            'hidden_size': 128,
            'n_lstm_layers': 2,
            'n_intervals': 20
        },
        {
            'name': 'LightCNN',
            'type': 'cnn',
            'input_dim': (8, 4096),
            'conv_channels': [32, 64, 128],
            'n_intervals': 20
        }
    ]

    # 匹配模型
    print("\n【2. 模型复杂度匹配】")
    results = match_model_to_data(data_complexity, candidate_models)

    print(f"\n{'模型':<20} {'参数量':<12} {'层数':<6} {'复杂度':<8} {'得分':<8} {'推荐'}")
    print("-" * 70)
    for r in results:
        recommend_mark = "✓" if r['recommended'] else "✗"
        print(f"{r['model_name']:<20} {r['n_params']:<12,} {r['n_layers']:<6} "
              f"{r['model_complexity_score']:<8.2f} {r['final_score']:<8.2f} {recommend_mark}")

    # 保存结果
    output_dir = Path('outputs/model_complexity_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'data_complexity.json', 'w') as f:
        json.dump(data_complexity, f, indent=2)

    with open(output_dir / 'model_matching.json', 'w') as f:
        json.dump([{k: bool(v) if isinstance(v, np.bool_) else v for k, v in r.items()} for r in results], f, indent=2)

    print(f"\n结果已保存到: {output_dir}")

    # 给出建议
    print("\n【3. 建议】")
    best_model = results[0]
    print(f"  最匹配模型: {best_model['model_name']}")
    print(f"  参数量: {best_model['n_params']:,}")
    print(f"  复杂度得分: {best_model['final_score']:.2f}")

    if best_model['final_score'] < 3.0:
        print(f"  ✓ 该模型复杂度与数据匹配良好")
    else:
        print(f"  ✗ 所有候选模型都过于复杂，建议:")
        print(f"    - 使用传统机器学习(XGBoost/LightGBM)")
        print(f"    - 或设计更轻量的深度学习模型")
        print(f"    - 或获取更多数据")
