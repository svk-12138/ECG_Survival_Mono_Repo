<#
.SYNOPSIS
卒中论文训练启动脚本（Windows PowerShell）。

.DESCRIPTION
1. 先复制 configs\train_stroke_thesis.env.example，生成 configs\train_stroke_thesis.env。
2. 只修改 train_stroke_thesis.env，不要再改本脚本。
3. 修改完成后，优先运行 scripts\train_stroke_thesis.bat。
3. 如果习惯 PowerShell，也可运行：
   powershell -ExecutionPolicy Bypass -File scripts\train_stroke_thesis.ps1

设计目标：
- Win11 用户优先
- 不需要手动拼接很长的命令
- 常用实验切换尽量只改 1-2 个参数
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# 强制控制台使用 UTF-8，尽量避免 Win11 下中文输出乱码。
$Utf8Encoding = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $Utf8Encoding
[Console]::OutputEncoding = $Utf8Encoding
$OutputEncoding = $Utf8Encoding
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$Root = Split-Path -Parent $PSScriptRoot

function Resolve-PythonCommand {
    if ($env:PYTHON_BIN) {
        return @{
            Command = $env:PYTHON_BIN
            PrefixArgs = @()
        }
    }

    $BundledCandidates = @(
        (Join-Path $Root '..\..\runtime\env\python.exe'),
        (Join-Path $Root '..\runtime\env\python.exe'),
        (Join-Path $Root 'runtime\env\python.exe')
    )
    foreach ($candidate in $BundledCandidates) {
        try {
            $resolved = [System.IO.Path]::GetFullPath($candidate)
        }
        catch {
            continue
        }
        if (Test-Path -LiteralPath $resolved) {
            return @{
                Command = $resolved
                PrefixArgs = @()
            }
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return @{
            Command = $pythonCmd.Source
            PrefixArgs = @()
        }
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        return @{
            Command = $pyLauncher.Source
            PrefixArgs = @("-3")
        }
    }

    throw "[error] 未找到 Python。请先安装 Python 3.10+，或设置环境变量 PYTHON_BIN 指向 python.exe。"
}

function Test-NonEmpty {
    param([string]$Value)
    return -not [string]::IsNullOrWhiteSpace($Value)
}

function Resolve-RepoPath {
    param([string]$Value)
    if (-not (Test-NonEmpty $Value)) {
        return $Value
    }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return $Value
    }
    return Join-Path $Root $Value
}

function Test-IsPlaceholderPath {
    param([string]$Value)
    if (-not (Test-NonEmpty $Value)) {
        return $true
    }
    return $Value -like "D:\your\path\*"
}

function Get-TrainingEnvFile {
    $Configured = $env:TRAIN_STROKE_ENV_FILE
    if (Test-NonEmpty $Configured) {
        $Resolved = Resolve-RepoPath $Configured
        if (-not (Test-Path -LiteralPath $Resolved)) {
            throw "[error] 找不到 TRAIN_STROKE_ENV_FILE 指定的配置文件: $Resolved"
        }
        return $Resolved
    }

    $DefaultPath = Join-Path $Root 'configs\train_stroke_thesis.env'
    if (Test-Path -LiteralPath $DefaultPath) {
        return $DefaultPath
    }
    $ExamplePath = Join-Path $Root 'configs\train_stroke_thesis.env.example'
    throw "[error] 未找到本地训练配置文件: $DefaultPath`n[hint] 请先复制 $ExamplePath 为 $DefaultPath，然后只修改 .env 文件，不要再改 .ps1 脚本。"
}

function Unquote-EnvValue {
    param([string]$Value)
    $Text = ""
    if ($null -ne $Value) {
        $Text = $Value.Trim()
    }
    if ($Text.Length -ge 2) {
        if (($Text.StartsWith('"') -and $Text.EndsWith('"')) -or ($Text.StartsWith("'") -and $Text.EndsWith("'"))) {
            return $Text.Substring(1, $Text.Length - 2)
        }
    }
    return $Text
}

function Read-TrainingEnvFile {
    param([string]$Path)
    $Data = @{}
    $LineNumber = 0
    foreach ($RawLine in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $LineNumber += 1
        $Line = $RawLine.Trim()
        if (-not $Line -or $Line.StartsWith('#')) {
            continue
        }
        if ($Line.StartsWith('export ')) {
            $Line = $Line.Substring(7).Trim()
        }
        $EqIndex = $Line.IndexOf('=')
        if ($EqIndex -lt 1) {
            throw "[error] env 配置文件格式错误: $Path 第 $LineNumber 行应为 KEY=VALUE"
        }
        $Key = $Line.Substring(0, $EqIndex).Trim()
        $Value = $Line.Substring($EqIndex + 1)
        $Data[$Key] = Unquote-EnvValue $Value
    }
    return $Data
}

function Convert-EnvBool {
    param(
        [string]$Value,
        [string]$Key
    )
    $Normalized = ""
    if ($null -ne $Value) {
        $Normalized = $Value.Trim().ToLowerInvariant()
    }
    switch ($Normalized) {
        { $_ -in @('1', 'true', 'yes', 'y', 'on') } { return $true }
        { $_ -in @('0', 'false', 'no', 'n', 'off') } { return $false }
        default { throw "[error] env 配置 $Key 只能写 true/false（当前值: $Value）" }
    }
}

function Get-AutoTrainingInputs {
    $Candidates = @(
        (Join-Path $Root 'processed\training_inputs.json'),
        (Join-Path $Root '..\processed\training_inputs.json'),
        (Join-Path $Root '..\..\processed\training_inputs.json')
    )
    foreach ($candidate in $Candidates) {
        try {
            $resolved = [System.IO.Path]::GetFullPath($candidate)
        }
        catch {
            continue
        }
        if (-not (Test-Path -LiteralPath $resolved)) {
            continue
        }
        try {
            $data = Get-Content -LiteralPath $resolved -Raw -Encoding UTF8 | ConvertFrom-Json
            return @{
                Path = $resolved
                Data = $data
            }
        }
        catch {
            throw "[error] 无法读取自动训练配置: $resolved`n$($_.Exception.Message)"
        }
    }
    return $null
}

# ==================== 默认参数 ====================
$Manifest = "D:\your\path\stroke_manifest.json"
$XmlDir = "D:\your\path\xml_dir"
$CsvDir = ""
$TaskMode = "prediction"

# 模型预设（优先级高于 ModelType）：
# tcn_light       = TCN轻量版（~25k参数，适合1200样本）
# resnet_small    = ResNet1d小版（~12万参数，适合1万样本）★ 医生端推荐
# resnet_standard = ResNet1d标准版（~294万参数，适合10万+样本）
# cnn_transformer = CNN+Transformer（~69万参数，实验性）
$ModelName = ""

# 若不使用预设，直接指定模型架构（ModelName 留空时生效）
$ModelType = "resnet"

$LeadMode = "12lead"

# 固定划分文件：首次训练自动生成，后续自动复用，确保每次数据集组成完全一致
# 留空则每次随机划分（不推荐）
$SplitFile = "outputs/stroke_survival_thesis/dataset_split.json"
# ==================================================

# ==================== 常用参数 ====================
# 时间设置，单位要和 manifest 里的 time 一致
$NIntervals = 20
$MaxTime = 1825.0

# 风险时间点：
# - 写 null 表示整个随访窗口风险
# - 写 365.0 / 1095.0 / 1825.0 表示 1年/3年/5年风险
$PredictionHorizon = "null"

# ECG 预处理：
# - ApplyFilters=$true 时，会做带通滤波 + 工频陷波
# - 这部分更贴近论文中的常见 ECG 预处理思路
$WaveformType = "Rhythm"
$ResampleHz = 400.0
$ApplyFilters = $true
$BandpassLowHz = 0.5
$BandpassHighHz = 100.0
$NotchHz = 60.0
$NotchQ = 30.0
$TargetLen = 4096

# 训练参数
$Batch = 32
$Epochs = 80
$LR = 0.0005
$Dropout = 0.5
$WeightDecay = 0.0001
$NumWorkers = 0

# 留出法划分：
# - 仅在 CVFolds=1 时生效
# - 默认固定为 0.8 / 0.2 / 0.0，即训练/验证，无测试集
$TrainRatio = 0.8
$ValRatio = 0.2
$TestRatio = 0.0

# 是否启用交叉验证：
# - 1 表示使用上面的 train/val/test 比例
# - 大于 1 表示启用 K 折交叉验证，此时比例参数会被忽略
$CVFolds = 1
$CVSeed = 42
$TrainSeed = 42

# 早停与评估
$EvalThreshold = 0.5
$EarlyStopMetric = "auto"
$EarlyStopPatience = 15
$EarlyStopMinDelta = 0.0001
$PosWeightMult = 2.0

# 输出目录：
# - 可以写相对路径，脚本会自动保存到仓库目录下
$LogDir = "outputs\stroke_survival_thesis"

# 设备设置
$Device = ""
$UseDataParallel = $false
$DeviceIds = ""

# 如已有 best_params.json，可打开这两项
$UseBestParams = $false
$BestParams = "outputs\stroke_survival_thesis\best_params.json"
# ==================================================

$TrainingEnvFile = Get-TrainingEnvFile
$EnvConfig = Read-TrainingEnvFile $TrainingEnvFile
if ($EnvConfig.ContainsKey('PYTHON_BIN')) { $env:PYTHON_BIN = [string]$EnvConfig['PYTHON_BIN'] }
if ($EnvConfig.ContainsKey('MANIFEST')) { $Manifest = [string]$EnvConfig['MANIFEST'] }
if ($EnvConfig.ContainsKey('XML_DIR')) { $XmlDir = [string]$EnvConfig['XML_DIR'] }
if ($EnvConfig.ContainsKey('CSV_DIR')) { $CsvDir = [string]$EnvConfig['CSV_DIR'] }
if ($EnvConfig.ContainsKey('TASK_MODE')) { $TaskMode = [string]$EnvConfig['TASK_MODE'] }
if ($EnvConfig.ContainsKey('MODEL_NAME')) { $ModelName = [string]$EnvConfig['MODEL_NAME'] }
if ($EnvConfig.ContainsKey('MODEL_TYPE')) { $ModelType = [string]$EnvConfig['MODEL_TYPE'] }
if ($EnvConfig.ContainsKey('LEAD_MODE')) { $LeadMode = [string]$EnvConfig['LEAD_MODE'] }
if ($EnvConfig.ContainsKey('SPLIT_FILE')) { $SplitFile = [string]$EnvConfig['SPLIT_FILE'] }
if ($EnvConfig.ContainsKey('N_INTERVALS')) { $NIntervals = [string]$EnvConfig['N_INTERVALS'] }
if ($EnvConfig.ContainsKey('MAX_TIME')) { $MaxTime = [string]$EnvConfig['MAX_TIME'] }
if ($EnvConfig.ContainsKey('PREDICTION_HORIZON')) { $PredictionHorizon = [string]$EnvConfig['PREDICTION_HORIZON'] }
if ($EnvConfig.ContainsKey('WAVEFORM_TYPE')) { $WaveformType = [string]$EnvConfig['WAVEFORM_TYPE'] }
if ($EnvConfig.ContainsKey('RESAMPLE_HZ')) { $ResampleHz = [string]$EnvConfig['RESAMPLE_HZ'] }
if ($EnvConfig.ContainsKey('APPLY_FILTERS')) { $ApplyFilters = Convert-EnvBool ([string]$EnvConfig['APPLY_FILTERS']) 'APPLY_FILTERS' }
if ($EnvConfig.ContainsKey('BANDPASS_LOW_HZ')) { $BandpassLowHz = [string]$EnvConfig['BANDPASS_LOW_HZ'] }
if ($EnvConfig.ContainsKey('BANDPASS_HIGH_HZ')) { $BandpassHighHz = [string]$EnvConfig['BANDPASS_HIGH_HZ'] }
if ($EnvConfig.ContainsKey('NOTCH_HZ')) { $NotchHz = [string]$EnvConfig['NOTCH_HZ'] }
if ($EnvConfig.ContainsKey('NOTCH_Q')) { $NotchQ = [string]$EnvConfig['NOTCH_Q'] }
if ($EnvConfig.ContainsKey('TARGET_LEN')) { $TargetLen = [string]$EnvConfig['TARGET_LEN'] }
if ($EnvConfig.ContainsKey('BATCH')) { $Batch = [string]$EnvConfig['BATCH'] }
if ($EnvConfig.ContainsKey('EPOCHS')) { $Epochs = [string]$EnvConfig['EPOCHS'] }
if ($EnvConfig.ContainsKey('LR')) { $LR = [string]$EnvConfig['LR'] }
if ($EnvConfig.ContainsKey('DROPOUT')) { $Dropout = [string]$EnvConfig['DROPOUT'] }
if ($EnvConfig.ContainsKey('WEIGHT_DECAY')) { $WeightDecay = [string]$EnvConfig['WEIGHT_DECAY'] }
if ($EnvConfig.ContainsKey('NUM_WORKERS')) { $NumWorkers = [string]$EnvConfig['NUM_WORKERS'] }
if ($EnvConfig.ContainsKey('TRAIN_RATIO')) { $TrainRatio = [string]$EnvConfig['TRAIN_RATIO'] }
if ($EnvConfig.ContainsKey('VAL_RATIO')) { $ValRatio = [string]$EnvConfig['VAL_RATIO'] }
if ($EnvConfig.ContainsKey('TEST_RATIO')) { $TestRatio = [string]$EnvConfig['TEST_RATIO'] }
if ($EnvConfig.ContainsKey('CV_FOLDS')) { $CVFolds = [string]$EnvConfig['CV_FOLDS'] }
if ($EnvConfig.ContainsKey('CV_SEED')) { $CVSeed = [string]$EnvConfig['CV_SEED'] }
if ($EnvConfig.ContainsKey('TRAIN_SEED')) { $TrainSeed = [string]$EnvConfig['TRAIN_SEED'] }
if ($EnvConfig.ContainsKey('EVAL_THRESHOLD')) { $EvalThreshold = [string]$EnvConfig['EVAL_THRESHOLD'] }
if ($EnvConfig.ContainsKey('EARLY_STOP_METRIC')) { $EarlyStopMetric = [string]$EnvConfig['EARLY_STOP_METRIC'] }
if ($EnvConfig.ContainsKey('EARLY_STOP_PATIENCE')) { $EarlyStopPatience = [string]$EnvConfig['EARLY_STOP_PATIENCE'] }
if ($EnvConfig.ContainsKey('EARLY_STOP_MIN_DELTA')) { $EarlyStopMinDelta = [string]$EnvConfig['EARLY_STOP_MIN_DELTA'] }
if ($EnvConfig.ContainsKey('POS_WEIGHT_MULT')) { $PosWeightMult = [string]$EnvConfig['POS_WEIGHT_MULT'] }
if ($EnvConfig.ContainsKey('LOG_DIR')) { $LogDir = [string]$EnvConfig['LOG_DIR'] }
if ($EnvConfig.ContainsKey('DEVICE')) { $Device = [string]$EnvConfig['DEVICE'] }
if ($EnvConfig.ContainsKey('USE_DATA_PARALLEL')) { $UseDataParallel = Convert-EnvBool ([string]$EnvConfig['USE_DATA_PARALLEL']) 'USE_DATA_PARALLEL' }
if ($EnvConfig.ContainsKey('DEVICE_IDS')) { $DeviceIds = [string]$EnvConfig['DEVICE_IDS'] }
if ($EnvConfig.ContainsKey('USE_BEST_PARAMS')) { $UseBestParams = Convert-EnvBool ([string]$EnvConfig['USE_BEST_PARAMS']) 'USE_BEST_PARAMS' }
if ($EnvConfig.ContainsKey('BEST_PARAMS')) { $BestParams = [string]$EnvConfig['BEST_PARAMS'] }
Write-Host "[env] 已加载本地配置: $TrainingEnvFile"

$PythonSpec = Resolve-PythonCommand
$PythonBin = $PythonSpec.Command
$PythonPrefixArgs = $PythonSpec.PrefixArgs
$env:PYTHONHASHSEED = $TrainSeed.ToString()

$AutoTrainingInputs = Get-AutoTrainingInputs
$AutoRecommendedLeadMode = ""
if ($AutoTrainingInputs) {
    $AutoData = $AutoTrainingInputs.Data
    if (Test-IsPlaceholderPath $Manifest) {
        $Manifest = [string]$AutoData.manifest
    }
    if (Test-IsPlaceholderPath $XmlDir) {
        $XmlDir = [string]$AutoData.xml_dir
    }
    if (Test-IsPlaceholderPath $CsvDir) {
        $CsvDir = [string]$AutoData.csv_dir
    }
    if (Test-NonEmpty ([string]$AutoData.recommended_lead_mode)) {
        $AutoRecommendedLeadMode = [string]$AutoData.recommended_lead_mode
    }
    if ($LeadMode -eq "auto") {
        if (-not (Test-NonEmpty $AutoRecommendedLeadMode)) {
            throw "[error] LeadMode=auto，但自动训练配置中没有 recommended_lead_mode"
        }
        $LeadMode = $AutoRecommendedLeadMode
    }
    if ($WaveformType -eq "Rhythm" -and (Test-NonEmpty ([string]$AutoData.waveform_type))) {
        $WaveformType = [string]$AutoData.waveform_type
    }
    Write-Host "[auto] 已加载训练输入配置: $($AutoTrainingInputs.Path)"
    if ((Test-NonEmpty $AutoRecommendedLeadMode) -and ($LeadMode -ne $AutoRecommendedLeadMode)) {
        Write-Host "[warn] 自动推荐导联模式为 $AutoRecommendedLeadMode，但当前脚本将按手动设置的 $LeadMode 训练。"
    }
}

if (-not (Test-NonEmpty $Manifest)) {
    throw "[error] Manifest 不能为空"
}

if ((Test-NonEmpty $XmlDir) -and (Test-NonEmpty $CsvDir)) {
    throw "[error] XmlDir 和 CsvDir 只能填一个"
}

if ((-not (Test-NonEmpty $XmlDir)) -and (-not (Test-NonEmpty $CsvDir))) {
    throw "[error] XmlDir 和 CsvDir 至少要填一个"
}

if ($TaskMode -notin @("prediction", "classification")) {
    throw "[error] TaskMode 只能是 prediction 或 classification"
}

if ($ModelType -notin @("resnet", "tcn_light", "cnn_transformer", "cnn_gru", "cnn_transformer_small")) {
    throw "[error] ModelType 只能是 resnet、tcn_light、cnn_transformer、cnn_gru 或 cnn_transformer_small"
}

if ($LeadMode -eq "auto") {
    throw "[error] LeadMode=auto 需要先生成 training_inputs.json，或手动改成 8lead / 12lead"
}

if ($LeadMode -notin @("8lead", "12lead")) {
    throw "[error] LeadMode 只能是 8lead、12lead 或 auto"
}

$ManifestResolved = Resolve-RepoPath $Manifest
$XmlDirResolved = if (Test-NonEmpty $XmlDir) { Resolve-RepoPath $XmlDir } else { "" }
$CsvDirResolved = if (Test-NonEmpty $CsvDir) { Resolve-RepoPath $CsvDir } else { "" }

if (-not (Test-Path -LiteralPath $ManifestResolved)) {
    throw "[error] 找不到 Manifest: $ManifestResolved`n[hint] 可以写绝对路径，也可以写相对仓库根目录的路径。"
}

if ((Test-NonEmpty $XmlDirResolved) -and (-not (Test-Path -LiteralPath $XmlDirResolved))) {
    throw "[error] 找不到 XmlDir: $XmlDirResolved`n[hint] 可以写绝对路径，也可以写相对仓库根目录的路径。"
}

if ((Test-NonEmpty $CsvDirResolved) -and (-not (Test-Path -LiteralPath $CsvDirResolved))) {
    throw "[error] 找不到 CsvDir: $CsvDirResolved`n[hint] 可以写绝对路径，也可以写相对仓库根目录的路径。"
}

$LogDirResolved = Resolve-RepoPath $LogDir
$BestParamsResolved = Resolve-RepoPath $BestParams
$SplitFileResolved = if (Test-NonEmpty $SplitFile) { Resolve-RepoPath $SplitFile } else { "" }
New-Item -ItemType Directory -Force -Path $LogDirResolved | Out-Null

$CommandArgs = @()
$CommandArgs += Join-Path $Root 'scripts/run_survival_training.py'
$CommandArgs += @('--manifest', $ManifestResolved)
$CommandArgs += @('--task-mode', $TaskMode)

# 优先使用 MODEL_NAME 预设，若为空则使用 MODEL_TYPE
if (Test-NonEmpty $ModelName) {
    $CommandArgs += @('--model-name', $ModelName)
} else {
    $CommandArgs += @('--model-type', $ModelType)
}

$CommandArgs += @('--lead-mode', $LeadMode)
$CommandArgs += @('--n-intervals', $NIntervals.ToString())
$CommandArgs += @('--max-time', $MaxTime.ToString())
$CommandArgs += @('--target-len', $TargetLen.ToString())
$CommandArgs += @('--waveform-type', $WaveformType)
$CommandArgs += @('--resample-hz', $ResampleHz.ToString())
$CommandArgs += @('--bandpass-low-hz', $BandpassLowHz.ToString())
$CommandArgs += @('--bandpass-high-hz', $BandpassHighHz.ToString())
$CommandArgs += @('--notch-hz', $NotchHz.ToString())
$CommandArgs += @('--notch-q', $NotchQ.ToString())
$CommandArgs += @('--batch', $Batch.ToString())
$CommandArgs += @('--epochs', $Epochs.ToString())
$CommandArgs += @('--lr', $LR.ToString())
$CommandArgs += @('--dropout', $Dropout.ToString())
$CommandArgs += @('--weight-decay', $WeightDecay.ToString())
$CommandArgs += @('--num-workers', $NumWorkers.ToString())
$CommandArgs += @('--cv-folds', $CVFolds.ToString())
$CommandArgs += @('--cv-seed', $CVSeed.ToString())
$CommandArgs += @('--train-seed', $TrainSeed.ToString())
$CommandArgs += @('--train-ratio', $TrainRatio.ToString())
$CommandArgs += @('--val-ratio', $ValRatio.ToString())
$CommandArgs += @('--test-ratio', $TestRatio.ToString())
$CommandArgs += @('--eval-threshold', $EvalThreshold.ToString())
$CommandArgs += @('--early-stop-metric', $EarlyStopMetric)
$CommandArgs += @('--early-stop-patience', $EarlyStopPatience.ToString())
$CommandArgs += @('--early-stop-min-delta', $EarlyStopMinDelta.ToString())
$CommandArgs += @('--pos-weight-mult', $PosWeightMult.ToString())
$CommandArgs += @('--log-dir', $LogDirResolved)

if (Test-NonEmpty $XmlDirResolved) {
    $CommandArgs += @('--xml-dir', $XmlDirResolved)
}

if (Test-NonEmpty $CsvDirResolved) {
    $CommandArgs += @('--csv-dir', $CsvDirResolved)
}

if ($PredictionHorizon -ne 'null') {
    $CommandArgs += @('--prediction-horizon', $PredictionHorizon)
}

if ($ApplyFilters) {
    $CommandArgs += '--apply-filters'
}
else {
    $CommandArgs += '--no-apply-filters'
}

if (Test-NonEmpty $Device) {
    $CommandArgs += @('--device', $Device)
}

if ($UseDataParallel) {
    $CommandArgs += '--use-data-parallel'
}

if (Test-NonEmpty $DeviceIds) {
    $CommandArgs += @('--device-ids', $DeviceIds)
}

if ($UseBestParams) {
    $CommandArgs += @('--use-best-params', '--best-params', $BestParamsResolved)
}

if (Test-NonEmpty $SplitFileResolved) {
    $CommandArgs += @('--split-file', $SplitFileResolved)
}

$ModelDisplay = if (Test-NonEmpty $ModelName) { "preset:$ModelName" } else { $ModelType }
Write-Host '[info] 即将启动训练，关键参数如下：'
Write-Host "  task_mode=$TaskMode"
Write-Host "  model=$ModelDisplay"
Write-Host "  lead_mode=$LeadMode"
Write-Host "  manifest=$ManifestResolved"
Write-Host "  xml_dir=$XmlDirResolved"
Write-Host "  csv_dir=$CsvDirResolved"
Write-Host "  log_dir=$LogDirResolved"
Write-Host "  prediction_horizon=$PredictionHorizon"
Write-Host "  split_ratio=train:$TrainRatio val:$ValRatio test:$TestRatio"
Write-Host "  split_file=$SplitFileResolved"
Write-Host "  cv_folds=$CVFolds"
Write-Host "  train_seed=$TrainSeed"
if ($AutoTrainingInputs) {
    Write-Host "  auto_training_inputs=$($AutoTrainingInputs.Path)"
    if (Test-NonEmpty $AutoRecommendedLeadMode) {
        Write-Host "  auto_recommended_lead_mode=$AutoRecommendedLeadMode"
    }
}
if ($TrainingEnvFile) {
    Write-Host "  env_file=$TrainingEnvFile"
}
Write-Host "[cmd] $PythonBin $($PythonPrefixArgs -join ' ') $($CommandArgs -join ' ')"

$ErrorActionPreference = "Continue"
& $PythonBin @PythonPrefixArgs @CommandArgs
exit $LASTEXITCODE
