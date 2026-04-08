<#
.SYNOPSIS
卒中论文训练启动脚本（Windows PowerShell）。

.DESCRIPTION
1. 医生或学生只需要修改本文件顶部参数。
2. 修改完成后，优先运行 scripts\train_stroke_thesis.bat。
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

$PythonSpec = Resolve-PythonCommand
$PythonBin = $PythonSpec.Command
$PythonPrefixArgs = $PythonSpec.PrefixArgs

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

# ==================== 必改参数 ====================
# 如果仓库附近存在 processed\training_inputs.json，下面这些路径会自动优先读取；
# 若要手动指定路径，再修改下面 3 个变量。
$Manifest = "D:\your\path\stroke_manifest.json"

# ECG 数据二选一：
# 1. 用 XML 就填 XmlDir，把 CsvDir 留空
$XmlDir = "D:\your\path\xml_dir"
$CsvDir = ""

# 2. 用 CSV 就填 CsvDir，把 XmlDir 留空
# $XmlDir = ""
# $CsvDir = "D:\your\path\csv_dir"

# 任务类型：
# prediction    = 生存预测，建议作为论文主实验
# classification = 二分类，建议作为对照实验
$TaskMode = "prediction"

# 导联类型：
# 8lead  = I, II, V1-V6
# 12lead = I, II, III, aVR, aVL, aVF, V1-V6
# auto   = 若存在 training_inputs.json，则自动使用其中推荐值
$LeadMode = "12lead"
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
New-Item -ItemType Directory -Force -Path $LogDirResolved | Out-Null

$CommandArgs = @()
$CommandArgs += Join-Path $Root 'scripts/run_survival_training.py'
$CommandArgs += @('--manifest', $ManifestResolved)
$CommandArgs += @('--task-mode', $TaskMode)
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

Write-Host '[info] 即将启动训练，关键参数如下：'
Write-Host "  task_mode=$TaskMode"
Write-Host "  lead_mode=$LeadMode"
Write-Host "  manifest=$ManifestResolved"
Write-Host "  xml_dir=$XmlDirResolved"
Write-Host "  csv_dir=$CsvDirResolved"
Write-Host "  log_dir=$LogDirResolved"
Write-Host "  prediction_horizon=$PredictionHorizon"
Write-Host "  split_ratio=train:$TrainRatio val:$ValRatio test:$TestRatio"
Write-Host "  cv_folds=$CVFolds"
if ($AutoTrainingInputs) {
    Write-Host "  auto_training_inputs=$($AutoTrainingInputs.Path)"
    if (Test-NonEmpty $AutoRecommendedLeadMode) {
        Write-Host "  auto_recommended_lead_mode=$AutoRecommendedLeadMode"
    }
}
Write-Host "[cmd] $PythonBin $($PythonPrefixArgs -join ' ') $($CommandArgs -join ' ')"

$ErrorActionPreference = "Continue"
& $PythonBin @PythonPrefixArgs @CommandArgs
exit $LASTEXITCODE
