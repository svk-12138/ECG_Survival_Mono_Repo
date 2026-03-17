# PowerShell脚本: 上传并执行数据集重建
# 执行方式: .\upload_and_run_rebuild.ps1

$ScriptPath = "d:\WORKING\scripts\complete_dataset_rebuild.sh"
$RemoteHost = "admin123@61.167.54.113"
$RemotePath = "/home/admin123/workspace/complete_dataset_rebuild.sh"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "数据集重建脚本上传并执行" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 步骤1: 上传脚本
Write-Host "步骤1: 上传脚本到服务器..." -ForegroundColor Yellow
scp $ScriptPath ${RemoteHost}:${RemotePath}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 脚本上传成功" -ForegroundColor Green
} else {
    Write-Host "❌ 脚本上传失败" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 步骤2: 添加执行权限并运行
Write-Host "步骤2: 在服务器上执行脚本..." -ForegroundColor Yellow
Write-Host "这将需要30-60分钟，请耐心等待..." -ForegroundColor Yellow
Write-Host ""

# 使用ssh执行远程命令
ssh $RemoteHost "chmod +x $RemotePath && bash $RemotePath"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ 数据集重建完成！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "❌ 执行过程中出现错误" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
}
