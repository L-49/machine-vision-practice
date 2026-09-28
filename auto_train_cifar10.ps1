# auto_train_cifar10.ps1 - CIFAR-100 训练结束后自动启动 CIFAR-10 训练并实时显示进度
# 用法: powershell -ExecutionPolicy Bypass -File auto_train_cifar10.ps1
# 可多开（内置防重复启动锁），Ctrl+C 仅退出监视、不影响后台训练
$ErrorActionPreference = "Stop"
$py   = "D:\appdownload\python\python.exe"
$proj = "D:\新d盘\大学牲\23030127李思佳\大四上\机器视觉课程实践\机器视觉实践\pytorch-cifar100"
$dataTar    = Join-Path $proj "data\cifar-10-python.tar.gz"
$targetSize = 170498071
$md5Expect  = "c58f30108f718f92721af3b95e74349a"
$logDir     = Join-Path $proj "logs"
$lockFile   = Join-Path $logDir ".cifar10_training.lock"

function Get-Cifar100Proc {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train_cpu.py*" -and $_.CommandLine -notlike "*cifar10*" }
}
function Get-Cifar10Proc {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train_cifar10_cpu.py*" }
}

Write-Host "=============================================="
Write-Host " CIFAR-10 自动训练监控已启动"
Write-Host "=============================================="

# ===== 阶段1: 等待 CIFAR-100 训练结束 =====
if (Get-Cifar100Proc) {
    Write-Host "[监控] 检测到 CIFAR-100 训练进行中，等待其结束(每30秒检查)..."
    while (Get-Cifar100Proc) { Start-Sleep -Seconds 30 }
}
Write-Host "[监控] CIFAR-100 训练已结束。"

# ===== 阶段2: 等待 CIFAR-10 数据下载完成并校验 =====
while ($true) {
    $size = 0
    if (Test-Path $dataTar) { $size = (Get-Item $dataTar).Length }
    Write-Host ("[监控] 数据下载进度: {0:N0} / {1:N0} 字节 ({2:P1})" -f $size, $targetSize, ($size / $targetSize))
    if ($size -ge $targetSize) { break }
    Start-Sleep -Seconds 60
}
Write-Host "[监控] 下载完成，校验 MD5..."
$hash = (Get-FileHash $dataTar -Algorithm MD5).Hash.ToLower()
if ($hash -ne $md5Expect) { Write-Host "[错误] MD5 校验失败: $hash，请重新下载数据集"; exit 1 }
Write-Host "[监控] MD5 校验通过。"

# ===== 阶段3: 自动启动 CIFAR-10 训练（防双开）=====
$already = $false
if (Get-Cifar10Proc) {
    Write-Host "[监控] CIFAR-10 训练已在运行，直接进入监视。"
    $already = $true
}
if (Test-Path $lockFile) {
    $age = (Get-Date) - (Get-Item $lockFile).LastWriteTime
    if ($age.TotalHours -lt 24) { Write-Host "[监控] 发现启动锁(24h内)，跳过启动。"; $already = $true }
}
$log = $null
if (-not $already) {
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $log   = Join-Path $logDir "cifar10_resnet18_$stamp.log"
    New-Item -ItemType File -Path $lockFile -Force | Out-Null
    Write-Host "[监控] 启动 CIFAR-10 训练 (resnet18, 60 epoch)..."
    Write-Host "[监控] 日志文件: $log"
    $err = "$log.err"
    Start-Process -FilePath $py `
        -ArgumentList "-X","utf8","-u","train_cifar10_cpu.py","-net","resnet18","-epochs","60","-milestones","30,45,55" `
        -WorkingDirectory $proj -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden
    Start-Sleep -Seconds 10
} else {
    # 已在运行：找最新日志文件继续监视
    $latest = Get-ChildItem $logDir -Filter "cifar10_resnet18_*.log" -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) { $log = $latest.FullName }
}

# ===== 阶段4: 实时显示训练进度 =====
Write-Host "[监控] 开始实时显示训练进度 (Ctrl+C 退出监视，不影响训练)..."
$pos = 0
while ($true) {
    if ($log -and (Test-Path $log)) {
        $lines = @(Get-Content $log -Encoding UTF8 -ErrorAction SilentlyContinue)
        if ($lines.Count -gt $pos) {
            $lines[$pos..($lines.Count - 1)] | ForEach-Object { Write-Host $_ }
            $pos = $lines.Count
        }
    }
    if (-not (Get-Cifar10Proc)) { break }
    Start-Sleep -Seconds 20
}
Write-Host "=============================================="
Write-Host "[监控] CIFAR-10 训练已结束。"
Write-Host ("[监控] 最终指标见: {0}\checkpoint\cifar10_resnet18\<时间戳>\metrics.txt" -f $proj)
