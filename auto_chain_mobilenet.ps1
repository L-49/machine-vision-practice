# 串联训练（稳健版，用 Start-Process 独立进程，避免 stderr 警告经 PowerShell 管道导致中断）：
#   阶段1：CIFAR-100 MobileNet 续训 epoch 31~60（衰减点 40,50）
#   阶段2：CIFAR-10 MobileNet 全新训练 60 epoch（衰减点 30,45,55）
# stdout 直接写入 logs/ 目录，供 Flask 仪表盘实时读取；stderr 单独存 .err 文件。
$ErrorActionPreference = 'Continue'

$root = 'D:\新d盘\大学牲\23030127李思佳\大四上\机器视觉课程实践\机器视觉实践\pytorch-cifar100'
$py   = 'D:\appdownload\python\python.exe'
$logs = Join-Path $root 'logs'
Set-Location $root

# 防重复启动
$lock = Join-Path $env:TEMP 'chain_mobilenet.lock'
if (Test-Path $lock) {
    $oldPid = Get-Content $lock
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) {
        Write-Host "串联训练已在运行 (PID=$oldPid)，本次退出。"
        exit 0
    }
}
$PID | Out-File $lock -Encoding ASCII

function Run-Stage($name, $argString, $logPath) {
    Write-Host "========== $name 开始 $(Get-Date -Format 'HH:mm:ss') =========="
    Write-Host "日志: $logPath"
    $errPath = "$logPath.err"
    $p = Start-Process -FilePath $py -WorkingDirectory $root `
        -ArgumentList $argString `
        -RedirectStandardOutput $logPath -RedirectStandardError $errPath `
        -NoNewWindow -Wait -PassThru
    Write-Host "---------- $name 结束 $(Get-Date -Format 'HH:mm:ss') 退出码=$($p.ExitCode) ----------"
    return $p.ExitCode
}

try {
    # ---------------- 阶段 1：CIFAR-100 MobileNet 续训 ----------------
    $stamp1 = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log1 = Join-Path $logs "cifar100_mobilenet_$stamp1.log"
    $args1 = '-X utf8 -u train_cpu.py -net mobilenet -b 128 -epochs 60 ' +
              '-resume "checkpoint\mobilenet\2026-09-21_13-45-24\mobilenet-27-best.pth" ' +
              '-start_epoch 31 -best_acc 44.86 -milestones "40,50"'
    $code1 = Run-Stage '阶段1 CIFAR-100 MobileNet 续训 31~60' $args1 $log1
    if ($code1 -ne 0) { Write-Host "阶段1异常退出(码$code1)，仍继续阶段2。" }

    # ---------------- 阶段 2：CIFAR-10 MobileNet 全新训练 ----------------
    $stamp2 = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log2 = Join-Path $logs "cifar10_mobilenet_$stamp2.log"
    $args2 = '-X utf8 -u train_cifar10_cpu.py -net mobilenet -b 128 -epochs 60 ' +
              '-milestones "30,45,55"'
    $code2 = Run-Stage '阶段2 CIFAR-10 MobileNet 训练 60 epoch' $args2 $log2
    if ($code2 -ne 0) { Write-Host "阶段2异常退出(码$code2)。" }

    Write-Host "========== 全部串联训练完成 $(Get-Date -Format 'HH:mm:ss') =========="
}
finally {
    Remove-Item $lock -ErrorAction SilentlyContinue
}
