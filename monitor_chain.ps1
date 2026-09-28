# 串联训练监督器：监控 CIFAR-100 MobileNet -> CIFAR-10 MobileNet 的衔接
# 每 30 秒采样一次，阶段切换/异常写入状态日志
$ErrorActionPreference = 'Continue'
$root = 'D:\新d盘\大学牲\23030127李思佳\大四上\机器视觉课程实践\机器视觉实践\pytorch-cifar100'
$logs = Join-Path $root 'logs'
$stateLog = Join-Path $logs 'chain_monitor.log'
$chainPid = 37212   # auto_chain_mobilenet.ps1
$stage1Pid = 24620  # train_cpu.py (CIFAR-100)

function Write-State($text) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $text
    Add-Path -Path $stateLog -Value $line
    Write-Host $line
}
function Add-Path($Path,$Value) { Add-Content -Path $Path -Value $Value -Encoding UTF8 }

function Find-Stage2Proc {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*train_cifar10_cpu.py*' -and $_.CommandLine -like '*mobilenet*' }
}
function Get-LastEpoch($prefix) {
    $f = Get-ChildItem (Join-Path $logs ($prefix + '_*.log')) -ErrorAction SilentlyContinue |
         Sort-Object Name -Descending | Select-Object -First 1
    if (-not $f) { return 0 }
    $last = Get-Content $f.FullName -Encoding UTF8 -Tail 30 |
            Where-Object { $_ -match 'epoch\s*(\d+)' } | Select-Object -Last 1
    if ($last -match 'epoch\s*(\d+)') { return [int]$Matches[1] }
    return 0
}

Write-State "监督器启动 | 串联脚本 PID=$chainPid | 阶段1 PID=$stage1Pid"
$phase = 'STAGE1'
$stage1EndTime = $null
$lastReport = [datetime]::MinValue

while ($true) {
    Start-Sleep -Seconds 30
    $chainAlive = Get-Process -Id $chainPid -ErrorAction SilentlyContinue
    $stage1Alive = Get-Process -Id $stage1Pid -ErrorAction SilentlyContinue

    # ---- 阶段 1 运行中 ----
    if ($phase -eq 'STAGE1') {
        if ($stage1Alive) {
            # 每 10 分钟汇报一次进度
            if (((Get-Date) - $lastReport).TotalMinutes -ge 10) {
                $ep = Get-LastEpoch 'cifar100_mobilenet'
                Write-State ("阶段1进行中 | epoch={0}/60 | 串联脚本存活={1}" -f $ep, [bool]$chainAlive)
                $lastReport = Get-Date
            }
            if (-not $chainAlive) {
                Write-State "警告：串联脚本已退出但阶段1仍在运行！"
            }
        } else {
            Write-State "阶段1进程已退出 (PID=$stage1Pid)"
            $stage1EndTime = Get-Date
            $ep = Get-LastEpoch 'cifar100_mobilenet'
            Write-State ("阶段1最终 epoch={0}/60" -f $ep)
            $phase = 'WAIT_STAGE2'
        }
    }

    # ---- 等待阶段 2 启动（最多 10 分钟）----
    if ($phase -eq 'WAIT_STAGE2') {
        $s2 = Find-Stage2Proc
        if ($s2) {
            Write-State ("阶段2已启动 PID={0}" -f $s2.ProcessId)
            $script:stage2Pid = $s2.ProcessId
            $phase = 'STAGE2'
            $lastReport = [datetime]::MinValue
            continue
        }
        $waitMin = ((Get-Date) - $stage1EndTime).TotalMinutes
        if ($waitMin -gt 10) {
            Write-State "严重异常：阶段1结束 10 分钟后阶段2仍未启动！需要人工介入。"
            if (-not $chainAlive) {
                Write-State "串联脚本也已退出，监督器终止。"
                break
            }
        }
    }

    # ---- 阶段 2 运行中 ----
    if ($phase -eq 'STAGE2') {
        $s2proc = Get-Process -Id $script:stage2Pid -ErrorAction SilentlyContinue
        if ($s2proc) {
            if (((Get-Date) - $lastReport).TotalMinutes -ge 10) {
                $ep = Get-LastEpoch 'cifar10_mobilenet'
                Write-State ("阶段2进行中 | epoch={0}/60 | PID={1}" -f $ep, $script:stage2Pid)
                $lastReport = Get-Date
            }
        } else {
            $ep = Get-LastEpoch 'cifar10_mobilenet'
            Write-State ("阶段2进程已退出 | 最终 epoch={0}/60" -f $ep)
            $phase = 'DONE'
        }
    }

    # ---- 完成 ----
    if ($phase -eq 'DONE') {
        Start-Sleep -Seconds 5
        $chainAlive2 = Get-Process -Id $chainPid -ErrorAction SilentlyContinue
        if (-not $chainAlive2) {
            Write-State "串联脚本已退出，全部训练完成。监督器正常终止。"
            break
        }
        if (-not (Find-Stage2Proc)) {
            Write-State "无残留训练进程，监督器终止。"
            break
        }
    }
}