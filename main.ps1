$scriptPath = ".\generate_data.ps1"

$params = @(".\era_gpu.exe", ".\era_cpu.exe", ".\sunda_gpu.exe", ".\sunda_cpu.exe")

foreach ($param in $params) {
    & $scriptPath $param
}