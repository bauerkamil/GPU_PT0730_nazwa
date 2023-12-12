$scriptPath = ".\generate_data.ps1"

$params = @(".\pre_era_gpu.exe")

foreach ($param in $params) {
    & $scriptPath $param
}