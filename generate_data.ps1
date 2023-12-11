$exe_name = $args[0]

# array of first parameters
$params = @("1000000", "5000000", "9000000", "10000000", "50000000", "90000000")

# loop through the array
foreach ($param in $params) {
    # iterate 10 times
    foreach ($i in 1..10) {
        # run the exe with the parameter
        & $exe_name $param
        # delay for 1 second
        Start-Sleep -Seconds 0.5
    }
}