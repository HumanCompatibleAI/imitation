Param(
   $venv
)
$ErrorActionPreference = "Stop"  # exit immediately on any error

If ($venv -eq $null) {
   $venv = "venv"
}

virtualenv -p python3.9 $venv
& $venv\Scripts\activate
pip install ".[docs,parallel,test]"
