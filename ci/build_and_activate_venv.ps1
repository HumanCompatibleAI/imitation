Param(
   $venv
)
$ErrorActionPreference = "Stop"  # exit immediately on any error

If ($venv -eq $null) {
   $venv = "venv"
}

virtualenv -p python3.8 $venv
& $venv\Scripts\activate
# Note: We need to install these versions of setuptools and wheel to allow installing gym==0.21.0 on Windows.
# See https://github.com/freqtrade/freqtrade/issues/8376
python -m pip install --upgrade pip wheel==0.38.4 setuptools==65.5.1
pip install ".[docs,parallel,test]"
