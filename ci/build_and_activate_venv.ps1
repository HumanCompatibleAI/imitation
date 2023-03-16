Param(
   $venv,
   $atari_roms
)
$ErrorActionPreference = "Stop"  # exit immediately on any error

If ($venv -eq $null) {
   $venv = "venv"
}

virtualenv -p python3.8 $venv
& $venv\Scripts\activate
# Note: We need to install setuptools==66.1.1 to allow installing gym==0.21.0.
python -m pip install --upgrade pip setuptools==66.1.1

# download roms and separately install autorom
pip install autorom
wget ${atari_roms}
base64 Roms.tar.gz.b64 --decode &> Roms.tar.gz
AutoROM --accept-license --source-file Roms.tar.gz

pip install ".[docs,parallel,test]"
