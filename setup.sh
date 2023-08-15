# Pop!_OS 22.04 LTS
# Python 3.10.12

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip

# Install python packages
pip install numpy matplotlib setuptools

# Install PyTorch 2.0.0 + CUDA 11.7
if ["$1" -eq "cuda"] then
    pip install \
    torch==2.0.0+cu117 \
    torchvision==0.15.1+cu117 \
    torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu117
else
    pip install \
    torch==2.0.0+cpu \
    torchvision==0.15.1+cpu \
    torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cpu
fi

# Install GUI package
pip install -U wxPython

pip install opencv-python opencv-contrib-python


