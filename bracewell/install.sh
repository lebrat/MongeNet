# load bracewell modules
conda deactivate
module load python/3.7.2
module load cuda/10.1.168
module load cudnn/v7.6.4-cuda101

# create environment and redirect cache to local folder
virtualenv --python=python3.7 ./mongenet_venv/
source ./mongenet_venv/bin/activate
mkdir ./pip_cache/
export PIP_CACHE_DIR=/scratch1/fon022/MongeNet/pip_cache/

# install python libs
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install hydra-core --upgrade
pip install trimesh 
pip install matplotlib
pip install tensorboard

## install GeomLoss
module load gcc/8.3.0
module load cmake/3.15.5 
pip install pykeops
pip install geomloss
