:: 1. 克隆代码并创建环境
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
conda create -n gaussian_splatting python=3.10
conda activate gaussian_splatting

:: 2. 安装对应版本的 PyTorch (以 CUDA 12.4 为例)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

:: 3. 打开 "x64 Native Tools Command Prompt for VS 2022" 终端，执行以下命令：
D:\Anaconda3\Scripts\activate.bat gaussian_splatting//yourpath
cd /d D:\temp\3DGS\gaussian_splatting\Resources//yourpath
SET DISTUTILS_USE_SDK=1
pip install --no-build-isolation submodules\diff-gaussian-rasterization
pip install --no-build-isolation submodules\simple-knn

:: 4. 安装其他辅助库
pip install plyfile tqdm opencv-python

===============================================================

python train.py -s D:\temp\3DGS\gaussian_splatting\Data\Bottle

python render.py -m ./output/0f2871a8-7

cd D:\temp\3DGS\gaussian_splatting\Resources\output\0f2871a8-7\train\ours_30000\renders

ffmpeg -framerate 30 -i %05d.png -c:v libx264 -pix_fmt yuv420p output.mp4