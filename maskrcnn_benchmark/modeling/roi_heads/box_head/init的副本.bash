cd ..
conda create -n maskrcnnpython=3.7
conda activate maskrcnn
pip install ninja yacs cython matplotlib tqdm opencv-python
pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
export INSTALL_DIR=$PWD
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0

python setup.py install --cuda_ext --cpp_ext
cd $INSTALL_DIR
git clone https://github.com/Zyriix/iMaterialist.git
cd iMaterialist
python setup.py build develop