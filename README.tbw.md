## Environment

```shell
conda create -n scribblekitti python=3.8

conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install -r requirements.txt

# Download spconv v1.2.1 with two bugs fixed
wget https://cloud.tsinghua.edu.cn/f/f1a0860337224c7f8585/?dl=1 -O spconv.tar.gz && tar xzvf spconv.tar.gz
cd spconv
sudo apt-get install libboost-all-dev
CUDACXX=/usr/local/cuda/bin/nvcc python setup.py bdist_wheel
pip install dist/spconv-1.2.1-cp38-cp38-linux_x86_64.whl
```

## How-to

### Pretrain

```shell
python pretrain.py --config_path config/pretrain/bt_pls.yaml --dataset_config_path config/pretrain/semantickitti.yaml
```

* `pretrain/bt.yaml`: Barlow Twins SSRL with Cylinder3D
* `pretrain/bt_pls.yaml`: Barlow Twins SSRL with Cylinder3D with pyramid local semantic context 
* `?`: MEC SSRL with Cylinder3D
* `?`: MEC SSRL with Cylinder3D with pyramid local semantic context

### Train

```shell
python train.py --config_path config/pretrain/bt_pls.yaml --dataset_config_path config/pretrain/semantickitti.yaml
```
* `?`: plain Cylinder3D tuning
* `?`: Barlow Twins pretrained Cylinder3D tuning
* `?`: MEC pretrained Cylinder3D tuning

### Note

`label_directory`