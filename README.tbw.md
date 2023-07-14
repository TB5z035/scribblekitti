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

export WANDB_DIR=/data14/chenyh2306/scribblekitti/wandb/
export WANDB_CACHE_DIR=/data14/chenyh2306/scribblekitti/wandb/.cache/
export WANDB_CONFIG_DIR=/data14/chenyh2306/scribblekitti/wandb/.config/
export TMPDIR=~/tmp

cd /data14/chenyh2306/scribblekitti/
conda activate /data14/chenyh2306/anaconda3/envs/cyh2306/
srun -w discover-01 -t 3-0 -G 2 python train_mt.py --config_path /data14/chenyh2306/scribblekitti/config/train/cylinder3d/cylinder3d_mt.yaml --dataset_config_path /data14/chenyh2306/scribblekitti/config/dataset/semantickitti.yaml

CUDA_VISIBLE_DEVICES="7" python save_mt.py --config_path config/train/cylinder3d/cylinder3d_mt.yaml  --dataset_config_path config/dataset/semantickitti.yaml --checkpoint_path output/scribblekitti/cylinder3d_mt/ckpt/epoch=72-val_teacher_miou=60.03.ckpt --save_dir inference
```
* `?`: plain Cylinder3D tuning
* `?`: Barlow Twins pretrained Cylinder3D tuning
* `?`: MEC pretrained Cylinder3D tuning

### Note

`label_directory`