# Weakly Supervised Semantic Segmentation for Large-Scale Point Cloud (AAAI 2021, Mindspore)

# 论文
- [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16455)

# 环境配置

- python==3.7
- mindspore == 1.7

```shell
pip install -r requirements.txt
bash local_install_third_part.sh
```

# S3DIS数据集准备

## 方式1. 从百度网盘链接下载处理好的数据（推荐）

[百度网盘链接](https://pan.baidu.com/s/101vw5nE-a9CmznWbIcSG_w?pwd=50dh)

数据集目录如下：

```shell
dataset
-- S3DIS # S3DIS数据集
---- input_0.040
------ *.ply
------ *_KDTree.pkl
------ *_proj.pkl
---- original_ply
------ *.ply
WS3_Project # WS3项目路径
-- data    # 数据集相关的.py文件
-- model   # 模型相关的.py文件
-- utils   
-- train_modelarts_notebook_remove_bias.py  
-- eval_modelarts_notebook_remove_bias.py
-- train_modelarts_remove_bias.py  
-- eval_modelarts_remove_bias.py
-- train_gpu.py
-- eval_gpu.py
```

## 方式2. 从S3DIS官方下载数据集，并执行数据处理

1.
可在 [S3DIS链接](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1)
找到数据集 ，下载`Stanford3dDataset_v1.2_Aligned_Version.zip`
2. 将`Stanford3dDataset_v1.2_Aligned_Version.zip`解压至`dataset/S3DIS`目录下.
3. 安装依赖:

```shell
cd utils/cpp_wrappers
sh compile_warppers.sh
```

4. 数据处理

```shell
python utils/data_prepare_s3dis.py
```

数据集目录格式同方式1

# Ascend环境下训练和验证

## 通过ModelArts的notebook

- 镜像为 `mindspore1.7.0-cann5.1.0-py3.7-euler2.8.3`

### 训练

```shell
python train_modelarts_notebook_remove_bias.py \
--dataset_dir ../dataset/S3DIS \
--device_target Ascend \
--float16 True \
--outputs_dir ./outputs \
--name BatchS_6_Float16_PyNative_Ascend
```

### 验证

```shell
python eval_modelarts_notebook_remove_bias.py \
--dataset_dir ../dataset/S3DIS \
--device_target Ascend \
--model_path ./outputs/BatchS_6_Float16_PyNative_Ascend
```

## 通过ModelArts的训练作业

- 镜像为 `Ascend-Powered-Engine | mindspore_1.7.0-cann_5.1.0-py_3.7-euler_2.8.3-aarch64`

### 训练

- 启动文件: `train_modelarts_remove_bias.py`
- 训练输入： 需要指定`dataset_dir`
    - `dataset`: 在obs桶上，数据集的路径。比如: `/xxxx/xxx/dataset/S3DIS`
- 训练输出： 需要指定`output_dir`
    - `output_dir`: 模型参数和日志等输出文件的保存路径。比如:`/xxxx/xxx/WS3/outputs`
- 超参：
    - `float16`: True

### 验证

- 启动文件: `eval_modelarts_remove_bias.py`
- 训练输入： 需要指定`dataset_dir` 和 `model_path`
    - `dataset`: 在obs桶上，数据集的路径。比如: `/xxxx/xxx/dataset/S3DIS`
    - `model_path`：在obs桶上，模型参数保存的路径, 在`output_dir`
      内。比如：`/xxxx/xxx/WS3/outputs/TSteps500_MaxEpoch100_BatchS6_lr0.01_lrd0.95_ls1.0_Topk500_NumTrainEp030_LP_1_RS_888_PyNateiveM_2022-11-08_22-53`
- 训练输出： 需要指定`output_dir`
    - `output_dir`: 模型参数和日志等输出文件的保存路径。比如:`/xxxx/xxx/WS3/outputs`
- 超参：
    - `float16`: True

# GPU环境下训练和验证

### 训练

```shell
python train_gpu.py \
--dataset_dir ../dataset/S3DIS \
--device_id 0 \
--outputs_dir ./outputs \
--name BatchS_6_Float32_PyNative_GPU
```

### 验证

```shell
python eval_gpu.py \
--dataset_dir ../dataset/S3DIS \
--device_id 0 \
--model_path ./outputs/BatchS_6_Float32_PyNative_GPU
```

## Citing

### BibTeX

```bibtex
@inproceedings{zhang2021weakly,
  title={Weakly supervised semantic segmentation for large-scale point cloud},
  author={Zhang, Yachao and Li, Zonghao and Xie, Yuan and Qu, Yanyun and Li, Cuihua and Mei, Tao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={4},
  pages={3421--3429},
  year={2021}
}
```

