## OpenCompass Eval

本仓库使用OpenCompass 司南 2.0 大模型评测体系进行模型效果评测

### 💻 环境配置

#### 使用面向开源模型的GPU环境

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/qluoluo/opencompass_cached_eval opencompass
cd opencompass
pip install -e .
```
### 📕 数据准备

#### 使用OpenCompass提供的一些第三方数据集及自建中文数据集。运行以下命令手动下载解压。

在 OpenCompass 项目根目录下运行下面命令，将数据集准备至 ${OpenCompass}/data 目录下：
```
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

### 🚀 开始评测
已经做了Llama2与其他若干模型的适配工作。运行以下代码开始评测
opencompass详细用法可以至[OpenCompass官方教程](https://opencompass.readthedocs.io/zh-cn/latest/index.html)查看

```
## 以下的python脚本均使用了slurmRunner，如不处于集群中需要至脚本中改为LocalRunner
## 需要把slurmName自行替换为slurm分区名称
## 需要到对应脚本中改变模型地址

python run.py configs/eval_cachedflashInterLM2.py -w outputs/eval_result/ -p slurmName
python run.py configs/eval_cachedflashLlama.py -w outputs/eval_result/ -p slurmName
```