srun -p a800 --cpus-per-task=16 --mem-per-cpu=4G --gres=gpu:1 python
srun -p x090 --cpus-per-task=16 --mem-per-cpu=4G --gres=gpu:1 python

## 运行opencompass必须先设置
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU