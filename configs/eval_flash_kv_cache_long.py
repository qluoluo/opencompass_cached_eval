from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .datasets.collections.C_long import datasets
    from .my_model.flash_kv_cache import models

work_dir = './outputs/flash_kv_cache/'

volcano_infer_cfg = dict(
    bashrc_path="/fs-computility/llm/liuxiaoran/.bashrc",  # bashrc 路径
    # conda_env_name='flash2.0',  #conda 环境名
    conda_path="/fs-computility/llm/liuxiaoran/envs/flash2.0",  #conda环境启动路径
    volcano_config_path="/fs-computility/llm/liuxiaoran/projects/opencompass_cached_eval/configs/configs/volc_config/volcano_infer.yaml"  #配置文件路径
)

volcano_eval_cfg = dict(
    bashrc_path="/fs-computility/llm/liuxiaoran/.bashrc",  # bashrc 路径
    # conda_env_name='flash2.0',
    conda_path="/fs-computility/llm/liuxiaoran/envs/flash2.0",
    volcano_config_path="/fs-computility/llm/liuxiaoran/projects/opencompass_cached_eval/configs/configs/volc_config/volcano_eval.yaml"
)

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    runner=dict(
        type=VOLCRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask), 
        volcano_cfg=volcano_infer_cfg, 
        queue_name='hsllm_c', 
        retry=4),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=VOLCRunner,
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask),
        volcano_cfg=volcano_eval_cfg, 
        queue_name='hsllm_c', 
        retry=4),
)

# source /fs-computility/llm/liuxiaoran/.bashrc
# conda activate /fs-computility/llm/liuxiaoran/envs/flash2.0
# python run.py configs/eval_flash_kv_cache_long.py --debug 调试用
# python run.py configs/eval_flash_kv_cache_long.py 第一次用
#  . 第二次用
