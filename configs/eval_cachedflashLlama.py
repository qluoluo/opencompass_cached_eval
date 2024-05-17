from mmengine.config import read_base
with read_base():
    from .models.cachedLlama.cachedflashllama_7b import models
    from .datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen_a68305 import LongBench_narrativeqa_datasets
    from .datasets.longbench.longbenchqasper.longbench_qasper_gen_6b3efc import LongBench_qasper_datasets
    from .datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen_d3838e import LongBench_multifieldqa_en_datasets
    from .datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen_e9a7ef import LongBench_multifieldqa_zh_datasets

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

datasets = []
# datasets += LongBench_narrativeqa_datasets
datasets += LongBench_qasper_datasets
# datasets += LongBench_multifieldqa_en_datasets
# datasets += LongBench_multifieldqa_zh_datasets

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=1,
        task=dict(type=OpenICLInferTask),
        retry=1),
)