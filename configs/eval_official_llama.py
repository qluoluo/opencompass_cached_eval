from mmengine.config import read_base
with read_base():
    # from .models.cachedLlama.cachedLlama_7b import models
    from .models.hf_llama.hf_llama2_7b_chat import models
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .datasets.mmlu.mmlu_ppl import mmlu_datasets
    from .datasets.commonsenseqa.commonsenseqa_gen import commonsenseqa_datasets

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

datasets = []
datasets += siqa_datasets
datasets += mmlu_datasets
datasets += commonsenseqa_datasets

# from mmengine.config import read_base

# with read_base():
#     from .datasets.siqa.siqa_gen import siqa_datasets
#     from .datasets.winograd.winograd_ppl import winograd_datasets
#     from .models.opt.hf_opt_125m import opt125m
#     from .models.opt.hf_opt_350m import opt350m

# datasets = [*siqa_datasets, *winograd_datasets]
# models = [opt125m, opt350m]

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=5,
        task=dict(type=OpenICLInferTask),
        retry=1),
)