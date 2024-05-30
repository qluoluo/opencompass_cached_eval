from mmengine.config import read_base
with read_base():
    from .models.cachedLlama.cachedflashllama_7b_key_method_dim import models
    # from .datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen_a68305 import LongBench_narrativeqa_datasets
    # from .datasets.longbench.longbenchqasper.longbench_qasper_gen_6b3efc import LongBench_qasper_datasets
    # from .datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen_d3838e import LongBench_multifieldqa_en_datasets
    # from .datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen_e9a7ef import LongBench_multifieldqa_zh_datasets
        
    from .datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen import LongBench_narrativeqa_datasets
    from .datasets.longbench.longbenchqasper.longbench_qasper_gen import LongBench_qasper_datasets
    from .datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    from .datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import LongBench_multifieldqa_zh_datasets

    from .datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen import LongBench_hotpotqa_datasets
    from .datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    from .datasets.longbench.longbenchmusique.longbench_musique_gen import LongBench_musique_datasets
    from .datasets.longbench.longbenchdureader.longbench_dureader_gen import LongBench_dureader_datasets

    from .datasets.longbench.longbenchgov_report.longbench_gov_report_gen import LongBench_gov_report_datasets
    from .datasets.longbench.longbenchqmsum.longbench_qmsum_gen import LongBench_qmsum_datasets
    from .datasets.longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    from .datasets.longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets

    from .datasets.longbench.longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    from .datasets.longbench.longbenchtriviaqa.longbench_triviaqa_gen import LongBench_triviaqa_datasets
    from .datasets.longbench.longbenchsamsum.longbench_samsum_gen import LongBench_samsum_datasets
    from .datasets.longbench.longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets

    from .datasets.longbench.longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    from .datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    from .datasets.longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets

    from .datasets.longbench.longbenchlcc.longbench_lcc_gen import LongBench_lcc_datasets
    from .datasets.longbench.longbenchrepobench.longbench_repobench_gen import LongBench_repobench_datasets

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner, SlurmRunner_slurm8
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

datasets = []
datasets += LongBench_narrativeqa_datasets
datasets += LongBench_qasper_datasets
datasets += LongBench_multifieldqa_en_datasets
datasets += LongBench_multifieldqa_zh_datasets
datasets += LongBench_hotpotqa_datasets
datasets += LongBench_2wikimqa_datasets
datasets += LongBench_musique_datasets
datasets += LongBench_dureader_datasets
datasets += LongBench_gov_report_datasets
datasets += LongBench_qmsum_datasets
datasets += LongBench_multi_news_datasets
datasets += LongBench_vcsum_datasets
datasets += LongBench_trec_datasets
datasets += LongBench_triviaqa_datasets
datasets += LongBench_samsum_datasets
datasets += LongBench_lsht_datasets
datasets += LongBench_passage_count_datasets
datasets += LongBench_passage_retrieval_en_datasets
datasets += LongBench_passage_retrieval_zh_datasets
datasets += LongBench_lcc_datasets
datasets += LongBench_repobench_datasets


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        # type=SlurmRunner_slurm8,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask),
        retry=1),
)