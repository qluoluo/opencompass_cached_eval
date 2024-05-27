from mmengine.config import read_base

with read_base():
    
    # long task for long score
    
    from ..longbench.longbenchnarrativeqa.longbench_narrativeqa_gen import LongBench_narrativeqa_datasets
    from ..longbench.longbenchqasper.longbench_qasper_gen import LongBench_qasper_datasets
    from ..longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    from ..longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import LongBench_multifieldqa_zh_datasets

    from ..longbench.longbenchhotpotqa.longbench_hotpotqa_gen import LongBench_hotpotqa_datasets
    from ..longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    from ..longbench.longbenchmusique.longbench_musique_gen import LongBench_musique_datasets
    from ..longbench.longbenchdureader.longbench_dureader_gen import LongBench_dureader_datasets

    from ..longbench.longbenchgov_report.longbench_gov_report_gen import LongBench_gov_report_datasets
    from ..longbench.longbenchqmsum.longbench_qmsum_gen import LongBench_qmsum_datasets
    from ..longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    from ..longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets

    from ..longbench.longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    from ..longbench.longbenchtriviaqa.longbench_triviaqa_gen import LongBench_triviaqa_datasets
    from ..longbench.longbenchsamsum.longbench_samsum_gen import LongBench_samsum_datasets
    from ..longbench.longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets

    from ..longbench.longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    from ..longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    from ..longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets

    from ..longbench.longbenchlcc.longbench_lcc_gen import LongBench_lcc_datasets
    from ..longbench.longbenchrepobench.longbench_repobench_gen import LongBench_repobench_datasets
    
    # from ..leval.levaltpo.leval_tpo_gen import LEval_tpo_datasets
    # from ..leval.levalgsm100.leval_gsm100_gen import LEval_gsm100_datasets
    # from ..leval.levalquality.leval_quality_gen import LEval_quality_datasets
    # from ..leval.levalcoursera.leval_coursera_gen import LEval_coursera_datasets
    # from ..leval.levaltopicretrieval.leval_topic_retrieval_gen import LEval_tr_datasets
    # from ..leval.levalscientificqa.leval_scientificqa_gen import LEval_scientificqa_datasets
    
    # from ..leval.levalmultidocqa.leval_multidocqa_gen import LEval_multidocqa_datasets
    # from ..leval.levalpaperassistant.leval_paper_assistant_gen import LEval_ps_summ_datasets
    # from ..leval.levalnaturalquestion.leval_naturalquestion_gen import LEval_nq_datasets
    # from ..leval.levalfinancialqa.leval_financialqa_gen import LEval_financialqa_datasets
    # from ..leval.levallegalcontractqa.leval_legalcontractqa_gen import LEval_legalqa_datasets
    # from ..leval.levalnarrativeqa.leval_narrativeqa_gen import LEval_narrativeqa_datasets

    # from ..leval.levalnewssumm.leval_newssumm_gen import LEval_newssumm_datasets
    # from ..leval.levalgovreportsumm.leval_gov_report_summ_gen import LEval_govreport_summ_datasets
    # from ..leval.levalpatentsumm.leval_patent_summ_gen import LEval_patent_summ_datasets
    # from ..leval.levaltvshowsumm.leval_tvshow_summ_gen import LEval_tvshow_summ_datasets
    # from ..leval.levalmeetingsumm.leval_meetingsumm_gen import LEval_meetingsumm_datasets
    # from ..leval.levalreviewsumm.leval_review_summ_gen import LEval_review_summ_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
