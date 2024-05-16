from opencompass.models import HuggingFaceCausalLM

llama_models = [
    dict(
        type=HuggingFaceCausalLM,
        # 以下参数为 `HuggingFaceCausalLM` 的初始化参数
        path='meta-llama/Llama-2-7b-chat-hf',
        tokenizer_path='meta-llama/Llama-2-7b-chat-hf',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=4096,
        batch_padding=False,
        # 以下参数为各类模型都有的参数，非 `HuggingFaceCausalLM` 的初始化参数
        abbr='official-llama-7b-chat-hf',            # 模型简称，用于结果展示
        max_out_len=100,            # 最长生成 token 数
        batch_size=16,              # 批次大小
        run_cfg=dict(num_gpus=1),   # 运行配置，用于指定资源需求
    )
]