from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    begin='<｜begin▁of▁sentence｜>',
    round=[
        dict(role="HUMAN", begin='User: ', end='\n\n'),
        dict(role="BOT", begin="Assistant: ", end='<｜end▁of▁sentence｜>', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='deepseek-67b-chat-hf',
        path="deepseek-ai/deepseek-llm-67b-chat",
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=4, num_procs=1),
        batch_padding=True,
    )
]
