import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
from transformers import LlamaConfig

PromptType = Union[PromptList, str]

from ..huggingface import HuggingFaceCausalLM

import sys
sys.path.append("/remote-home/zgliu/wrote_program/kvcache_experiment")
from AttnCache import AttnCacheConfig


@MODELS.register_module()
class CachedHuggingFaceCausalLM(HuggingFaceCausalLM):
    pass