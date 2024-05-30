import torch
import einops
from transformers import LlamaConfig, PretrainedConfig
from functools import partial
from typing import Tuple
import random
import copy
from sklearn.decomposition import IncrementalPCA

class AttnCacheConfig():
    def __init__(
            self,
            start_size=4,                                       # 挑选最开始的向量的个数
            recent_size=1024,                                   # 挑选最近向量的个数
            mid_size=256,                                       # 挑选中间向量的个数
            
            storage_range:str='all',                            # 存储方式，可选全部存储或者只存储开头和结尾的部分，该版本存储时会把头维度展开
            # all, start-recent

            compress_range:str='mid',                           # 压缩范围，可选全部压缩或者只压缩中间的部分
            # mid, all
            
            key_reserved_dim:int=512,                           # key压缩后保留的维度
            key_compress_method:str='none',                     # value压缩方式，不压缩可设置为none
            key_compress_split_head:bool=False,                 # 是否以head维度分开压缩key

            value_reserved_dim:int=512,                         # value压缩后保留的维度
            value_compress_method:str='none',                   # value压缩方式，不压缩可设置为none
            value_compress_split_head:bool=False,               # 是否以head维度分开压缩value

            similarity_method:str='dotproduct',                 # 相似度计算方式
            retrieve_method:str='topk',                         # 检索方式
            retrieve_split_head:bool=False,                     # 是否以head维度分开检索value

            span_chunk_size:int=1,                              # 取出最相似的token和周围的token合计的块大小

            **kwargs
        ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.mid_size = mid_size
        self.storage_range = storage_range
        self.compress_range = compress_range

        self.similarity_method = similarity_method
        self.retrieve_method = retrieve_method
        self.retrieve_split_head = retrieve_split_head
        self.span_chunk_size = span_chunk_size
        self.additional_params = kwargs

        self.reserved_dim = {
            'key': key_reserved_dim,
            'value': value_reserved_dim,
        }
        self.compress_method = {
            'key': key_compress_method,
            'value': value_compress_method,
        }
        self.compress_split_head = {
            'key': key_compress_split_head,
            'value': value_compress_split_head,
        }

    def to_dict(self):
        return {**self.__dict__, **self.additional_params}
    
    def __str__(self):
        text = f"AttnCacheConfig:"
        for k, v in self.to_dict().items():
            text += f"\n\t{k}: {v}"
        return text

class AttnCache():
    _initial_hint = True
    _detailed_info_hint = True

    def __init__(self, attn_cache_config: AttnCacheConfig, model_config: LlamaConfig):
        assert type(attn_cache_config) == AttnCacheConfig

        self.attn_config = attn_cache_config
        self.model_config = model_config

        if AttnCache._initial_hint:
            print("="*100)
            print("AttnCache Version2.0 is initialized with configuration:")
            print(self.attn_config)
            print("="*100 + "\n")
            AttnCache._initial_hint = False

        self.storage_function = self.get_avail_method(self.attn_config.storage_range, {
            "all": self.all_storage,
            "start-recent": self.start_recent_storage,
        })

        # 只存储开头结尾时，不支持检索方法
        if self.attn_config.storage_range == "start-recent" and self.attn_config.retrieve_method != "none":
            raise ValueError("start-recent storage method cannot be used with retrieve method")
        
        # 只存储开头结尾时，不支持压缩中间的
        if self.attn_config.storage_range == "start-recent" and self.attn_config.compress_range == "mid":
            raise ValueError("start-recent storage method cannot be used with compress range mid")
        
        # 不能同时start recent mid全部设为0
        if self.attn_config.mid_size == 0 and self.attn_config.recent_size == 0 and self.attn_config.start_size == 0:
            raise ValueError("mid_size, recent_size, and start_size cannot all be 0")
        
        kv_group_num = model_config.num_attention_heads // model_config.num_key_value_heads
        assert kv_group_num * model_config.num_key_value_heads == model_config.num_attention_heads, "num_attention_heads must be divisible by num_key_value_heads"
        kv_dim = self.model_config.hidden_size // kv_group_num
        assert kv_dim * kv_group_num == self.model_config.hidden_size, "kv_dim must be divisible by kv_group_num"
        q_dim = self.model_config.hidden_size
        
        # kv_dim是k和v的隐藏维度的数量，q_dim是q的隐藏维度的数量
        # 这里会打印k和v存储时未压缩之前的维度大小
        # 如果每个头分散来算，那么压缩之前的维度就是每个头的维度，否则就是kv总体的hidden states最后一维的大小
        self.pre_compress_dim = dict()
        self.reserved_dim = dict()

        self.head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads

        for obj_name in ['key', 'value']:
            if self.attn_config.compress_split_head is True:
                self.pre_compress_dim[obj_name] = self.head_dim
                raise NotImplementedError("Not Implemented for compress_split_head")
            else:
                self.pre_compress_dim[obj_name] = kv_dim
            self.reserved_dim[obj_name] = self.attn_config.reserved_dim[obj_name]
        
        
        self.compress_function = dict()
        self.decompress_function = dict()
        self.cut_decompress_right_matrix = dict()
        self.pca_begin = dict()
        self.pca_model = dict()

        for obj_name in ['key', 'value']:
            compress_available_method = {
                "none": partial(self.none_compress, obj_name=obj_name),
                "cut-random": partial(self.cut_compress, obj_name=obj_name),
                "cut-prefix": partial(self.cut_compress, obj_name=obj_name), 
                "cut-suffix": partial(self.cut_compress, obj_name=obj_name),
                "cut-head-prefix": partial(self.cut_compress, obj_name=obj_name),
                "cut-head-suffix": partial(self.cut_compress, obj_name=obj_name),
                "proj": partial(self.proj_compress, obj_name=obj_name),
                "incrementalpca": partial(self.incremental_pca_compress, obj_name=obj_name),
            }
            self.compress_function[obj_name] = self.get_avail_method(self.attn_config.compress_method[obj_name], compress_available_method, obj_name=obj_name)

            # 初始化cut_reserved_dim_idx
            if self.attn_config.compress_method[obj_name].startswith("cut"):
                if not hasattr(self, 'cut_reserved_dim_idx') or obj_name not in self.cut_reserved_dim_idx.keys():
                    strategy = self.attn_config.compress_method[obj_name].split('-', maxsplit=1)[1]
                    assert strategy in ['suffix', 'prefix', 'random', 'head-prefix', 'head-suffix'], "strategy must be suffix, prefix, random, head-prefix, head-suffix"
                    self.cut_compress_init(strategy, obj_name)

            # 初始化pca
            if self.attn_config.compress_method[obj_name].endswith('pca'):
                self.pca_begin[obj_name] = False
                self.pca_model[obj_name] = IncrementalPCA(n_components=self.reserved_dim[obj_name])

            decompress_available_method = {
                "none": partial(self.none_decompress, obj_name=obj_name),
                "cut-random": partial(self.cut_decompress_by_proj, obj_name=obj_name),
                "cut-prefix": partial(self.cut_decompress_by_proj, obj_name=obj_name),
                "cut-suffix": partial(self.cut_decompress_by_proj, obj_name=obj_name),
                "cut-head-prefix": partial(self.cut_decompress_by_proj, obj_name=obj_name),
                "cut-head-suffix": partial(self.cut_decompress_by_proj, obj_name=obj_name),
                "proj": partial(self.proj_decompress, obj_name=obj_name),
                "incrementalpca": partial(self.incremental_pca_decompress, obj_name=obj_name),
            }
            self.decompress_function[obj_name] = self.get_avail_method(self.attn_config.compress_method[obj_name], decompress_available_method, obj_name=obj_name, raise_error=False)
        

        self.get_similarity_function = self.get_avail_method(self.attn_config.similarity_method, {
            "dotproduct": self.dot_similarity,
        })

        self.retrieve_function = self.get_avail_method(self.attn_config.retrieve_method, {
            "none": self.none_retrieve,
            "topk": self.topk_retrieve_in_mid_keys,
            "random": self.random_retrieve,
        })


        # 分别是 start recent mid，如果压缩范围是all，都需要压缩了再存储，如果压缩范围是mid，则只需要压缩mid
        self.cache = {
            'key': {
                "start": None,
                "recent": None,
                "mid": None,
            },
            'value': {
                "start": None,
                "recent": None,
                "mid": None,
            },
        }

        # 打印详细信息
        if AttnCache._detailed_info_hint:
            print(f"input hidden_dim: {q_dim=} {kv_dim=}")
            print(f"{self.head_dim=}")
            print(f"{self.pre_compress_dim=}")
            print(f"{self.reserved_dim=}")
            print(f"{self.compress_function=}")
            print(f"{self.decompress_function=}")
            print(f"{self.get_similarity_function=}")
            print(f"{self.retrieve_function=}")
            print(f"{self.cache=}")

            print("-" * 50)
            AttnCache._detailed_info_hint = False
    
    #检查init中的参数错误
    def get_avail_method(self, method_name: str, available_methods: dict, raise_error=True, **kwargs):
        method_name = method_name.lower()
        if method_name not in available_methods and raise_error:
            raise ValueError(f"Method '{method_name}' is not available. Available methods: {', '.join(available_methods.keys())}")
        return partial(available_methods.get(method_name, None), **kwargs)
    
    def print_cache_shape(self):
        print('-'*100)
        for k, v in self.cache.items():
            print(f"{k}:")
            for i, j in v.items():
                print(f"{i} {j.shape if j is not None else 'None'}")
        print('-'*100)

    # 通过投影矩阵更新反解矩阵
    def update_layer_info(self, q_proj_weight, k_proj_weight, v_proj_weight):
        # k_proj_weight.shape = [proj_dim, hidden_dim]
        # 如果是cut方法，并且还没有准备好反解矩阵
        if self.attn_config.compress_method['key'].startswith("cut") and self.cut_decompress_right_matrix.get('key', None) is None:
                # 如果是合并解压
                if self.attn_config.compress_split_head['key'] is False:
                    cut_k_proj_weight = k_proj_weight[self.cut_reserved_dim_idx['key'], :].float()
                    cut_k_proj_weight_inv = torch.pinverse(cut_k_proj_weight).to(k_proj_weight.device).to(k_proj_weight.dtype)
                    self.cut_decompress_right_matrix['key'] = torch.matmul(k_proj_weight, cut_k_proj_weight_inv).transpose(0, 1)
                # 如果是分开解压
                else:
                    # raise NotImplementedError("cut-split-head is not supported yet")
                    inv_proj_list = []
                    for i in range(self.model_config.num_key_value_heads):
                        proj = k_proj_weight[i * self.head_dim:(i + 1) * self.head_dim, :]
                        cut_proj = proj[self.cut_reserved_dim_idx, :]

        if self.attn_config.compress_method['value'].startswith("cut"):
            if self.cut_decompress_right_matrix.get('value', None) is None:
                cut_v_proj_weight = v_proj_weight[self.cut_reserved_dim_idx['value'], :].float()
                cut_v_proj_weight_inv = torch.pinverse(cut_v_proj_weight).to(v_proj_weight.device).to(v_proj_weight.dtype)
                self.cut_decompress_right_matrix['value'] = torch.matmul(v_proj_weight, cut_v_proj_weight_inv).transpose(0, 1)

    def dynamic_concat(self, old_tensor: torch.Tensor, new_tensor: torch.Tensor, dim=-2):
        if old_tensor is None:
            return new_tensor
        else:
            new_tensor = new_tensor.to(old_tensor.device).to(old_tensor.dtype)
            return torch.cat([old_tensor, new_tensor], dim=dim)

    # 存储方式函数
    def all_storage(self, states: torch.Tensor, obj_name: str):
        
        # offset标志着下一个进入缓冲区的索引序号
        offset_seq_len = 0

        # 如果全部压缩，进来之前压缩就行了
        if self.attn_config.compress_range == "all":
            states = self.compress_function[obj_name](states, update_state=True)

        # 如果缓存连最开始的start_size都没满
        if self.attn_config.start_size > 0 and offset_seq_len < states.shape[-2] and (self.cache[obj_name]['start'] is None or self.cache[obj_name]['start'].shape[-2] < self.attn_config.start_size):
            cat_seq_len = min(states.shape[-2] - offset_seq_len, self.attn_config.start_size - self.get_cached_size(obj_name))
            self.cache[obj_name]['start'] = self.dynamic_concat(
                self.cache[obj_name]['start'],
                states[..., offset_seq_len:offset_seq_len+cat_seq_len, :])
            
            offset_seq_len += cat_seq_len

        # 如果startsize装完了，那就不继续了
        if offset_seq_len == states.shape[-2]:
            return
        

        # recent实际上相当于一个先进先出的队列
        # 无论之前有没有填充recent，都完全拼接到recent最后，再把recent开头的多余的给mid即可
        self.cache[obj_name]['recent'] = self.dynamic_concat(
            self.cache[obj_name]['recent'],
            states[..., offset_seq_len:, :]
        )

        # 如果recent有多余的部分
        if self.cache[obj_name]['recent'].shape[-2] > self.attn_config.recent_size:
            # 把recent多余部分移到mid的末尾
            remove_recent_head_len = self.cache[obj_name]['recent'].shape[-2] - self.attn_config.recent_size

            # 如果压缩范围是只压缩中间的部分，则移出的部分需要先压缩再移到mid部分中，否则直接移到mid即可
            move_part = self.cache[obj_name]['recent'][..., :remove_recent_head_len, :]
            if self.attn_config.compress_range == 'mid':
                move_part = self.compress_function[obj_name](move_part, update_state=True)

            self.cache[obj_name]['mid'] = self.dynamic_concat(self.cache[obj_name]['mid'], move_part)

            # recent移除前端部分
            self.cache[obj_name]['recent'] = self.cache[obj_name]['recent'][..., remove_recent_head_len:, :]

    def start_recent_storage(self, states: torch.Tensor, obj_name: str):
        # 新方法只需要把all_storage的cache结果的mid设为None即可
        self.all_storage(self, states, obj_name)
        self.cache[obj_name]['mid'] = None

    def get_cached_size(self, obj_name):
        cached_size = 0
        for v in self.cache[obj_name].values():
            if v is not None:
                cached_size += v.shape[-2]
        return cached_size


    # 压缩方式函数
    def none_compress(self, matrix: torch.Tensor, **kwargs):
        return matrix
    
    def none_decompress(self, matrix: torch.Tensor, **kwargs):
        return matrix

    # 第一次使用cut_compress会进行初始化
    def cut_compress_init(self, strategy: str, obj_name: str):
        if not hasattr(self, 'cut_reserved_dim_idx'):
            self.cut_reserved_dim_idx = dict()

        obj_pre_compress_dim = self.pre_compress_dim[obj_name]
        obj_reserved_dim = self.reserved_dim[obj_name]

        if strategy == "random":
            self.cut_reserved_dim_idx[obj_name] = torch.randperm(obj_pre_compress_dim)[:obj_reserved_dim]
        elif strategy == "prefix":
            self.cut_reserved_dim_idx[obj_name] = torch.arange(obj_pre_compress_dim)[:obj_reserved_dim]
        elif strategy == "suffix":
            self.cut_reserved_dim_idx[obj_name] = torch.arange(obj_pre_compress_dim)[-obj_reserved_dim:]
        elif strategy == "head-prefix":
            reserved_head_dim = obj_reserved_dim // self.model_config.num_key_value_heads
            assert reserved_head_dim > 0, "reserved_head_dim must greater than 0"
            self.cut_reserved_dim_idx[obj_name] = torch.cat([torch.arange(self.head_dim)[:reserved_head_dim] + i * self.head_dim for i in range(self.model_config.num_key_value_heads)])
        elif strategy == "head-suffix":
            reserved_head_dim = obj_reserved_dim // self.model_config.num_key_value_heads
            assert reserved_head_dim > 0, "reserved_head_dim must greater than 0"
            self.cut_reserved_dim_idx[obj_name] = torch.cat([torch.arange(self.head_dim)[-reserved_head_dim:] + i * self.head_dim for i in range(self.model_config.num_key_value_heads)])
        else:
            self.cut_reserved_dim_idx = None
            raise NotImplementedError(f"strategy {strategy} not implemented")

    def cut_compress(self, matrix: torch.Tensor, obj_name: str, **kwargs):
        ## obj_name 用于区分k和v具体压缩时的不同参数

        reserved_dim = self.reserved_dim[obj_name]
        if reserved_dim > matrix.shape[-1] or reserved_dim <= 0:
            raise ValueError(f"reserved_dim {reserved_dim} should be less than hidden_dim {matrix.shape[-1]} and greater than 0")

        return matrix[..., self.cut_reserved_dim_idx[obj_name]]
    
    def cut_decompress_by_proj(self, matrix: torch.Tensor, obj_name: str):
        if matrix is None:
            return None
        self.cut_decompress_right_matrix[obj_name] = self.cut_decompress_right_matrix[obj_name].to(matrix.dtype).to(matrix.device)
        return torch.matmul(matrix, self.cut_decompress_right_matrix[obj_name])

    def incremental_pca_compress(self, matrix: torch.Tensor, update_state: bool, obj_name: str, **kwargs):
        # 将 Tensor 转换为 NumPy 数组进行 PCA 处理
        reserved_dim = self.reserved_dim[obj_name]
        batch_size, seq_len, hidden_dim = matrix.shape
        assert batch_size == 1, "incremental_pca_compress only support batch_size=1"

        if reserved_dim > matrix.shape[-1] or reserved_dim <= 0:
            raise ValueError(f"reserved_dim {reserved_dim} should be less than hidden_dim {matrix.shape[-1]} and greater than 0")
        if reserved_dim == matrix.shape[-1]:
            return matrix
        
        # 如果还没有开始PCA，判断是否达到PCA的阈值，到了就开始PCA
        if self.pca_begin[obj_name] is False:
            if self.cache[obj_name]['mid'] is None or self.cache[obj_name]['mid'].shape[-2] <= self.reserved_dim[obj_name]:
                return matrix
            else:
                self.pca_begin[obj_name] = True
                mid_cache_cpu = self.cache[obj_name]['mid'].reshape(-1, self.cache[obj_name]['mid'].shape[-1]).to('cpu')
                mid_cache_cpu = self.pca_model[obj_name].fit_transform(mid_cache_cpu)
                self.cache[obj_name]['mid'] = torch.from_numpy(mid_cache_cpu).to(self.cache[obj_name]['mid'].device)
                

        matrix_reshaped = matrix.reshape(-1, hidden_dim).to('cpu')

        if update_state:
            transformed_matrix = self.pca_model[obj_name].fit_transform(matrix_reshaped)
        else:
            transformed_matrix = self.pca_model[obj_name].transform(matrix_reshaped)
        # import ipdb; ipdb.set_trace()
        transformed_matrix = torch.from_numpy(transformed_matrix).to(matrix.device).reshape(batch_size, seq_len, -1)
        return transformed_matrix
    
    def incremental_pca_decompress(self, compressed_matrix: torch.Tensor, obj_name: str, **kwargs):  
        # 确保 compressed_matrix 的维度与 PCA 模型的输出维度匹配
        if compressed_matrix is None:
            return None
        
        if self.pca_begin[obj_name] is False:
            return compressed_matrix

        if compressed_matrix.shape[-1] != self.pca_model[obj_name].n_components:  
            raise ValueError(f"The shape of compressed_matrix {compressed_matrix.shape} does not match the number of PCA components {self.pca_model[obj_name].n_components}.")  
    
        
        # 将 Tensor 转换为 NumPy 数组进行 PCA 重建  
        compressed_matrix_reshaped = compressed_matrix.to('cpu').numpy()  
        
        # 使用 PCA 模型的主成分来重建原始数据  
        # 注意：这只是一个近似值，因为 PCA 是不可逆的  
        reconstructed_matrix = self.pca_model[obj_name].inverse_transform(compressed_matrix_reshaped)  
        
        # 将重建的数据转换回 PyTorch Tensor，并返回到原始设备  
        batch_size, seq_len, hidden_dim = compressed_matrix.shape
        reconstructed_matrix = torch.from_numpy(reconstructed_matrix).reshape(batch_size, seq_len, -1).to(compressed_matrix.device)  

        # print(f"{reconstructed_matrix.shape=}")
        
        return reconstructed_matrix

    def proj_compress(self, matrix: torch.Tensor, update_state: bool):
        reserved_dim = self.attn_config.reserved_dim
        if reserved_dim > matrix.shape[-1] or reserved_dim <= 0:
            raise ValueError(f"reserved_dim {reserved_dim} should be less than hidden_dim {matrix.shape[-1]} and greater than 0")
        if reserved_dim == matrix.shape[-1]:
            return matrix
        
        self.projection_matrix = self.projection_matrix.to(matrix.device)
        self.projection_matrix = self.projection_matrix.to(matrix.dtype)
        compressed_matrix = torch.matmul(matrix, self.projection_matrix)

        return compressed_matrix
    
    def proj_decompress(self, matrix: torch.Tensor):
        if matrix is None:
            return None
        # 将压缩矩阵映射回原始空间
        self.projection_matrix_pseudo_inv = self.projection_matrix_pseudo_inv.to(matrix.device)
        self.projection_matrix_pseudo_inv = self.projection_matrix_pseudo_inv.to(matrix.dtype)
        decompressed_matrix = torch.matmul(matrix, self.projection_matrix_pseudo_inv)
        
        return decompressed_matrix

    # 更新cache
    def update_cache(self, key_states: torch.Tensor, value_states: torch.Tensor, position_ids: torch.LongTensor):
        '''更新缓存'''
        # 改变形状
        if len(key_states.shape) == 4:
            key_states_reshaped = einops.rearrange(key_states, 'b h s d -> b s (h d)')
            value_states_reshaped = einops.rearrange(value_states, 'b h s d -> b s (h d)')
        else:
            assert len(key_states.shape) == 3
            key_states_reshaped = key_states
            value_states_reshaped = value_states
        # 存储，压缩已经集成到存储当中了
        self.storage_function(key_states_reshaped, "key")
        self.storage_function(value_states_reshaped, "value")

        # self.print_cache_shape()

    # 相似度计算函数
    def dot_similarity(self, q: torch.Tensor, k: torch.Tensor):
        """返回是一个[batch_size * k_cache_size]的矩阵"""
        return torch.einsum("bie,bje->bje", q, k).sum(2)

    # 取临近的块函数
    def fetch_chunk(self, indices, selected_len:int):
        if self.cache['key']['start'].shape[0] != 1 and self.attn_config.span_chunk_size > 1:
            raise ValueError("Only support single chunk for span_chunk_size > 1")

        if self.attn_config.span_chunk_size == 1:
            return indices

        single_batch_position_ids = self.position_ids_cache[0, :]
        span_length = self.attn_config.span_chunk_size // 2

        if self.attn_config.span_chunk_size % 2 == 0:
            offsets = torch.arange(-span_length, span_length, device=single_batch_position_ids.device)
        else:
            offsets = torch.arange(-span_length, span_length + 1, device=single_batch_position_ids.device)

        fetch_seq_ids = (single_batch_position_ids.unsqueeze(1) + offsets).view(-1)
        fetch_seq_ids = fetch_seq_ids.clamp(0, selected_len - 1)  # clamp to avoid out-of-bounds
        fetch_seq_ids = torch.unique(fetch_seq_ids)
        fetch_seq_ids, _ = torch.sort(fetch_seq_ids, dim=-1)
        return fetch_seq_ids.unsqueeze(0)
    
    def none_retrieve(self, q: torch.Tensor, key_cache, value_cache):
        return key_cache, value_cache
    
    def random_retrieve(self, q: torch.Tensor, key_cache, value_cache):
        if key_cache is None or key_cache.shape[-2] < self.attn_config.mid_size:
            return key_cache, value_cache
        idx_reserved = torch.randperm(key_cache.shape[-2])[:self.attn_config.mid_size]
        idx_reserved, _ = torch.sort(idx_reserved, dim=-1)
        return key_cache[..., idx_reserved, :], value_cache[..., idx_reserved, :]
    
    # topk检索方式
    def topk_retrieve_in_mid_keys(self, q: torch.Tensor, key_cache, value_cache):
        # topk
        # 如果缓存为空或者长度不够，则不需要检索，直接返回缓存
        if key_cache is None or key_cache.shape[-2] <= self.attn_config.mid_size:
            return key_cache, value_cache
        print(f"{q.shape=}, {key_cache.shape=}")
        scores = self.get_similarity_function(q, key_cache)
        # scores.shape = [batch_size * k_cache_size]
        scores, indices = torch.topk(scores, self.attn_config.mid_size, dim=-1)
        # indices.shape = [batch_size * mid_size]
        idx_reserved = self.fetch_chunk(indices, key_cache.shape[-2])
        
        # print("unsorted idx_reserved: ", idx_reserved)
        idx_reserved, _ = torch.sort(idx_reserved, dim=-1)
        # print("sorted idx_reserved: ", idx_reserved)

        # 分两个是因为这里key和value的最后一维的维度未必相同
        idx_reserved_key = idx_reserved.unsqueeze(-1).expand(-1, -1, key_cache.shape[-1])
        idx_reserved_value = idx_reserved.unsqueeze(-1).expand(-1, -1, value_cache.shape[-1])
        
        key_cache_fetched = torch.gather(key_cache, dim=-2, index=idx_reserved_key)
        value_cache_fetched = torch.gather(value_cache, dim=-2, index=idx_reserved_value)
        
        return key_cache_fetched, value_cache_fetched

    # 查询cache
    def query_cache(self, query_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.get_cached_size('key') == 0:
            return query_states, None, None
        # query_shape = [batch_size, head_num, query_seq_len, hidden_dim]
        # 首先修正形状
        query_states_reshaped = einops.rearrange(query_states, 'b h s d -> b s (h d)')
        # 首先均对query进行压缩，然后进行相似度检索，从midkey中找到相似的key
        # 对这些key和value进行还原，并且返回
        query_states_reshaped = self.compress_function['key'](query_states_reshaped, update_state=False)
        key_cache_retrieved, value_cache_retrieved = self.retrieve_function(query_states_reshaped, self.cache['key']['mid'], self.cache['value']['mid'])
        # if key_cache_retrieved is not None:
        #     self.print_cache_shape()
        #     print(f"{key_cache_retrieved.shape=}, {value_cache_retrieved.shape=}")

        key_cache_retrieved = self.decompress_function['key'](key_cache_retrieved)
        value_cache_retrieved = self.decompress_function['value'](value_cache_retrieved)
        

        key_cache_list = [x for x in [self.cache['key']['start'], key_cache_retrieved, self.cache['key']['recent']] if x is not None]
        value_cache_list = [x for x in [self.cache['value']['start'], value_cache_retrieved, self.cache['value']['recent']] if x is not None]
        if key_cache_list == []:
            key_cache = None
            value_cache = None
        else:
            key_cache = torch.cat(key_cache_list, dim=-2)
            value_cache = torch.cat(value_cache_list, dim=-2)
            key_cache = einops.rearrange(key_cache, 'b s (h d) -> b h s d', h=self.model_config.num_key_value_heads)
            value_cache = einops.rearrange(value_cache, 'b s (h d) -> b h s d', h=self.model_config.num_key_value_heads)
        return query_states, key_cache, value_cache
    
    # 清空cache
    def clean_cache(self):
        self.cache = {
            'key': {
                "start": None,
                "recent": None,
                "mid": None,
            },
            'value': {
                "start": None,
                "recent": None,
                "mid": None,
            },
        }
        # torch.cuda.empty_cache()