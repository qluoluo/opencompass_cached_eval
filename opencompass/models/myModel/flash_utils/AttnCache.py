import torch
import einops
from transformers import LlamaConfig, PretrainedConfig
from functools import partial
from typing import Tuple
import random
import copy

class AttnCacheConfig():
    def __init__(
            self,
            start_size=4,                       # 挑选最开始的向量的个数
            recent_size=1024,                   # 挑选最近向量的个数
            mid_size=256,                       # 挑选中间向量的个数
            storage_method:str='all',           # 存储方式，可选全部存储或者只存储开头和结尾的部分
            compress_method:str='cut-prefix',   # 压缩方式
            compress_range:str='mid',           # 压缩范围，可选全部压缩或者只压缩中间的部分
            recover:bool=True,                  # 是否恢复向量到原来的维度
            reserved_dim=3072,                  # 压缩到多少维度
            similarity_method:str='dotproduct', # 相似度计算方式
            retrieve_method:str='normal',       # 检索方式
            span_chunk_size:int=1,              # 取出最相似的token和周围的token合计的块大小
            new_decompress_method:bool=True,    # 是否使用proj反解
            max_storage_mid_size=-1,
            **kwargs
        ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.mid_size = mid_size
        self.storage_method = storage_method
        self.compress_method = compress_method
        self.compress_range = compress_range
        self.recover = recover
        self.reserved_dim = reserved_dim
        self.similarity_method = similarity_method
        self.retrieve_method = retrieve_method
        self.span_chunk_size = span_chunk_size
        self.new_decompress_method = new_decompress_method
        self.max_storage_mid_size = max_storage_mid_size

    def __str__(self):
        return (f"AttnCacheConfig(\n"
                f"  start_size={self.start_size},\n"
                f"  recent_size={self.recent_size},\n"
                f"  mid_size={self.mid_size},\n"
                f"  storage_method='{self.storage_method}',\n"
                f"  compress_method='{self.compress_method}',\n"
                f"  compress_range='{self.compress_range}',\n"
                f"  recover={self.recover},\n"
                f"  reserved_dim={self.reserved_dim},\n"
                f"  similarity_method='{self.similarity_method}',\n"
                f"  retrieve_method='{self.retrieve_method}',\n"
                f"  span_chunk_size={self.span_chunk_size}\n"
                f"  new_decompress_method={self.new_decompress_method}\n"
                f"  max_storage_mid_size={self.max_storage_mid_size}\n"
                f")")
    
    def to_dict(self):
        return {
            "start_size": self.start_size,
            "recent_size": self.recent_size,
            "mid_size": self.mid_size,
            "storage_method": self.storage_method,
            "compress_method": self.compress_method,
            "compress_range": self.compress_range,
            "recover": self.recover,
            "reserved_dim": self.reserved_dim,
            "similarity_method": self.similarity_method,
            "retrieve_method": self.retrieve_method,
            "span_chunk_size": self.span_chunk_size,
            "new_decompress_method": self.new_decompress_method,
            "max_storage_mid_size": self.max_storage_mid_size,
        }

class AttnCache():
    _initial_hint = True

    def __init__(self, attn_cache_config: AttnCacheConfig, llama_config):
        assert type(attn_cache_config) == AttnCacheConfig 
        # assert type(llama_config) == PretrainedConfig

        self.attn_config = attn_cache_config
        self.llama_config = llama_config

        if AttnCache._initial_hint:
            print("="*100)
            print("AttnCache is initialized with configuration:")
            print(self.attn_config)
            print("="*100 + "\n")
            AttnCache._initial_hint = False

        self.storage_function = self.get_avail_method(self.attn_config.storage_method, {
            "all": self.all_storage,
            "start-recent": self.start_recent_storage,
        })

        # 只存储开头结尾时，不支持检索方法
        if self.attn_config.storage_method == "start-recent" and self.attn_config.retrieve_method != "none":
            raise ValueError("start-recent storage method cannot be used with retrieve method")
        
        # 只存储开头结尾时，不支持压缩中间的
        if self.attn_config.storage_method == "start-recent" and self.attn_config.compress_range == "mid":
            raise ValueError("start-recent storage method cannot be used with compress range mid")
        
        # 指定压缩范围时，必须指定压缩方法
        # if self.attn_config.compress_range != 'none' and self.attn_config.compress_method == 'none':
        #     raise ValueError("compress range cannot be used with compress method none")
        
        # 不能同时start recent mid全部设为0
        if self.attn_config.mid_size == 0 and self.attn_config.recent_size == 0 and self.attn_config.start_size == 0:
            raise ValueError("mid_size, recent_size, and start_size cannot all be 0")

        self.compress_function = self.get_avail_method(self.attn_config.compress_method, {
            "none": self.none_compress,
            "cut-random": partial(self.cut_compress, strategy='random'),
            "cut-prefix": partial(self.cut_compress, strategy='prefix'),
            "cut-suffix": partial(self.cut_compress, strategy='suffix'),
            "cut-head-prefix": partial(self.cut_compress, strategy='head-prefix'),
            "cut-head-suffix": partial(self.cut_compress, strategy='head-suffix'),
            "proj": self.proj_compress,
            "incrementalpca": self.incremental_pca_compress,
        })

        if not self.attn_config.new_decompress_method:
        # if False:
            self.decompress_function = self.get_avail_method(self.attn_config.compress_method, {
                "none": self.none_decompress,
                "cut-random": self.cut_decompress,
                "cut-prefix": self.cut_decompress,
                "cut-suffix": self.cut_decompress,
                "cut-head-prefix": self.cut_decompress,
                "cut-head-suffix": self.cut_decompress,
                "proj": self.proj_decompress,
                "incrementalpca": self.incremental_pca_decompress,
            }, raise_error=False)
        else:
            self.decompress_function = self.get_avail_method(self.attn_config.compress_method, {
                "none": self.none_decompress,
                "cut-random": self.cut_decompress_by_kproj,
                "cut-prefix": self.cut_decompress_by_kproj,
                "cut-suffix": self.cut_decompress_by_kproj,
                "cut-head-prefix": self.cut_decompress_by_kproj,
                "cut-head-suffix": self.cut_decompress_by_kproj,
                "proj": self.proj_decompress,
                "incrementalpca": self.incremental_pca_decompress,
            }, raise_error=False)

        self.retrieve_function = self.get_avail_method(self.attn_config.retrieve_method, {
            "none": self.none_retrieve,
            "normal": self.normal_retrieve_in_mid_keys,
            "random": self.random_retrieve,
            # "FAISS": self.faiss_retrieve,
        })

        self.get_similarity_function = self.get_avail_method(self.attn_config.similarity_method, {
            "dotproduct": self.dot_similarity,
            # "cosine": self.cosine_similarity,
        })

        # 分别是 start recent mid，如果压缩范围是all，都需要压缩了再存储，如果压缩范围是mid，则只需要压缩mid
        self.key_cache = [None] * 3
        self.value_cache = [None] * 3
        # self.position_ids_cache = [None] * 3

        # -----------------------------------------------------------------------------------------------------------------------------------
        # 下文是一些部分功能所需的必要的插件

        # 所有的cut方式都可以最开始确定保留的维度，故在初始化函数中完成
        if self.attn_config.compress_method == "cut-random":
            self.reserved_dim_idx = torch.randperm(self.llama_config.hidden_size)[:self.attn_config.reserved_dim]
        elif self.attn_config.compress_method == "cut-prefix":
            self.reserved_dim_idx = torch.arange(self.llama_config.hidden_size)[:self.attn_config.reserved_dim]
        elif self.attn_config.compress_method == "cut-suffix":
            self.reserved_dim_idx = torch.arange(self.llama_config.hidden_size)[-self.attn_config.reserved_dim:]
        elif self.attn_config.compress_method == "cut-head-prefix":
            head_dim = self.llama_config.hidden_size // self.llama_config.num_attention_heads
            reserved_head_dim = self.attn_config.reserved_dim // self.llama_config.num_attention_heads
            assert reserved_head_dim > 0, "reserved_dim must greater than 0"
            self.reserved_dim_idx = torch.cat([torch.arange(head_dim)[:reserved_head_dim] + i * head_dim for i in range(self.llama_config.num_attention_heads)])
        elif self.attn_config.compress_method == "cut-head-suffix":
            head_dim = self.llama_config.hidden_size // self.llama_config.num_attention_heads
            reserved_head_dim = self.attn_config.reserved_dim // self.llama_config.num_attention_heads
            assert reserved_head_dim > 0, "reserved_dim must greater than 0"
            self.reserved_dim_idx = torch.cat([torch.arange(head_dim)[-reserved_head_dim:] + i * head_dim for i in range(self.llama_config.num_attention_heads)])
        else:
            self.reserved_dim_idx = None

        # self.q_proj_weight = None
        # self.k_proj_weight = None
        # self.v_proj_weight = None
        # self.o_proj_weight = None
        # self.real_kproj_pseudo_inv = None
        # self.real_vproj_pseudo_inv = None
        self.multi_right_matrix = None
        
        # 增量PCA需要初始化
        if self.attn_config.compress_method == "incrementalpca":
            from sklearn.decomposition import IncrementalPCA
            self.pca_model = IncrementalPCA(n_components=self.attn_config.reserved_dim)

        # 随机投影矩阵初始化投影矩阵
        if self.attn_config.compress_method == "proj":
            random_matrix = torch.randn(self.llama_config.hidden_size, self.attn_config.reserved_dim, dtype=torch.float)
            q, _ = torch.linalg.qr(random_matrix)
            self.projection_matrix = q
            self.projection_matrix_pseudo_inv = torch.pinverse(self.projection_matrix)
    
    #检查init中的参数错误
    def get_avail_method(self, method_name: str, available_methods: dict, raise_error=True):
        method_name = method_name.lower()
        if method_name not in available_methods and raise_error:
            raise ValueError(f"Method '{method_name}' is not available. Available methods: {', '.join(available_methods.keys())}")
        return available_methods.get(method_name, None)


    @torch.no_grad()
    def update_layer_info(self, q_proj_weight, k_proj_weight, v_proj_weight):
        # if self.k_proj_weight is not k_proj_weight:
        if self.multi_right_matrix is None and self.attn_config.compress_method.startswith('cut'):
            # self.q_proj_weight = q_proj_weight
            # self.k_proj_weight = k_proj_weight
            # self.v_proj_weight = v_proj_weight

            new_k_proj_weight = k_proj_weight[self.reserved_dim_idx, :].float()
            new_k_proj_weight_inv = torch.pinverse(new_k_proj_weight).to(k_proj_weight.device).to(k_proj_weight.dtype)
            self.multi_right_matrix = torch.mm(k_proj_weight, new_k_proj_weight_inv).transpose(0, 1)

    # 存储方式函数
    def all_storage(self, key_states: torch.Tensor, value_states: torch.Tensor, position_ids: torch.LongTensor):
        # if self.key_cache is None:
        #     self.key_cache = key_states
        #     self.value_cache = value_states
        #     self.position_ids_cache = position_ids
        # else:
        #     self.key_cache = torch.concat([self.key_cache, key_states], dim=-2)
        #     self.value_cache = torch.concat([self.value_cache, value_states], dim=-2)
        #     self.position_ids_cache = torch.concat([self.position_ids_cache, position_ids], dim=-1)

        # offset标志着下一个进入缓冲区的索引序号
        offset_seq_len = 0

        if self.attn_config.compress_range == "all":
            key_states = self.compress_function(key_states, update_state=True)

        # 如果缓存连最开始的start_size都没满
        if self.attn_config.start_size > 0 and offset_seq_len < key_states.shape[-2] and (self.key_cache[0] is None or self.key_cache[0].shape[-2] < self.attn_config.start_size):
            cat_seq_len = min(key_states.shape[-2] - offset_seq_len, self.attn_config.start_size - self.get_cached_size())
            
            if self.key_cache[0] == None:
                self.key_cache[0] = key_states[..., offset_seq_len:offset_seq_len+cat_seq_len, :]
                self.value_cache[0] = value_states[..., offset_seq_len:offset_seq_len+cat_seq_len, :]
                # self.position_ids_cache[0] = position_ids[..., offset_seq_len:offset_seq_len+cat_seq_len]
            else:
                self.key_cache[0] = torch.cat([self.key_cache[0], key_states[..., offset_seq_len:offset_seq_len+cat_seq_len, :]], dim=-2)
                self.value_cache[0] = torch.cat([self.value_cache[0], value_states[..., offset_seq_len:offset_seq_len+cat_seq_len, :]], dim=-2)
                # self.position_ids_cache[0] = torch.cat([self.position_ids_cache[0],  position_ids[..., offset_seq_len:offset_seq_len+cat_seq_len]], dim=-1)
            offset_seq_len += cat_seq_len

        # 如果startsize装完了，那就不继续了
        if offset_seq_len == key_states.shape[-2]:
            return
        
        if self.attn_config.recent_size > 0:
            # recent实际上相当于一个先进先出的队列
            # 无论之前有没有填充recent，都完全拼接到recent最后，再把recent开头的多余的给mid即可
            if self.key_cache[1] == None:
                self.key_cache[1] = key_states[..., offset_seq_len:, :]
                self.value_cache[1] = value_states[..., offset_seq_len:, :]
                # self.position_ids_cache[1] = position_ids[..., offset_seq_len:]
            else:
                self.key_cache[1] = torch.cat([self.key_cache[1], key_states[..., offset_seq_len:, :]], dim=-2)
                self.value_cache[1] = torch.cat([self.value_cache[1], value_states[..., offset_seq_len:, :]], dim=-2)
                # self.position_ids_cache[1] = torch.cat([self.position_ids_cache[1],  position_ids[..., offset_seq_len:]], dim=-1)

            # 如果recent有多余的部分
            if self.key_cache[1].shape[-2] > self.attn_config.recent_size:
                # 把recent多余部分移到mid的末尾
                remove_recent_head_len = self.key_cache[1].shape[-2] - self.attn_config.recent_size

                # 如果压缩范围是只压缩中间的部分，则移出的部分需要先压缩再移到mid部分中，否则直接移到mid即可
                
                if self.key_cache[2] is None:
                    if self.attn_config.compress_range == 'mid':
                        self.key_cache[2] = self.compress_function(self.key_cache[1][..., :remove_recent_head_len, :], update_state=True)
                    else:
                        self.key_cache[2] = self.key_cache[1][..., :remove_recent_head_len, :]

                    self.value_cache[2] = self.value_cache[1][..., :remove_recent_head_len, :]
                    # self.position_ids_cache[2] = self.position_ids_cache[1][..., :remove_recent_head_len]
                else:
                    if self.attn_config.compress_range == 'mid':
                        self.key_cache[2] = torch.cat([self.key_cache[2], self.compress_function(self.key_cache[1][..., :remove_recent_head_len, :], update_state=True)], dim=-2)
                    else:
                        self.key_cache[2] = torch.cat([self.key_cache[2], self.key_cache[1][..., :remove_recent_head_len, :]], dim=-2)
                    
                    self.value_cache[2] = torch.cat([self.value_cache[2], self.value_cache[1][..., :remove_recent_head_len, :]], dim=-2)
                    # self.position_ids_cache[2] = torch.cat([self.position_ids_cache[2],  self.position_ids_cache[1][..., :remove_recent_head_len]], dim=-1)
                
                # recent移除前端部分
                self.key_cache[1] = self.key_cache[1][..., remove_recent_head_len:, :]
                self.value_cache[1] = self.value_cache[1][..., remove_recent_head_len:, :]
                # self.position_ids_cache[1] = self.position_ids_cache[1][..., remove_recent_head_len:]
        else:
            # 当然，如果recent_size=0，那就直接拼到mid后面
            if self.attn_config.compress_range == 'mid':
                self.key_cache[2] = torch.cat([self.key_cache[2], self.compress_function(key_states[..., offset_seq_len:, :], update_state=True)], dim=-2)
            else:
                self.key_cache[2] = torch.cat([self.key_cache[2], key_states[..., offset_seq_len:, :]], dim=-2)
            self.value_cache[2] = torch.cat([self.value_cache[2], value_states[..., offset_seq_len:, :]], dim=-2)
            # self.position_ids_cache[2] = torch.cat([self.position_ids_cache[2], position_ids[..., offset_seq_len:]], dim=-1)
        
        if self.attn_config.max_storage_mid_size > 0 and self.key_cache[2] is not None and self.key_cache[2].shape[-2] > self.attn_config.max_storage_mid_size:
            self.key_cache[2] = self.key_cache[2][..., -self.attn_config.max_storage_mid_size:, :]
            self.value_cache[2] = self.value_cache[2][..., -self.attn_config.max_storage_mid_size:, :]
            # self.position_ids_cache[2] = self.position_ids_cache[2][..., -self.attn_config.max_storage_mid_size:]

        # self.position_ids_cache = [None] * 3
    def start_recent_storage(self, key_states: torch.Tensor, value_states: torch.Tensor, position_ids: torch.LongTensor):
        # 只存储start和recent部分
        # if self.key_cache is None:
        #     self.key_cache = key_states
        #     self.value_cache = value_states
        #     self.position_ids_cache = position_ids
        # else:
        #     self.key_cache = torch.concat([self.key_cache, key_states], dim=-2)
        #     self.value_cache = torch.concat([self.value_cache, value_states], dim=-2)
        #     self.position_ids_cache = torch.concat([self.position_ids_cache, position_ids], dim=-1)

        # if self.attn_config.recent_size > 0:
        #     self.key_cache = torch.concat([self.key_cache[..., :self.attn_config.start_size, :],
        #                                 self.key_cache[..., self.attn_config.start_size:, :][..., -self.attn_config.recent_size:, :]], dim=-2)
        #     self.value_cache = torch.concat([self.value_cache[..., :self.attn_config.start_size, :],
        #                                     self.value_cache[..., self.attn_config.start_size:, :][..., -self.attn_config.recent_size:, :]], dim=-2)
        # else:
        #     self.key_cache = self.key_cache[..., :self.attn_config.start_size, :]
        #     self.value_cache = self.value_cache[..., :self.attn_config.start_size, :]

        # self.position_ids_cache = torch.concat([self.position_ids_cache[..., :self.attn_config.start_size],
        #                                           self.position_ids_cache[..., self.attn_config.start_size:][..., -self.attn_config.recent_size:]], dim=-1)
        
        # 新方法只需要把all_storage的结果的第三个(mid)设为None即可
        self.all_storage(key_states, value_states, position_ids)
        self.key_cache[2] = None
        self.value_cache[2] = None
        # self.position_ids_cache[2] = None

    def get_cached_size(self):
        cached_size = 0
        for i in range(3):
            if self.key_cache[i] is not None:
                cached_size += self.key_cache[i].shape[-2]
        return cached_size


    # 压缩方式函数
    def none_compress(self, matrix: torch.Tensor, update_state: bool):
        return matrix
    
    def none_decompress(self, matrix: torch.Tensor):
        return matrix

    def cut_compress(self, matrix: torch.Tensor, strategy: str, update_state: bool):
        assert strategy in ['suffix', 'prefix', 'random', 'head-prefix', 'head-suffix'], "strategy must be suffix, prefix, random, head-prefix, head-suffix"

        reserved_dim = self.attn_config.reserved_dim
        if reserved_dim > matrix.shape[-1] or reserved_dim <= 0:
            raise ValueError(f"reserved_dim {reserved_dim} should be less than hidden_dim {matrix.shape[-1]} and greater than 0")

        return matrix[..., self.reserved_dim_idx]

    def cut_decompress(self, matrix: torch.Tensor):
        # assert strategy in ['suffix', 'prefix', 'random', 'head-prefix', 'head-suffix'], "strategy must be suffix, prefix, random, head-prefix, head-suffix"
        if matrix is None:
            return None
        origin_shape = list(matrix.shape)
        origin_shape[-1] = self.llama_config.hidden_size
        new_matrix = torch.zeros(origin_shape, dtype=matrix.dtype, device=matrix.device)
        new_matrix[..., self.reserved_dim_idx] = matrix
        return new_matrix
    
    def cut_decompress_by_kproj(self, matrix: torch.Tensor):
        if matrix is None:
            return None
        self.multi_right_matrix = self.multi_right_matrix.to(matrix.dtype).to(matrix.device)
        return torch.matmul(matrix, self.multi_right_matrix)

    def incremental_pca_compress(self, matrix: torch.Tensor, update_state: bool):
        # 将 Tensor 转换为 NumPy 数组进行 PCA 处理
        reserved_dim = self.attn_config.reserved_dim
        if reserved_dim > matrix.shape[-1] or reserved_dim <= 0:
            raise ValueError(f"reserved_dim {reserved_dim} should be less than hidden_dim {matrix.shape[-1]} and greater than 0")
        if reserved_dim == matrix.shape[-1]:
            return matrix
        
        batch_size, seq_len, hidden_dim = matrix.shape
        matrix_reshaped = matrix.reshape(-1, hidden_dim)
        matrix_reshaped = matrix_reshaped.to('cpu').numpy()
        if update_state or not hasattr(self.pca_model, 'components_'):
            transformed_matrix = self.pca_model.fit_transform(matrix_reshaped)
        else:
            transformed_matrix = self.pca_model.transform(matrix_reshaped)
        return torch.from_numpy(transformed_matrix).to(matrix.device)
    
    def incremental_pca_decompress(self, compressed_matrix: torch.Tensor):  
        # 确保 compressed_matrix 的维度与 PCA 模型的输出维度匹配  
        if compressed_matrix.shape[1] != self.pca_model.n_components_:  
            raise ValueError("The shape of compressed_matrix does not match the number of PCA components.")  
        
        # 将 Tensor 转换为 NumPy 数组进行 PCA 重建  
        compressed_matrix_reshaped = compressed_matrix.to('cpu').numpy()  
        
        # 使用 PCA 模型的主成分来重建原始数据  
        # 注意：这只是一个近似值，因为 PCA 是不可逆的  
        reconstructed_matrix = self.pca_model.inverse_transform(compressed_matrix_reshaped)  
        
        # 将重建的数据转换回 PyTorch Tensor，并返回到原始设备  
        batch_size, seq_len, hidden_dim = (  
            compressed_matrix.shape[0] * (compressed_matrix.shape[1] // self.attn_config.reserved_dim),  
            compressed_matrix.shape[1] // self.attn_config.reserved_dim,  
            self.attn_config.reserved_dim  
        )  
        reconstructed_matrix = torch.from_numpy(reconstructed_matrix).reshape(batch_size, seq_len, hidden_dim).to(compressed_matrix.device)  
        
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
        self.storage_function(key_states_reshaped, value_states_reshaped, position_ids)

    # 相似度计算函数
    def dot_similarity(self, q: torch.Tensor, k: torch.Tensor):
        """返回是一个[batch_size * k_cache_size]的矩阵"""
        return torch.einsum("bie,bje->bje", q, k).sum(2)

    # 取临近的块函数
    def fetch_chunk(self, indices, selected_len:int):
        if self.key_cache[0].shape[0] != 1 and self.attn_config.span_chunk_size > 1:
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
    
    # normal检索方式
    def normal_retrieve_in_mid_keys(self, q: torch.Tensor, key_cache, value_cache):
        # topk
        # 如果缓存为空或者长度不够，则不需要检索，直接返回缓存
        if key_cache is None or key_cache.shape[-2] < self.attn_config.mid_size:
            return key_cache, value_cache
        
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
        if self.get_cached_size() == 0:
            return query_states, None, None
        # query_shape = [batch_size, head_num, query_seq_len, hidden_dim]
        # 首先修正形状
        query_states_reshaped = einops.rearrange(query_states, 'b h s d -> b s (h d)')
        # 首先均对query进行压缩，然后进行相似度检索，从midkey中找到相似的key
        # 然后有两种做法：1. 对这些key进行还原，并且返回 2.直接返回压缩的query和key
        query_states_reshaped = self.compress_function(query_states_reshaped, update_state=False)
        key_cache, value_cache = self.retrieve_function(query_states_reshaped, self.key_cache[2], self.value_cache[2])
        if self.attn_config.recover:
            if self.decompress_function == None:
                raise NotImplementedError(f"Not implemented Recover yet with compose method {self.attn_config.compress_method}")
            key_cache = self.decompress_function(key_cache)
            # value_cache = self.decompress_function(value_cache)
            
            key_cache_list = [x for x in [self.key_cache[0], key_cache, self.key_cache[1]] if x is not None]
            value_cache_list = [x for x in [self.value_cache[0], value_cache, self.value_cache[1]] if x is not None]
            if key_cache_list == []:
                key_cache = None
                value_cache = None
                # print(f"query_length = {query_states.shape[-2]}, fetched_length = {0}")
            else:
                key_cache = torch.cat(key_cache_list, dim=-2)
                value_cache = torch.cat(value_cache_list, dim=-2)
                key_cache = einops.rearrange(key_cache, 'b s (h d) -> b h s d', h=self.llama_config.num_attention_heads)
                value_cache = einops.rearrange(value_cache, 'b s (h d) -> b h s d', h=self.llama_config.num_attention_heads)
                # print(f"query_length = {query_states.shape[-2]}, fetched_length = {key_cache.shape[-2]}")
            return query_states, key_cache, value_cache
                
        else:
            raise NotImplementedError("Not implemented unrecover yet")
    
    # 清空cache
    def clean_cache(self):
        self.key_cache = [None] * 3
        self.value_cache = [None] * 3
        # self.position_ids_cache = None
        torch.cuda.empty_cache()