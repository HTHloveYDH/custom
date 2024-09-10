from dataclasses import dataclass


@dataclass  
class DeepseekConfig:  
    d_model: int = 1024
    num_heads: int = 16
    head_dim: int = 128
    dropout: int = 0.0
    q_lora_rank: int = None
    qk_rope_head_dim: int = 64
    kv_lora_rank: int = 512
    v_head_dim: int = 128
    qk_nope_head_dim: int = 1024
    vocab_size: int = 102400