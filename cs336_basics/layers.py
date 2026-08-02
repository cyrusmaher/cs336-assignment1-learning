# Student implementation placeholder. Fill this in by working through the model layer sections.
from torch import nn
import torch
import einx

def init_linear(out_features, in_features, device=None, dtype=None):
    init_scale = 2 / (in_features + out_features)
    return (
        (torch.randn(out_features, in_features, device=device, dtype=dtype) * init_scale)
        .clip(-3, 3)
        .to(device=device, dtype=dtype)
    )

def init_embedding(num_embeddings, embedding_dim, device=None, dtype=None):
    return (
        torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        .clip(-3, 3)
        .to(device=device, dtype=dtype)
    )

def init_rmsnorm(d_model, device=None, dtype=None):
    return (torch.ones(d_model, device=device, dtype=dtype)).to(device=device, dtype=dtype)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
    in_features: int final dimension of the input
    out_features: int final dimension of the output
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            init_linear(out_features, in_features, device, dtype)
        )

    def forward(self, x: torch.Tensor):
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
    num_embeddings: int number of embeddings
    embedding_dim: int dimension of the embeddings
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.embedding = nn.Parameter(
            init_embedding(num_embeddings, embedding_dim, device, dtype)
        )
    
    def forward(self, tokens):
        return self.embedding[tokens]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            init_rmsnorm(d_model, device, dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        result = (x / rms_norm) * self.weight
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        """
        d_model: int Hidden dimension of the model
        d_ff: int Intermediate dimension of the model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        # w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def glu(self, x, W1, W2):
        return torch.sigmoid(x @ W1.T) * (x @ W2.T)
    
    def silu(self, x):
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check input shape: should be (..., d_model)
        assert x.shape[-1] == self.d_model, f"Input features last dim {x.shape[-1]} != d_model {self.d_model}"
        
        # Check parameter shapes
        assert self.W1.weight.shape == (self.d_ff, self.d_model), f"W1 shape {self.W1.weight.shape} != (d_ff, d_model) ({self.d_ff}, {self.d_model})"
        assert self.W2.weight.shape == (self.d_model, self.d_ff), f"W2 shape {self.W2.weight.shape} != (d_model, d_ff) ({self.d_model}, {self.d_ff})"
        assert self.W3.weight.shape == (self.d_ff, self.d_model), f"W3 shape {self.W3.weight.shape} != (d_ff, d_model) ({self.d_ff}, {self.d_model})"

        # Compute SwiGLU: (SiLU(x @ W1.T) * (x @ W3.T)) @ W2.T
        x_w1 = self.W1(x)  # (..., d_ff)
        x_w3 = self.W3(x)  # (..., d_ff)
        out = self.silu(x_w1) * x_w3         # (..., d_ff)
        out = self.W2(out)  # (..., d_model)
        return out

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be input
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.register_rotation_buffer()

    def register_rotation_buffer(self):
        # RoPE implementations typically precompute cos and sin matrices of size (max_seq_len, d_k // 2)
        # For each position, you want cos and sin "angles" that will be applied to blocks of the input vector
        position = torch.arange(self.max_seq_len, device=self.device).float()
        
        # RoPE uses log space for theta frequencies (as in GPT-NeoX, phi2, etc)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device).float() / self.d_k))
        angle = torch.einsum('i,j->ij', position, inv_freq)  # (max_seq_len, d_k // 2)
        cos = torch.cos(angle)  # (max_seq_len, d_k // 2)
        sin = torch.sin(angle)  # (max_seq_len, d_k // 2)
        rotation = torch.stack(
            (
                torch.stack((cos, -sin), dim=-1),
                torch.stack((sin,  cos), dim=-1),
            ),
            dim=-2,
        )

        self.register_buffer(
            "rotation_cached",
            rotation,
            persistent=False,
        )
 
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        """
        Process
        an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        that you should tolerate 𝑥 with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        𝑥 along the sequence dimension.
        """
        assert x.shape[-1] == self.d_k, f"d_k {x.shape[-1]} != {self.d_k}"
        
        x_pairs = einx.id(
            "... seq (groups input) -> ... seq groups input",
            x,
            input=2,
        )

        rotated_pairs_einx = torch.einsum(
            "... sgi, ... sgoi -> ... sgo",
            x_pairs,
            self.rotation_cached[token_positions].to(dtype=x.dtype),
        )

        rotated_pairs_reshaped = einx.id(
            "... seq groups output -> ... seq (groups output)",
            rotated_pairs_einx,
        )
        return rotated_pairs_reshaped
        
def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    x_scaled = x - x.max(dim=axis, keepdim=True).values
    x_scaled_exp = torch.exp(x_scaled)
    return x_scaled_exp / x_scaled_exp.sum(dim=axis, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    prod = einx.dot(
        "... queries [d], ... keys [d] -> ... queries keys",
        Q,
        K,
    ) / (Q.shape[-1] ** 0.5)
    
    if mask is not None:
        masked_prod = prod.masked_fill(~mask, -float('inf'))
    else:
        masked_prod = prod

    return softmax(masked_prod, axis=-1) @ V

def rearrange_multihead(x: torch.Tensor, num_heads: int, d_head: int) -> torch.Tensor:
    return einx.id(
        "... seq (num_heads d_head) -> ... num_heads seq d_head",
        x,
        num_heads=num_heads,
        d_head=d_head,
    )
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, rope=None):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} is not divisible by num_heads {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.device = device
        self.dtype = dtype
        self.rope = rope

        self.Wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wo = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q_multihead = rearrange_multihead(Q, self.num_heads, self.d_head)
        K_multihead = rearrange_multihead(K, self.num_heads, self.d_head)
        V_multihead = rearrange_multihead(V, self.num_heads, self.d_head)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            Q_multihead_rope = self.rope(Q_multihead, token_positions)
            K_multihead_rope = self.rope(K_multihead, token_positions)
        else:
            Q_multihead_rope = Q_multihead
            K_multihead_rope = K_multihead
        
        attn = scaled_dot_product_attention(Q_multihead_rope, K_multihead_rope, V_multihead, mask)

        attn_wide = einx.id(
            "... num_heads seq d_head -> ... seq (num_heads d_head)",
            attn,
        )

        return self.Wo(attn_wide)
            
            
class CausalMultiHeadSelfAttention(MultiHeadSelfAttention):
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2], dtype=bool, device=x.device))
        return super().forward(x, mask, token_positions)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None, rope=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.rope = rope
        self.attention = CausalMultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype, rope=rope)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rms1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms2 = RMSNorm(d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        x = x + self.attention(self.rms1(x), token_positions)
        x = x + self.ffn(self.rms2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        """
        At minimum, your implementation should accept all the
        aforementioned construction parameters for the Transformer block, as well as these additional
        parameters:
        vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the
        token embedding matrix.
        26
        context_length: int The maximum context length, necessary for determining the dimensionality
        of the RoPE sin and cos buffer.
        num_layers: int The number of Transformer blocks to use.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype
        
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.pos_embedding = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype, 
                             rope=self.pos_embedding) for _ in range(num_layers)]
        )
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        logits = self.linear(self.final_norm(x))

        return logits