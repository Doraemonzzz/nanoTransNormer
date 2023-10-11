import math
import inspect
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from .lightning_attention import lightning_attention

##### for print module
def print_module(module):
    named_modules = set()
    for p in module.named_modules():
        named_modules.update([p[0]])
    named_modules = list(named_modules)

    string_repr = ''
    for p in module.named_parameters():
        name = p[0].split('.')[0]
        if name not in named_modules:
            string_repr = string_repr + '('+ name +'): ' \
                +'Tensor(' + str(tuple(p[1].shape))+ ', requires_grad='+ str(p[1].requires_grad) +')\n'

    return string_repr.rstrip("\n")
##### for print module

##### Linearized Relative Positional Encoding: https://openreview.net/forum?id=xoLyps2qWc&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DTMLR%2FAuthors%23your-submissions)
class Lrpe(nn.Module):
    def __init__(
        self,
        num_heads=8,
        embed_dim=64,
    ):
        super().__init__()
        d = num_heads * embed_dim
        
        self.index = torch.empty(0)
        self.theta = nn.Parameter(10000**(-2 / d *
                                                torch.arange(d)).reshape(
                                                    num_heads, 1, -1))

    def extra_repr(self):
        return print_module(self)

    def forward(self, x, offset=0):
        # x: b, h, n, d
        # offset: for k, v cache
        n = x.shape[-2]
        if self.index.shape[0] < n:
            self.index = torch.arange(n).reshape(1, -1, 1).to(x)
        index = self.index[:, :n] + offset
        theta = self.theta * index
        x = torch.concat([x * torch.cos(theta), x * torch.sin(theta)], dim=-1)

        return x

##### Simple Gated Linear Unit: https://arxiv.org/pdf/2307.14995.pdf
class SGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        d1 = config.n_embd
        d2 = 2 * d1
        bias = config.bias
        self.l1 = nn.Linear(d1, d2, bias=bias)
        self.l2 = nn.Linear(d1, d2, bias=bias)
        self.l3 = nn.Linear(d2, d1, bias=bias)

    def forward(self, x):
        o1 = self.l1(x)
        o2 = self.l2(x)
        output = o1 * o2
        output = self.l3(output)

        return output
    
##### srmsnorm: https://arxiv.org/pdf/2307.14995.pdf
def get_srmsnorm(use_triton):
    if use_triton:
        print("Use lightning attention and triton version srmsnorm!!!")
        from .srms_triton import SimpleRMSNorm
        norm = SimpleRMSNorm()
    else:
        print("Use naive linear attention and naive srmsnorm!!!")
        from .srms import SimpleRMSNorm
        norm = SimpleRMSNorm()
        
    return norm
        

##### Norm Linear Attention: https://arxiv.org/pdf/2210.10340.pdf
def linear_attention(q, k, v, attn_mask):
    energy = torch.einsum("... n d, ... m d -> ... n m", q, k)
    energy = energy * attn_mask
    output = torch.einsum("... n m, ... m d -> ... n d", energy, v)

    return output
    
class NormLinearAttention(nn.Module):

    def __init__(
        self,
        config,
        use_lrpe=True,
    ):
        super().__init__()

        hidden_dim = config.n_embd
        bias = config.bias
        self.use_triton = config.use_triton
        self.n_head = config.n_head
        self.use_lrpe = use_lrpe
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.qkvu_proj = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias)
        if self.use_lrpe:
            self.lrpe = Lrpe(num_heads=self.n_head, embed_dim=hidden_dim//self.n_head)
        self.act = F.silu
        self.norm = get_srmsnorm(self.use_triton)

        # for inference only
        self.offset = 0

    def forward(
        self,
        x,
        attn_mask=None,  # (b, h, n, m)
        attn_padding_mask=None,  # (b, m)
        past_key_value=None,
        slope_rate=None,
    ):
        if not self.training:
            return self.inference(
                x, attn_mask=attn_mask, attn_padding_mask=attn_padding_mask,
                past_key_value=past_key_value, slope_rate=slope_rate
                
            )
        # x: b n d
        n = x.shape[-2]
        # linear map
        q, k, v, u = self.qkvu_proj(x).chunk(4, dim=-1)
        # reshape
        q, k, v = map(
            lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_head),
            [q, k, v])
        # act
        q = self.act(q)
        k = self.act(k)

        # lrpe
        if self.use_lrpe:
            offset = 0
            q = self.lrpe(q, offset=offset)
            k = self.lrpe(k, offset=offset)
        
        # treat for padding token
        if attn_padding_mask is not None:
            v = v.masked_fill(
                (1 - attn_padding_mask).unsqueeze(1).unsqueeze(-1).to(torch.bool),
                0
            )

        # compute
        if attn_mask == None:
            attn_mask = (torch.tril(torch.ones(n, n))).to(q)
            
        if slope_rate != None:
            attn_mask = torch.exp(slope_rate * attn_mask)
        
        if self.use_triton:
            output = lightning_attention(q, k, v, True,
                                         slope_rate.squeeze(-1).squeeze(-1))
        else:
            output = linear_attention(q, k, v, attn_mask)

        # reshape
        output = rearrange(output, 'b h n d -> b n (h d)')
        # normalize
        output = self.norm(output)
        
        # gate
        output = u * output
        # outproj
        output = self.out_proj(output)

        return output, past_key_value

    def inference(
        self,
        x,
        attn_mask=None,  # (b, h, n, m)
        attn_padding_mask=None,  # (b, m)
        past_key_value=None,
        slope_rate=None,
    ):
        # x: b n d
        n = x.shape[-2]
        # linear map
        q, k, v, u = self.qkvu_proj(x).chunk(4, dim=-1)
        # reshape
        q, k, v = map(
            lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_head),
            [q, k, v])
        # act
        q = self.act(q)
        k = self.act(k)
        
        # lrpe
        if self.use_lrpe:
            q = self.lrpe(q, offset=self.offset)
            k = self.lrpe(k, offset=self.offset)
            
            # get offset of q
            if past_key_value == None:
                self.offset = q.shape[-2]
            else:
                self.offset += 1

        # get decay
        ratio = torch.exp(-slope_rate)
        
        if past_key_value == None:
            # only use for the first time
            if attn_mask == None:
                attn_mask = (torch.tril(torch.ones(n, n))).to(q)

            if slope_rate != None:
                attn_mask = torch.exp(slope_rate * attn_mask)

            # treat for padding token
            if attn_padding_mask is not None:
                v = v.masked_fill(
                    (1 - attn_padding_mask).unsqueeze(1).unsqueeze(-1).to(torch.bool),
                    0
                )
            
            # for inference, we use torch version since there is a bug in triton version
            output = linear_attention(q, k, v, attn_mask)
            
            # only compute kv when generate
            do_generate = os.environ.get('do_generate', default=False)
            if do_generate:
                # compute kv
                # b, h, n, e, d
                kv_outproduct = torch.einsum('... n e, ... n d -> ... n e d', k, v)
                # 1, 1, n, 1, 1
                index = torch.arange(n - 1, -1, -1).reshape(1, 1, -1, 1, 1).to(x)
                # (h, 1, 1) -> (1, h, 1, 1, 1); (1, h, 1, 1, 1), (1, 1, n, 1, 1) -> (1, h, n, 1, 1)
                decay = ratio.unsqueeze(0).unsqueeze(-1)**index
                kv_outproduct_with_decay = kv_outproduct * decay
                kv = torch.sum(kv_outproduct_with_decay, dim=-3)
            else:
                kv = None
        else:
            kv = ratio * past_key_value + torch.einsum('... n d, ... n e -> ... d e',
                                            k, v)
            output = torch.einsum('... n e, ... e d -> ... n d',
                                q, kv)

        # reshape
        output = rearrange(output, 'b h n d -> b n (h d)')
        # normalize
        output = self.norm(output)
        # gate
        output = u * output
        # outproj
        output = self.out_proj(output)

        return output, kv


class Block(nn.Module):

    def __init__(self, config, use_lrpe=True):
        super().__init__()
        # token mixer
        self.token_mixer = NormLinearAttention(config, use_lrpe=use_lrpe)
        self.token_norm = get_srmsnorm(config.use_triton)
        # channel mixer
        self.channel_mixer = SGLU(config)
        self.channel_norm = get_srmsnorm(config.use_triton)
        
    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        attn_mask=None,  # (b, h, n, m)
        attn_padding_mask=None,  # (b, m)
        past_key_value=None,
        slope_rate=None,
    ):
        # token mixer
        residual = x
        x, present_key_value = self.token_mixer(
            x=self.token_norm(x),
            attn_mask=attn_mask,
            attn_padding_mask=attn_padding_mask,
            past_key_value=past_key_value,
            slope_rate=slope_rate,
        )
        x = self.residual_connection(x, residual)

        # channel mixer
        residual = x
        x = self.channel_mixer(self.channel_norm(x))
        x = self.residual_connection(x, residual)

        outputs = (x, present_key_value)

        return outputs

@dataclass
class TransNormerConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False
    use_triton: bool = True

class TransNormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_layer = config.n_layer

        self.transnormer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, use_lrpe=_ > 0) for _ in range(config.n_layer)]), # only use lrpe in the first layer
            ln_f = get_srmsnorm(config.use_triton),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transnormer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # mask
        self._attn_mask = torch.empty(0)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                
        # h, 1, 1
        self.slopes = self._build_slopes_tensor(config.n_head)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_key_values=[]):
        _attn_mask = self._prepare_attn_mask(idx)

        # forward the TransNormer model itself
        x = self.transnormer.wte(idx) # token embeddings of shape (b, n, n_embd)
        if len(past_key_values) == 0:
            past_key_values = [None for _ in range(self.n_layer)]
        new_past_key_values = []
        origin_slope_rate = self.slopes.to(x.device)
        for i, block in enumerate(self.transnormer.h):
            slope_rate = origin_slope_rate * (1 - i / (self.n_layer - 1) + 1e-5)
            x, new_past_key_value = block(x, attn_mask=_attn_mask, past_key_value=past_key_values[i], slope_rate=slope_rate)
            new_past_key_values.append(new_past_key_value)
        x = self.transnormer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, new_past_key_values
    
    @staticmethod
    def _build_slopes_tensor(n_attention_heads: int):

        def get_slopes(n):

            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n
                )  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(
                    math.log2(n)
                )  # when the number of heads is not a power of 2, we use this workaround.
                return (get_slopes_power_of_2(closest_power_of_2) + get_slopes(
                    2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        # h, 1, 1
        slopes = torch.tensor(get_slopes(n_attention_heads)).reshape(
            n_attention_heads, 1, 1)

        return slopes
    
    def _prepare_attn_mask(self, x):
        b, n = x.shape

        if self._attn_mask.shape[-1] < n:
            def get_mask(n):
                mask = torch.triu(
                    torch.zeros(n, n).float().fill_(float("-inf")), 1)
                # -n, ..., -2, -1, 0
                for i in range(n):
                    x = torch.arange(i + 1)
                    mask[i, :i + 1] = -torch.flip(x, [0])

                return mask

            arr = []
            for slope in self.slopes:
                arr.append(get_mask(n))
            self._attn_mask = torch.stack(arr, dim=0).to(x)

        _attn_mask = self._attn_mask[:, :n, :n]
        num_heads = _attn_mask.shape[0]

        return _attn_mask[None, :, :, :].expand(b, num_heads, n, n)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    # need change
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        past_key_values = [None for _ in range(self.n_layer)]
        output = idx
        for _ in range(max_new_tokens):
            idx_cond = idx
            # forward the model to get the logits for the index in the sequence
            logits, _, past_key_values = self(idx_cond, past_key_values=past_key_values)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = idx_next
            output = torch.cat((output, idx), dim=1)

        return output
