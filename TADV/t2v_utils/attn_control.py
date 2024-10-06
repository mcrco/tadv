import torch
import inspect
import math
from diffusers.models.unets.unet_3d_condition import UNet3DConditionModel
from typing import Optional, Dict
from diffusers.models.attention_processor import Attention
from diffusers.utils.deprecation_utils import deprecate
from diffusers.utils import logging

class AttentionController():
    def __init__(self):
        self.attn_dict = dict()

    def reset(self):
        self.attn_dict = dict()

    def forward(self, attn_key, attn):
        if attn_key in self.attn_dict:
            raise Exception('Attention already registered for key! Perhaps you forgot to reset?')
        self.attn_dict[attn_key] = attn

    def __call__(self, attn_key, attn):
        self.forward(attn_key, attn)

    def process_attns(self):
        cross_attns = dict()
        temp_attns = dict()
        for attn_key, attn in self.attn_dict.items():
            if 'cross' in attn_key:
                attns = cross_attns
                attn = attn.permute(0, 2, 1)
            elif 'temp' in attn_key:
                attns = temp_attns
                attn = attn.permute(1, 2, 0)
            else:
                raise Exception(f"Invalid attention key: {attn_key}")

            attn_dim = attn.shape[-1]
            if attn_dim not in attns:
                attns[attn_dim] = []
            attns[attn_dim].append(attn)

        return cross_attns, temp_attns

    def get_avg_attns(self):
        cross_attns, temp_attns = self.process_attns()

        avg_cross_attns = dict()
        for attn_dim, attns in cross_attns.items():
            avg_cross_attns[attn_dim] = torch.stack(attns).mean(0)

        avg_temp_attns = dict()
        for attn_dim, attns in temp_attns.items():
            avg_temp_attns[attn_dim] = torch.stack(attns).mean(0)

        return avg_cross_attns, avg_temp_attns

def register_attn_hook(unet: UNet3DConditionModel, attention_store: AttentionController):
    # modify torch.nn.functional.scaled_dot_product_attention to extract ca_maps
    def get_custom_attn_forward(attn_key):
        def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
            L, S = query.size(-2), key.size(-2)
            scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
            attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
            if is_causal:
                assert attn_mask is None
                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                attn_bias.to(query.dtype)

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                else:
                    attn_bias += attn_mask
            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight += attn_bias
            attn_weight = torch.softmax(attn_weight, dim=-1)

            ret_attn = attn_weight.mean(1) # average across heads
            attention_store(attn_key, ret_attn)

            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            return attn_weight @ value

        def attn_processor_call(
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            if len(args) > 0 or kwargs.get("scale", None) is not None:
                deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                deprecate("scale", "1.0.0", deprecation_message)

            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

        def attn_forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **cross_attention_kwargs,
        ) -> torch.Tensor:
            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
            quiet_attn_parameters = {"ip_adapter_masks"}
            unused_kwargs = [
                k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
            ]
            if len(unused_kwargs) > 0:
                logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
                logger.warning(
                    f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
                )
            cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

            return attn_processor_call(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

        return attn_forward

    down_cross_count = 0
    down_temp_count = 0
    for downsample_block in unet.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            for attn_block in downsample_block.attentions: # transformer2d
                trans_block = attn_block.transformer_blocks[0]  # BasicTransformerBlock
                cross_attn = trans_block.attn2
                custom_forward = get_custom_attn_forward('down_cross_' + str(down_cross_count))
                cross_attn.forward = custom_forward.__get__(cross_attn, Attention)
                down_cross_count += 1
            for attn_block in downsample_block.temp_attentions: # transformer2d
                trans_block = attn_block.transformer_blocks[0]  # BasicTransformerBlock
                temp_attn = trans_block.attn2
                custom_forward = get_custom_attn_forward('down_temp_' + str(down_temp_count))
                temp_attn.forward = custom_forward.__get__(temp_attn, Attention)
                down_temp_count += 1

    mid_cross_count = 0
    mid_temp_count = 0
    mid_block = unet.mid_block
    for attn_block in mid_block.attentions:  # transformer2d
        trans_block = attn_block.transformer_blocks[0]  # BasicTransformerBlock
        cross_attn = trans_block.attn2
        custom_forward = get_custom_attn_forward('mid_cross_' + str(mid_cross_count))
        cross_attn.forward = custom_forward.__get__(cross_attn, Attention)
        mid_cross_count += 1
    for attn_block in mid_block.temp_attentions:  # transformer2d
        trans_block = attn_block.transformer_blocks[0]  # BasicTransformerBlock
        temp_attn = trans_block.attn2
        custom_forward = get_custom_attn_forward('mid_temp_' + str(mid_temp_count))
        temp_attn.forward = custom_forward.__get__(temp_attn, Attention)
        mid_temp_count += 1

    up_cross_count = 0
    up_temp_count = 0
    for upsample_block in unet.up_blocks:
        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            for attn_block in upsample_block.attentions: # transformer2d
                trans_block = attn_block.transformer_blocks[0]  # BasicTransformerBlock
                cross_attn = trans_block.attn2
                custom_forward = get_custom_attn_forward('up_cross_' + str(up_cross_count))
                cross_attn.forward = custom_forward.__get__(cross_attn, Attention)
                up_cross_count += 1
            for attn_block in upsample_block.temp_attentions: # transformer2d
                trans_block = attn_block.transformer_blocks[0]  # BasicTransformerBlock
                temp_attn = trans_block.attn2
                custom_forward = get_custom_attn_forward('up_temp_' + str(up_temp_count))
                temp_attn.forward = custom_forward.__get__(temp_attn, Attention)
                up_temp_count += 1

    print(f"{down_cross_count} down cross attentions.")
    print(f"{down_cross_count} down temp attentions.")
    print(f"{mid_cross_count} mid cross attentions.")
    print(f"{mid_cross_count} mid cross attentions.")
    print(f"{up_cross_count} down temp attentions.")
    print(f"{up_cross_count} down temp attentions.")
