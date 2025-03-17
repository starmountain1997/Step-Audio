from amp.common.patch_transformers import patch_get_class_in_module
from amp.module_patcher import when_imported
from loguru import logger
import time
import torch
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) :
        time_start = time.time()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        logger.info(f"modeling_step1 forward taken: {time.time() - time_start}")

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def patch_step_audio(mod):
    package_name = mod.__name__.split(".")[-1]
    if package_name == "modeling_step1":
        logger.info(f"Patching {package_name}")
        logger.info(f"{mod} is patched.")
        mod.Step1ForCausalLM.forward = forward

@when_imported("transformers")
def patch_step_audio(mod):
    get_class_in_module_patched=patch_get_class_in_module(func=patch_step_audio)
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched