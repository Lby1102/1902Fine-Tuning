import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
import config

def load_model():
    '''
    load model and do fine tuning with lora method
    '''
    model=Qwen2_5_VLForConditionalGeneration.form_pretrained(
        config.model_dir,
        torch_dtype=torch.bfloar16,
        device_map='auto'
    )

    #LoRA settings
    lora_config=LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        e=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            'q_proj','k_proj','v_proj','o_proj'
        ]
    )

    model=get_peft_model(model,lora_config)
    model.print_trainable_parameters()
    return model