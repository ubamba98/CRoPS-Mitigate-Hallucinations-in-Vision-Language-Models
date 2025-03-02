from methods.samplers.crops_sample import patch_crops_sampling
from methods.model_forward.crops_llama_forward import patch_llama_forward
from methods.model_forward.crops_qwen_forward import patch_qwen_forward

def patch_everything():
    patch_crops_sampling()
    patch_llama_forward()
    patch_qwen_forward()