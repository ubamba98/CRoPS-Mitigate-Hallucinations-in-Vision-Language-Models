from methods.samplers.sid_sample import patch_sid_sampling  
from methods.model_forward.sid_llama_forward import patch_llama_forward
from methods.model_forward.sid_qwen_forward import patch_qwen_forward

def patch_everything():
    patch_sid_sampling()
    patch_llama_forward()
    patch_qwen_forward()