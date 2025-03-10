from methods.samplers.m3id_sample import patch_m3id_sampling 
from methods.model_forward.crops_qwen_forward import patch_qwen_forward
from methods.model_forward.crops_llama_forward import patch_llama_forward

def patch_everything():
    patch_m3id_sampling()
    patch_llama_forward()
    patch_qwen_forward()