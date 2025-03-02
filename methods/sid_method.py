from methods.samplers.sid_sample import patch_sid_sampling  
from methods.model_forward.crops_llama_forward import patch_llama_forward

def patch_everything():
    patch_sid_sampling()
    patch_llama_forward()