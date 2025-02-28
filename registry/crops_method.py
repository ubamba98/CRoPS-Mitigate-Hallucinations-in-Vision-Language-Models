from samplers.crops_sample import patch_crops_sampling
from model_forward.crops_llama_forward import patch_llama_forward

def patch_everything():
    patch_crops_sampling()
    patch_llama_forward()