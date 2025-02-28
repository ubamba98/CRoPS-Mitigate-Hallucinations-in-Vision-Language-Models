from registry.crops_method import patch_everything
patch_everything()

from generation_configs.crops_generation_config import GenerationConfigCRoPS
from constants.crops_constants import (
    DEFAULT_LAMBDA_LANG_PRIOR,  
    DEFAULT_ALPHA_STAT_BIAS,
    DEFAULT_BETA_CUTOFF,
    DEFAULT_MAX_THRESHOLD_PLAUSIBILITY_CONSTRAINT,
    DEFAULT_AGGREGATE_LAYER_FAST_V,
    DEFAULT_MINUMUM_FAST_V_TOKENS,
    DEFAULT_AGGREGATE_LAYER_TEXT_MASK,
    DEFAULT_MINIMUM_TEXT_TOKENS
)

from constants.default_generation_constants import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K
)

from utils.reproducibility_utilt import set_reproducibility

import torch
import argparse
from transformers import AutoModelForImageTextToText, AutoProcessor

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")

    # Generation config
    parser.add_argument("--do_sample", action='store_true',  default=True)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    # CRoPS config
    parser.add_argument("--lambda_lang_prior", type=float, default=DEFAULT_LAMBDA_LANG_PRIOR)
    parser.add_argument("--alpha_stat_bias", type=float, default=DEFAULT_ALPHA_STAT_BIAS)
    parser.add_argument("--beta_cutoff", type=float, default=DEFAULT_BETA_CUTOFF)
    parser.add_argument("--max_threshold_plausibility_constraint", type=float, default=DEFAULT_MAX_THRESHOLD_PLAUSIBILITY_CONSTRAINT)
    parser.add_argument("--aggregate_layer_fast_v", type=int, default=DEFAULT_AGGREGATE_LAYER_FAST_V)
    parser.add_argument("--minumum_fast_v_tokens", type=int, default=DEFAULT_MINUMUM_FAST_V_TOKENS)
    parser.add_argument("--aggregate_layer_text_mask", type=int, default=DEFAULT_AGGREGATE_LAYER_TEXT_MASK)
    parser.add_argument("--minimum_text_tokens", type=int, default=DEFAULT_MINIMUM_TEXT_TOKENS)

    # Evaluation config
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def main():
    args = args_parser()
    set_reproducibility(args.seed)

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Load image
    image = Image.open(args.image_path)


if __name__ == "__main__":
    main()
