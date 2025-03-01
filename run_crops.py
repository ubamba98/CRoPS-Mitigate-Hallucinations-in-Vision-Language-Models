from methods.crops_method import patch_everything
patch_everything()

from methods.generation_configs.crops_generation_config import GenerationConfigCRoPS

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

from benchmark.chair_benchmark import ChairBenchmarkDataset
from utils.reproducibility_util import set_reproducibility

import os
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from accelerate import PartialState
from accelerate.utils import gather_object
from transformers import AutoModelForImageTextToText, AutoProcessor

distributed_state = PartialState()

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")

    # Generation config
    parser.add_argument("--do_sample", action='store_true',  default=True)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
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
    parser.add_argument("--seed", type=int, default=42)

    # Chair benchmark config
    parser.add_argument("--run_chair_benchmark", action='store_true',  default=True)
    parser.add_argument("--coco_path", type=str, default='dataset/annotations')
    parser.add_argument("--coco_file", type=str, default='instances_val2014.json')
    parser.add_argument("--coco_base_image_path", type=str, default='dataset/val2014')
    parser.add_argument("--chair_test_size", type=int, default=500)

    return parser.parse_args()

def main():
    args = args_parser()
    set_reproducibility(args.seed)

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16,
        device_map={"": distributed_state.device}
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_name)

    if args.run_chair_benchmark:
        run_chair_benchmark(model, processor, args)

def run_chair_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "CRoPS")
    os.makedirs(experiment_name, exist_ok=True)

    chair_benchmark = ChairBenchmarkDataset(
            coco_path=args.coco_path,
            coco_file=args.coco_file,
            base_image_path=args.coco_base_image_path,
            chair_test_size=args.chair_test_size
        )
    
    with distributed_state.local_main_process_first():
        test_dataset = chair_benchmark.get_test_dataset()

    conversation_lang_prior = [
            {
                "role":"system",
                "content": [
                    {
                        "type": "text", 
                        "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please describe this image in detail"},
                ],
            },
        ]

    input_ids_lang_prior = processor.apply_chat_template(
            conversation_lang_prior,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(distributed_state.device, torch.bfloat16)
    input_ids_lang_prior = input_ids_lang_prior["input_ids"]

    with distributed_state.split_between_processes(test_dataset) as process_test_dataset:
        generations = []
        for test_image in tqdm(process_test_dataset, desc=f"Running Chair Benchmark. Process: {distributed_state.process_index}"):
            conversation = [
                {
                    "role":"system",
                    "content": [
                        {
                            "type": "text", 
                            "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": test_image["image_path"]},
                        {"type": "text", "text": "Please describe this image in detail"},
                    ],
                },
            ]

            inputs = processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to(distributed_state.device, torch.bfloat16)

            image_tokens = np.where(inputs["input_ids"].cpu().numpy()==32000)[1]
            generation_config = GenerationConfigCRoPS(
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                temperature=args.temperature,
                do_sample=args.do_sample,
                key_position={
                    "image_start": image_tokens[0],
                    "image_end": image_tokens[-1],
                },
                input_ids_lang_prior=input_ids_lang_prior,
                aggregate_layer_fast_v=args.aggregate_layer_fast_v,
                minumum_fast_v_tokens=args.minumum_fast_v_tokens,
                aggregate_layer_text_mask=args.aggregate_layer_text_mask,
                minimum_text_tokens=args.minimum_text_tokens,
                lambda_lang_prior=args.lambda_lang_prior,
                alpha_stat_bias=args.alpha_stat_bias,
                beta_cutoff=args.beta_cutoff,
                max_threshold_plausibility_constraint=args.max_threshold_plausibility_constraint,
                use_cache=False,
            )

            output_ids = model.generate(**inputs, generation_config=generation_config)

            output_text = processor.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)            

            generations.append({
                chair_benchmark.image_id_key: test_image["image_id"],
                chair_benchmark.caption_key: output_text
            })

    generations = gather_object(generations)
    if distributed_state.is_main_process:
        generations_path = os.path.join(experiment_name, "chair_generations.jsonl")
        chair_benchmark.dump_generations(generations, generations_path)
        chair_benchmark.evaluate(generations_path, dump_results=True)

if __name__ == "__main__":
    main()