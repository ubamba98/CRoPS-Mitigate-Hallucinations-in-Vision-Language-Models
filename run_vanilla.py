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
from tqdm.auto import tqdm
from accelerate import PartialState
from accelerate.utils import gather_object
from transformers import (
    AutoModelForImageTextToText, 
    AutoProcessor, 
    GenerationConfig, 
    BitsAndBytesConfig
)
from transformers import logging
logging.set_verbosity_error()

distributed_state = PartialState()

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--load_in_8bit", action='store_true', default=False)
    parser.add_argument("--load_in_4bit", action='store_true', default=False)

    # Generation config
    parser.add_argument("--do_sample", action='store_true',  default=True)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    # Evaluation config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, required=True)

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
    if args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
        )
    else:
        bnb_config = None

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16,
        device_map={"": distributed_state.device},
        quantization_config=bnb_config
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_name)

    if args.run_chair_benchmark:
        run_chair_benchmark(model, processor, args)

def run_chair_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "vanilla", args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)

    chair_benchmark = ChairBenchmarkDataset(
            coco_path=args.coco_path,
            coco_file=args.coco_file,
            base_image_path=args.coco_base_image_path,
            chair_test_size=args.chair_test_size
        )
    
    with distributed_state.local_main_process_first():
        test_dataset = chair_benchmark.get_test_dataset()

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

            generation_config = GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                temperature=args.temperature,
                do_sample=args.do_sample,
                use_cache=True,
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