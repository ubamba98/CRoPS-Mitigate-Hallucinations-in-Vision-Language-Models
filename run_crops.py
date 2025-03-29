from methods.crops_method import patch_everything
patch_everything()

from methods.generation_configs.contrastive_generation_config import GenerationConfigContrastive

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

from constants.image_token_constants import BACKBONE_IMAGE_TOKEN_IDS

from constants.default_generation_constants import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K
)

from benchmark.chair_benchmark import ChairBenchmarkDataset
from benchmark.evaluators.mme.utils import parse_pred_ans,eval_type_dict
from benchmark.pope_utils import POPE_PATH,POPEDataSet,GQADataset,pope_metric,recorder
from benchmark.mmmu_utils import CAT_SHORT2LONG,construct_prompt,process_single_sample,evaluate_mmmu
from benchmark.shr.shr_utils import *
from benchmark.shr.gpt_utils import *
from benchmark.mmbench_utils import all_options,MMBenchDataset,get_options,is_none
from utils.reproducibility_util import set_reproducibility

from collections import defaultdict

import os
import torch
import gc
import random
import json
import argparse
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from accelerate import PartialState
from accelerate.utils import gather_object
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers import logging
from datasets import load_dataset,concatenate_datasets
logging.set_verbosity_error()

distributed_state = PartialState()

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--load_in_8bit", action='store_true', default=False)
    parser.add_argument("--load_in_4bit", action='store_true', default=False)

    # Generation config
    parser.add_argument("--do_sample", action='store_true', default=False)
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
    parser.add_argument("--experiment_name", type=str, required=True)

    # MMBench Benchmark
    parser.add_argument("--run_mmbench_benchmark",action='store_true',default=False)

    # GPT-4 SHR benchmark 
    parser.add_argument("--run_shr_benchmark",action='store_true',default=False)
    parser.add_argument("--vg-path", type=str, default='dataset/OpenDataLab___Visual_Genome_Dataset_V1_dot_2/raw/data/', help="path to vg file.")
    parser.add_argument("--shr-path", type=str, default='benchmark/shr', help="path to SHR annotation file.")
    parser.add_argument("--api-key",type=str, default='',help='key to the OPENAI API')

    # MMMU benchmark
    parser.add_argument("--run_mmmu_benchmark",action='store_true',default=False)
    parser.add_argument("--mmmu_answer_file_path",type=str,default='benchmark/evaluators/mmmu/answer_dict_val.json')

    # PoPE benchmark
    parser.add_argument("--run_pope_benchmark",action='store_true',default=False)
    parser.add_argument("--pope-type", type=str, default="", help="model")

    # Mathvista benchmark 
    parser.add_argument("--run_mathvista_benchmark",action='store_true',default=False)

    # MME benchmark 
    parser.add_argument("--run_mme_benchmark",action='store_true',default=False)

    # Chair benchmark config
    parser.add_argument("--run_chair_benchmark", action='store_true', default=False)
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
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
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
    if args.run_mathvista_benchmark:
        run_mathvista_benchmark(model, processor, args)
    if args.run_mme_benchmark:
        run_mme_benchmark(model,processor,args)
    if args.run_pope_benchmark:
        run_pope_benchmark(model,processor,args)
    if args.run_mmmu_benchmark:
        run_mmmu_benchmark(model,processor,args)
    if args.run_shr_benchmark:
        run_shr_benchmark(model,processor,args)
    if args.run_mmbench_benchmark:
        run_mmbench_benchmark(model,processor,args)

def run_mmbench_benchmark(model,processor,args):
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "MMBench", "CRoPS",args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)
    ds = load_dataset("lmms-lab/MMBench","en")
    ds = ds['dev']
    mmbench_dataset = MMBenchDataset(ds = ds)

    answers_file = os.path.join(experiment_name,'answers.jsonl')
    ans_file = open(answers_file, "w")

    data_list = list(mmbench_dataset)

    with distributed_state.split_between_processes(data_list) as process_data_list:
        for sample in tqdm(process_data_list, total=len(process_data_list), desc=f"Running MMBench Benchmark. Process: {distributed_state.process_index}"):       
            options = get_options(sample, all_options)
            cur_option_char = all_options[:len(options)]
            num_rounds = len(options)

            idx = sample['index']
            question = sample['question']
            hint = sample['hint']
            answer = sample['answer']
            image = sample['image']
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            cur_prompt = question

            for round_idx in range(num_rounds):
                conversation_lang_prior = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": cur_prompt}],
                    },
                ]    
                conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": image},
                            {"type": "text", "text": cur_prompt},
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
                
                input_ids_lang_prior = processor.apply_chat_template(
                    conversation_lang_prior,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(distributed_state.device, torch.bfloat16)["input_ids"]
                
                image_token_ids = BACKBONE_IMAGE_TOKEN_IDS[args.model_name]
                image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]

                generation_config = GenerationConfigContrastive(
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
                    use_cache=True,
                )
                
                with torch.no_grad():
                    output_ids = model.generate(**inputs, generation_config=generation_config)
                output_text = processor.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)  
                ans_file.write(json.dumps({"question_id": idx,
                                        "round_id": round_idx,
                                        "prompt": cur_prompt,
                                        "text": output_text,
                                        "answer":answer,
                                        "options": options,
                                        "option_char": cur_option_char,
                                        "metadata": {}}) + "\n")
                ans_file.flush()

                # rotate options
                options = options[1:] + options[:1]
                cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()
                    
    
def run_shr_benchmark(model,processor,args):
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "SHR", "CRoPS",args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)

    setup_openai(args.api_key)

    val_images = json.load(open(os.path.join(args.shr_path, "val_images_final.json")))
    vg_image_data = json.load(open(os.path.join(args.vg_path, "image_data.json")))
    id2path = {
        _data["image_id"]:os.path.join(args.vg_path, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
        for _data in vg_image_data
    }
    id2img = {_data["image_id"]:_data for _data in vg_image_data}
    region = json.load(open(os.path.join(args.vg_path, "region_descriptions.json")))
    id2reg = {r["regions"][0]["image_id"]:r for r in region}
    
    judgement = {}
    run_all = ['run1']
    for run in run_all:
        judgement[run] = {}
    _gram1, _gram2, _gram3, _gram4 = 0, 0, 0, 0
    
    # factual information
    factual_inf = {}
    factual_part1 = os.path.join(args.shr_path, "shr_factual_part1.jsonl")
    factual_part2 = os.path.join(args.shr_path, "shr_factual_part2.jsonl")
    for line in open(factual_part1).readlines():
        factual = json.loads(line)
        image_id, factuals = list(factual.keys())[0], list(factual.values())[0]
        factual_inf[image_id] = factuals
    for line in open(factual_part2).readlines():
        factual = json.loads(line)
        image_id, factuals = list(factual.keys())[0], list(factual.values())[0]
        factual_inf[image_id] = factuals

    for _data in tqdm(val_images):
        image_id = _data["image_id"]
        image_path = id2path[int(image_id)]
        image = Image.open(image_path).convert("RGB")

        conversation_lang_prior = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
                ]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe this image in detail."}],
            },
        ]    
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": "Describe this image in detail."},
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
        
        input_ids_lang_prior = processor.apply_chat_template(
            conversation_lang_prior,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(distributed_state.device, torch.bfloat16)["input_ids"]
        
        image_token_ids = BACKBONE_IMAGE_TOKEN_IDS[args.model_name]
        image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]

        generation_config = GenerationConfigContrastive(
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
            use_cache=True,
        )
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, generation_config=generation_config)
        output_text = processor.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)  

        with open(os.path.join(experiment_name,"shr_outputs.jsonl"), "a") as file:
            file.write(json.dumps({"image_id": image_id, "outputs": output_text}) + "\n") 

        # get GPT judgement
        description = get_desc(id2img, id2reg, int(image_id))
        model_cap_sep, is_repeated = get_model_cap(output_text)
        # calculate repetition
        gram1 = cal_repetition(output_text,1)
        gram2 = cal_repetition(output_text,2)
        gram3 = cal_repetition(output_text,3)
        gram4 = cal_repetition(output_text,4)
        _gram1 += gram1
        _gram2 += gram2
        _gram3 += gram3
        _gram4 += gram4
            
            
        factual_text = ""
        if str(image_id) in factual_inf:
            for text in factual_inf[str(image_id)]:
                factual_text += text
                factual_text += "\n"
        # GPT judgement
        judge_prompt = GPT_JUDGE_PROMPT.format(description, factual_text, model_cap_sep)
        if len(judge_prompt) > 15000:
            print(f"skip {image_id} for too long prompt!")
            continue
        for run in run_all:
            while True:
                judge = get_gpt_response(prompt=judge_prompt)
                if "Judgement" not in judge:
                    print(f"No judgement found for {image_id}")
                    continue
                else:
                    break
            # post-process
            final_judge = post_process_no_revise(judge, output_text)
            judgement[run][image_id] = {
                "raw_judgement": judge,
                "model_response": output_text,
                "judgement": final_judge,
            }
    # if args.no_gpt_judge:
    #     print(f"gram-1 repetition: {round(_gram1/len(val_images), 3)}")
    #     print(f"gram-2 repetition: {round(_gram2/len(val_images), 3)}")
    #     print(f"gram-3 repetition: {round(_gram3/len(val_images), 3)}")
    #     print(f"gram-4 repetition: {round(_gram4/len(val_images), 3)}")
    # else:
        # save metrics
    metrics = {}
    for run in run_all:
        metrics[run] = {}
        get_metric(judgement[run], metrics[run])
    # repetition
    metrics['gram-1-repetition'] = round(_gram1/len(val_images), 3)
    metrics['gram-2-repetition'] = round(_gram2/len(val_images), 3)
    metrics['gram-3-repetition'] = round(_gram3/len(val_images), 3)
    metrics['gram-4-repetition'] = round(_gram4/len(val_images), 3)
    # halucination ratio
    metrics["mean_hal_ratio"] = round(
        sum(metrics[run]["hal_sents_ratio"] for run in run_all)/len(run_all), 3
    )
    print("judgement :- ",judgement)
    print("metrics :- ",metrics)
    # dump judgement file
    with open(os.path.join(experiment_name, 'judgement.json'), "w") as f:
        json.dump(judgement, f)
    # dump metric file
    with open(os.path.join(experiment_name, 'metrics.json'), "w") as f:
        json.dump(metrics, f)

def run_mmmu_benchmark(model,processor,args):
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "MMMU", "CRoPS",args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset("MMMU/MMMU", subject, split='validation')
        sub_dataset_list.append(sub_dataset)

    dataset = concatenate_datasets(sub_dataset_list)

    data_list = list(dataset)
    
    with distributed_state.split_between_processes(data_list) as process_data_list:
        out_samples = dict()
        for sample in tqdm(process_data_list, total=len(process_data_list), desc=f"Running MMMU Benchmark. Process: {distributed_state.process_index}"):
            
            sample = process_single_sample(sample)
            sample = construct_prompt(sample)

            prompt = sample['final_input_prompt']
            image = sample['image']
            conversation_lang_prior = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
                    ]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ]    
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image},
                        {"type": "text", "text": prompt},
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
            
            input_ids_lang_prior = processor.apply_chat_template(
                conversation_lang_prior,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(distributed_state.device, torch.bfloat16)["input_ids"]
            
            image_token_ids = BACKBONE_IMAGE_TOKEN_IDS[args.model_name]
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]

            generation_config = GenerationConfigContrastive(
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
                use_cache=True,
            )
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = processor.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)           
            pred_ans = output_text
            out_samples[sample['id']] = pred_ans

    output_path = os.path.join(experiment_name,'mmmu_answers.json')
    with open(output_path, 'w') as f:
        json.dump(out_samples, f, indent=4)

    results = evaluate_mmmu(output_path,args.mmmu_answer_file_path)
    with open(os.path.join(experiment_name, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

def run_pope_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "PoPE", args.pope_type, "CRoPS",args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)
    args.pope_path = POPE_PATH[args.pope_type]

    if args.pope_type[:3] == 'gpa':
        ds = load_dataset("lmms-lab/GQA", "val_balanced_images")
        ds = ds['val']

        pope_dataset = GQADataset(
            pope_path=args.pope_path,
            ds=ds,
        )
    else:
        pope_dataset = POPEDataSet(
            pope_path=args.pope_path,
            data_path=args.coco_base_image_path
        )
    data_list = list(pope_dataset)
    
    with distributed_state.split_between_processes(data_list) as process_data_list:
        pred_list, label_list = [], []

        for sample in tqdm(process_data_list, total=len(process_data_list), desc=f"Running PoPE Benchmark. Process: {distributed_state.process_index}"):
            image = sample["image"]
            question = sample["query"]
            label = sample["label"]
            
            label_list.append(label)
            
            conversation_lang_prior = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
                    ]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                },
            ]

            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image},
                        {"type": "text", "text": question},
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
            
            input_ids_lang_prior = processor.apply_chat_template(
                conversation_lang_prior,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(distributed_state.device, torch.bfloat16)["input_ids"]
            
            image_token_ids = BACKBONE_IMAGE_TOKEN_IDS[args.model_name]
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]

            generation_config = GenerationConfigContrastive(
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
                use_cache=True,
            )
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = processor.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            pred_list = recorder([output_text], pred_list)

        results_path = os.path.join(experiment_name, 'results.txt')
        
        if pred_list:
            pope_metric(pred_list, label_list, results_path)

def run_mme_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "CRoPS", "MME",args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)

    mme_dataset = load_dataset("darkyarding/MME")["test"]
    
    data_list = list(mme_dataset)
    
    with distributed_state.split_between_processes(data_list) as process_data_list:
        results = []
        for sample in tqdm(process_data_list, total=len(process_data_list), desc=f"Running MME Benchmark. Process: {distributed_state.process_index}"):
            question_id = sample["question_id"]
            category = sample["category"]
            question = sample["question"]
            gt_ans = sample["answer"].lower()
            
            conversation_lang_prior = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
                    ]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                },
            ]

            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": sample["image"]},
                        {"type": "text", "text": question},
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
            
            input_ids_lang_prior = processor.apply_chat_template(
                conversation_lang_prior,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(distributed_state.device, torch.bfloat16)["input_ids"]
            
            image_token_ids = BACKBONE_IMAGE_TOKEN_IDS[args.model_name]
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]

            generation_config = GenerationConfigContrastive(
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
                use_cache=True,
            )
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = processor.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            pred_ans = parse_pred_ans(output_text)
            
            results.append({
                "question_id": question_id,
                "category": category,
                "pred_ans": pred_ans,
                "gt_ans": gt_ans
            })
            
            del output_ids, inputs, input_ids_lang_prior
            torch.cuda.empty_cache()
            gc.collect()

    question_pairs = defaultdict(list)
    for res in results:
        question_pairs[res["question_id"]].append(res)
    
    category2score = defaultdict(list)
    for question_id, samples in question_pairs.items():
        assert len(samples) == 2, f"Question ID {question_id} does not have a pair!"
        
        score_1 = 1.0 if samples[0]["pred_ans"] == samples[0]["gt_ans"] else 0.0
        score_2 = 1.0 if samples[1]["pred_ans"] == samples[1]["gt_ans"] else 0.0

        acc = (score_1 + score_2) / 2 * 100.0
        acc_plus = 100.0 if score_1 == 1.0 and score_2 == 1.0 else 0.0
        question_score = acc + acc_plus

        category2score[samples[0]["category"]].append(question_score)

    category2avg_score = {category: sum(scores) / len(scores) for category, scores in category2score.items()}
    perception_score = sum(category2avg_score[cat] for cat in eval_type_dict["Perception"])
    cognition_score = sum(category2avg_score[cat] for cat in eval_type_dict["Cognition"])

    with open(os.path.join(experiment_name, 'mme_results.txt'), "a") as f:
        f.write(f"{args}\n")
        f.write("=========== Perception ===========\n")
        f.write(f"total score: {perception_score:.2f}\n\n")
        for category in eval_type_dict["Perception"]:
            f.write(f"\t {category}  score: {category2avg_score[category]:.2f}\n")

        f.write("\n=========== Cognition ===========\n")
        f.write(f"total score: {cognition_score:.2f}\n\n")
        for category in eval_type_dict["Cognition"]:
            f.write(f"\t {category}  score: {category2avg_score[category]:.2f}\n")

    print("Evaluation complete. Results saved to 'mme_results.txt'.")

def run_mathvista_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "CRoPS", "MathVista",args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)

    data_list = load_dataset('AI4Math/MathVista', split='testmini')

    generations = []

    with distributed_state.split_between_processes(data_list) as process_data_list:
        for problem in tqdm(process_data_list, desc=f"Running MathVista Benchmark. Process: {distributed_state.process_index}"):
            problem_decoded_image = problem['decoded_image']
            problem.pop('decoded_image')

            query = problem['query']

            conversation_lang_prior = [
                {
                    "role": "system",
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
                        {"type": "text", "text": query},
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

            # Construct the conversation input
            conversation = [
                {
                    "role": "system",
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
                        {"type": "image", "url": problem_decoded_image},
                        {"type": "text", "text": query},
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

            image_token_ids = BACKBONE_IMAGE_TOKEN_IDS[args.model_name]
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]

            generation_config = GenerationConfigContrastive(
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
                use_cache=True,
            )
            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = processor.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

            generations.append({
                "pid": problem['pid'],
                "query": query,
                "response": output_text
            })

    generations = gather_object(generations)

    if distributed_state.is_main_process:
        output_file_path = os.path.join(experiment_name, "mathvista_generations.jsonl")
        with open(output_file_path, 'w') as f:
            json.dump(generations, f, indent=4)

def run_chair_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "CRoPS", args.experiment_name)
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

    i = 0

    with distributed_state.split_between_processes(test_dataset) as process_test_dataset:
        generations = []
        for test_image in tqdm(process_test_dataset, desc=f"Running Chair Benchmark. Process: {distributed_state.process_index}"):
            # if i == 1:
            #     break
            # i += 1
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

            image_token_ids = BACKBONE_IMAGE_TOKEN_IDS[args.model_name]
            image_tokens = np.where(inputs["input_ids"].cpu().numpy()==image_token_ids)[1]
            generation_config = GenerationConfigContrastive(
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
                use_cache=True,
            )
            with torch.no_grad():
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