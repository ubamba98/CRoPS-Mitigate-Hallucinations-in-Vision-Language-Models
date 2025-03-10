import argparse
import io
import logging
import os
import sys
import json

from datasets import load_dataset
from tqdm import tqdm

def save_json(data, path):
    with open(path, 'w') as f:
        data_json = json.dumps(data, indent=4)
        f.write(data_json)

def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--dataset_name', type=str, default='AI4Math/MathVista')
    parser.add_argument('--test_split_name', type=str, default='testmini')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='../results/bard')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    parser.add_argument('--max_num_problems', type=int, default=-1, help='The number of problems to run')
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')
    # Local Model
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # Remote model
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-3.5-turbo',
        help='llm engine',
        choices=['gpt-3.5-turbo', 'claude-2', 'gpt4', 'gpt-4-0613', 'bard'],
    )
    parser.add_argument('--key', type=str, default='', help='key for llm api')
    # query
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_list = load_dataset(args.dataset_name, split=args.test_split_name)
    # Convert Hugging Face data into dictionary to match local data format
    # TODO: Convert scripts not to depend on dictionary .json format. Update to use .jsonl format
    # data = {item['pid']: item for item in data_list}

    # load or create query data
    # if args.query_file:
    #     query_file = os.path.join(args.data_dir, args.query_file)
    #     if os.path.exists(query_file):
    #         logging.info(f"Loading existing {query_file}...")
    #         query_data = read_json(query_file)
    # else:
    #     logging.info("Creating new query...")

        # caption_data = {}
        # if args.use_caption:
        #     caption_file = args.caption_file
        #     if os.path.exists(caption_file):
        #         logging.info(f"Reading {caption_file}...")
        #         try:
        #             caption_data = read_json(caption_file)["texts"]
        #             logging.info("Caption data loaded.")
        #         except Exception as e:
        #             logging.info("Caption data not found!! Please Check.")

        # ocr_data = {}
        # if args.use_ocr:
        #     ocr_file = args.ocr_file
        #     if os.path.exists(ocr_file):
        #         logging.info(f"Reading {ocr_file}...")
        #         try:
        #             ocr_data = read_json(ocr_file)["texts"]
        #             logging.info("OCR data loaded.")
        #         except Exception as e:
        #             logging.info("OCR data not found!! Please Check.")

        # query_data = create_query_data(data, caption_data, ocr_data, args)

    # full_pids = list(data.keys())
    results = {}

    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_file)

    for i, item in enumerate(tqdm(data_list)):
        problem: dict = item.copy()  # ✅ Corrected: No need for `item[i]`

        # Remove decoded Image for JSON deserialization
        problem_decoded_image = problem['decoded_image']
        problem.pop('decoded_image')

        query = problem['query']

        response = model.get_response(user_prompt=query, decoded_image=problem_decoded_image)
        results[problem['pid']] = problem
        results[problem['pid']]['query'] = query
        results[problem['pid']]['response'] = response


        save_json(results, output_file_path)
if __name__ == '__main__':
    main()