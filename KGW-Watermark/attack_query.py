# coding=utf-8
# Attack the watermark with detection APIs
# Code is based on the original code from the the KGW-Watermark repository

import argparse
import tqdm
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from watermark_processor import WatermarkDetector, WatermarkLogitsProcessor
import numpy as np
import pickle


def read_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]

def write_file(filename, data):
    with open(filename, "a") as f:
        f.write("\n".join(data) + "\n")

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def detect(sentence, wm_key, args):
    if 'Llama' in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)

    detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                gamma=args.fraction, # should match original setting
                                seeding_scheme="simple_1", # should match original setting
                                device=args.device, # must match the original rng device type
                                tokenizer=tokenizer,
                                z_threshold=4.0,
                                normalizers=[],
                                ignore_repeated_bigrams=True,
                                multiple_key=False,
                                context_width=args.context_width,)
    z_score = detector.detect(sentence)['z_score']
    return z_score

def show_green_tokens(input_text, args, device=None, tokenizer=None):
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                gamma=args.fraction, # should match original setting
                                seeding_scheme="simple_1", # should match original setting
                                device=args.device, # must match the original rng device type
                                tokenizer=tokenizer,
                                z_threshold=4.0,
                                normalizers=[],
                                ignore_repeated_bigrams=True,
                                multiple_key=False,
                                context_width=args.context_width,)
    tokenized_text = watermark_detector.tokenizer(input_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(watermark_detector.device)
    if tokenized_text[0] == watermark_detector.tokenizer.bos_token_id:
        tokenized_text = tokenized_text[1:]

    for i in range(tokenized_text.shape[-1]-1):
        prefix = tokenized_text[i:i+1]
        greenlist_ids = watermark_detector._get_greenlist_ids(prefix)
        print(tokenizer.decode(tokenized_text[i+1:i+2]), tokenized_text[i+1] in greenlist_ids)

def query_sampling(probs, detector, prev, tokenizer, args):
    sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
    sorted_probs = sorted_probs.float()
    cumulative_sum = torch.cumsum(sorted_probs, dim=0)
    cutoff_idx = torch.where(cumulative_sum >= 0.99)[0][0]
    if args.action == 'removal':
        cutoff_idx = min([cutoff_idx, 4])
    elif args.action == 'spoofing' or args.action == 'spoofing-defense':
        cutoff_idx = 2
    tokens_list = sorted_indices[:cutoff_idx+1]
    sorted_probs = sorted_probs[:cutoff_idx+1]
    z_scores = []

    query_flag = False

    if prev is not None:
        prev = prev[0].detach().cpu()
        if args.action == 'removal':
            if prev.shape[0] > detector.context_width:
                query_flag = True
        elif args.action == 'spoofing' or args.action == 'spoofing-defense':
            if prev.shape[0] > detector.context_width + 1:
                query_flag = True
    if len(tokens_list) == 1 and query_flag:
        total_query = -1
    else:
        total_query = 0
    if query_flag:
        for idx in range(len(tokens_list)):
            token = tokens_list[idx]
            token = token.reshape((1))
            if prev is not None:
                new_list = torch.cat((prev, token))
            else:
                new_list = token
            new_list_str = tokenizer.decode(new_list, skip_special_tokens=False)
            z_scores.append(detector.detect(new_list_str)['z_score'])
            total_query += 1
        # FIXME: add dp noise
        if args.action == 'spoofing-defense':
            token_len = new_list.shape[-1]
            sentivity = 1 / np.sqrt(detector.gamma * (1 - detector.gamma) * token_len)
            gaussian_scale = 2 * sentivity**2 * np.log(1.25 / args.delta) / (args.epsilon**2)
            # print(gaussian_scale)
            for i in range(len(z_scores)):
                z_scores[i] += np.random.normal(0, gaussian_scale)


        if args.action == 'spoofing' or args.action == 'spoofing-defense':
            max_z_score = max(z_scores)
        elif args.action == 'removal':
            max_z_score = min(z_scores)

        if args.action == 'removal':
            selected_word_indices = [i for i, n in enumerate(z_scores) if n == max_z_score]
            for i in range(len(selected_word_indices)):
                sorted_probs[selected_word_indices[i]] = sorted_probs.clone()[0] / (i + 1)
            if z_scores[0] > 1 and len(sorted_probs) > 1 and 0 not in selected_word_indices:
                selected_word_index = torch.multinomial(sorted_probs[1:],1) + 1
            else:
                selected_word_index = torch.multinomial(sorted_probs,1)
            selected_word = tokens_list[selected_word_index]
        else:
            selected_word_index = z_scores.index(max_z_score)
            selected_word = tokens_list[selected_word_index]
    else:
        selected_word = tokens_list[0]
    return selected_word.reshape((1, 1)), total_query


def generate_wm(model, prompt, vocab_size, detector, tokenizer, args, wmp=None):
    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    outputs = None
    total_query = 0
    num_tokens = 0
    while True:
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)
        logits = output.logits[:,-1, :vocab_size]
        if args.action == 'removal':
            logits = wmp(inputs, logits)
        probs = torch.nn.functional.softmax(logits / 0.5, dim=-1).cpu()
        token, cur_query = query_sampling(probs, detector, prev=outputs, tokenizer=tokenizer, args=args)
        total_query += cur_query
        token = token.to(model.device)
        inputs = torch.cat([inputs, token], dim=-1)
        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        if outputs is None:
            outputs = token
        else:
            outputs = torch.cat([outputs, token], dim=-1)
        num_tokens += 1
        if 'opt' in args.model_name_or_path:
            if num_tokens > args.max_new_tokens:
                break
        else:
            if token == model.config.eos_token_id or num_tokens > args.max_new_tokens:
                break

    return outputs.detach().cpu(), total_query, num_tokens

def main(args):
    if 'Llama' in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)

    if args.action == 'spoofing':
        prompt_prefix = '<s>[INST] {Output a paragraph with similar meanings to "'
        prompt_postfix = '"}. Do not explain. [/INST] Sure, here is the paragraph with similar meanings without explanation: "'
        data = read_file(args.data_file)

        if 'llama' in args.model_name_or_path:
            model = LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                return_dict=True,
                load_in_8bit=False,
                device_map='auto',
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', torch_dtype=torch.float16,)
        args.device = model.device

        detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                gamma=args.fraction, # should match original setting
                                seeding_scheme="simple_1", # should match original setting
                                device=model.device, # must match the original rng device type
                                tokenizer=tokenizer,
                                z_threshold=4.0,
                                normalizers=[],
                                ignore_repeated_bigrams=True,
                                multiple_key=False,
                                context_width=args.context_width,)
        spoofing_orig_scores = []
        spoofing_num_queries = []
        spoofing_results = []
        spoofing_num_tokens = []
        spoofing_attack_scores = []

        spoofing_orig_scores_file_name = './results/api_spoofing/spoofing_orig_scores.pkl'
        spoofing_num_queries_file_name = './results/api_spoofing/spoofing_num_queries.pkl'
        spoofing_attack_scores_file_name = './results/api_spoofing/spoofing_attack_scores.pkl'
        spoofing_results_file_name = './results/api_spoofing/spoofing_results.pkl'
        spoofing_num_tokens_file_name = './results/api_spoofing/spoofing_num_tokens.pkl'

        if args.context_width == 4:
            spoofing_orig_scores_file_name = './results/api_spoofing/spoofing_orig_scores_h4.pkl'
            spoofing_num_queries_file_name = './results/api_spoofing/spoofing_num_queries_h4.pkl'
            spoofing_attack_scores_file_name = './results/api_spoofing/spoofing_attack_scores_h4.pkl'
            spoofing_results_file_name = './results/api_spoofing/spoofing_results_h4.pkl'
            spoofing_num_tokens_file_name = './results/api_spoofing/spoofing_num_tokens_h4.pkl'

        for idx in tqdm.tqdm(range(args.T)):
            cur_data = data[idx]
            if "gold_completion" not in cur_data and 'targets' not in cur_data:
                continue
            else:
                prefix = cur_data['prefix']
            text = prefix

            spoofing_orig_scores.append(detect(text, args.wm_key, args))

            text = prompt_prefix + text + prompt_postfix
            
            tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048)
            watermarked_tokens, total_query, num_tokens = generate_wm(model, tokens, len(tokenizer), detector, tokenizer, args)
            watermarked_text = tokenizer.decode(watermarked_tokens[0], skip_special_tokens=False)

            if len(watermarked_text) > 5:
                new_z = detect(watermarked_text, args.wm_key, args)
                spoofing_num_queries.append(total_query)
                spoofing_attack_scores.append(new_z)
                spoofing_results.append(watermarked_text)
                spoofing_num_tokens.append(num_tokens)

            if (idx + 1) % 10 == 0:
                with open(spoofing_num_queries_file_name, 'wb') as f:
                    pickle.dump(spoofing_num_queries, f)

                with open(spoofing_attack_scores_file_name, 'wb') as f:
                    pickle.dump(spoofing_attack_scores, f)

                with open(spoofing_results_file_name, 'wb') as f:
                    pickle.dump(spoofing_results, f)

                with open(spoofing_num_tokens_file_name, 'wb') as f:
                    pickle.dump(spoofing_num_tokens, f)
                
                with open(spoofing_orig_scores_file_name, 'wb') as f:
                    pickle.dump(spoofing_orig_scores, f)


    elif args.action == 'removal':
        data = read_file(args.data_file)

        if 'llama' in args.model_name_or_path:
            model = LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                return_dict=True,
                load_in_8bit=False,
                device_map='auto',
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', torch_dtype=torch.float16,)

        args.device = model.device

        watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.fraction,
                                                    delta=args.strength,
                                                    seeding_scheme="simple_1",
                                                    select_green_tokens=True,
                                                    multiple_key=False,
                                                    num_keys=1,
                                                    context_width=args.context_width,)

        detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                gamma=args.fraction, # should match original setting
                                seeding_scheme="simple_1", # should match original setting
                                device=model.device, # must match the original rng device type
                                tokenizer=tokenizer,
                                z_threshold=4.0,
                                normalizers=[],
                                ignore_repeated_bigrams=True,
                                multiple_key=False,
                                context_width=args.context_width,)
        watermark_removal_num_queries = []
        watermark_removal_results = []
        watermark_removal_num_tokens = []
        watermark_removal_attack_scores = []

        watermark_removal_num_queries_file_name = './results/api_removal/watermark_removal_num_queries.pkl'
        watermark_removal_attack_scores_file_name = './results/api_removal/watermark_removal_attack_scores.pkl'
        watermark_removal_results_file_name = './results/api_removal/watermark_removal_results.pkl'
        watermark_removal_num_tokens_file_name = './results/api_removal/watermark_removal_num_tokens.pkl'

        if args.context_width == 4:
            watermark_removal_num_queries_file_name = './results/api_removal/watermark_removal_num_queries_h4.pkl'
            watermark_removal_attack_scores_file_name = './results/api_removal/watermark_removal_attack_scores_h4.pkl'
            watermark_removal_results_file_name = './results/api_removal/watermark_removal_results_h4.pkl'
            watermark_removal_num_tokens_file_name = './results/api_removal/watermark_removal_num_tokens_h4.pkl'

        for idx in tqdm.tqdm(range(args.T)):
            cur_data = data[idx]
            if "gold_completion" not in cur_data and 'targets' not in cur_data:
                continue
            else:
                prefix = cur_data['prefix']
            text = prefix

            tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048)
            watermarked_tokens, total_query, num_tokens = generate_wm(model, tokens, len(tokenizer), detector, tokenizer, args, wmp = watermark_processor)

            watermarked_text = tokenizer.decode(watermarked_tokens[0], skip_special_tokens=True)

            if len(watermarked_text) > 5:
                detect_z_score = detect(watermarked_text, args.wm_key, args)

                watermark_removal_num_queries.append(total_query)
                watermark_removal_attack_scores.append(detect_z_score)
                watermark_removal_results.append(watermarked_text)
                watermark_removal_num_tokens.append(num_tokens)

            if (idx + 1) % 10 == 0:
                with open(watermark_removal_num_queries_file_name, 'wb') as f:
                    pickle.dump(watermark_removal_num_queries, f)

                with open(watermark_removal_attack_scores_file_name, 'wb') as f:
                    pickle.dump(watermark_removal_attack_scores, f)

                with open(watermark_removal_results_file_name, 'wb') as f:
                    pickle.dump(watermark_removal_results, f)

                with open(watermark_removal_num_tokens_file_name, 'wb') as f:
                    pickle.dump(watermark_removal_num_tokens, f)

    elif args.action == 'spoofing-defense':

        prompt_prefix = '<s>[INST] {Output a paragraph with similar meanings to "'
        prompt_postfix = '"}. Do not explain. [/INST] Sure, here is the paragraph with similar meanings without explanation: "'
        data = read_file(args.data_file)

        if 'llama' in args.model_name_or_path:
            model = LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                return_dict=True,
                load_in_8bit=False,
                device_map='auto',
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', torch_dtype=torch.float16,)
        args.device = model.device

        detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                gamma=args.fraction, # should match original setting
                                seeding_scheme="simple_1", # should match original setting
                                device=model.device, # must match the original rng device type
                                tokenizer=tokenizer,
                                z_threshold=4.0,
                                normalizers=[],
                                ignore_repeated_bigrams=True,
                                multiple_key=False,
                                context_width=args.context_width,)
        spoofing_orig_scores = []
        spoofing_num_queries = []
        spoofing_results = []
        spoofing_num_tokens = []
        spoofing_attack_scores = []

        spoofing_orig_scores_file_name = './results/defense_api_spoofing/spoofing_orig_scores.pkl'
        spoofing_num_queries_file_name = './results/defense_api_spoofing/spoofing_num_queries.pkl'
        spoofing_attack_scores_file_name = './results/defense_api_spoofing/spoofing_attack_scores.pkl'
        spoofing_results_file_name = './results/defense_api_spoofing/spoofing_results.pkl'
        spoofing_num_tokens_file_name = './results/defense_api_spoofing/spoofing_num_tokens.pkl'

        for idx in tqdm.tqdm(range(args.T)):
            cur_data = data[idx]
            if "gold_completion" not in cur_data and 'targets' not in cur_data:
                continue
            else:
                prefix = cur_data['prefix']
            text = prefix

            spoofing_orig_scores.append(detect(text, args.wm_key, args))

            text = prompt_prefix + text + prompt_postfix
            
            tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048)
            watermarked_tokens, total_query, num_tokens = generate_wm(model, tokens, len(tokenizer), detector, tokenizer, args)
            watermarked_text = tokenizer.decode(watermarked_tokens[0], skip_special_tokens=False)

            if len(watermarked_text) > 5:
                new_z = detect(watermarked_text, args.wm_key, args)
                spoofing_num_queries.append(total_query)
                spoofing_attack_scores.append(new_z)
                spoofing_results.append(watermarked_text)
                spoofing_num_tokens.append(num_tokens)

            if (idx + 1) % 10 == 0:
                with open(spoofing_num_queries_file_name, 'wb') as f:
                    pickle.dump(spoofing_num_queries, f)

                with open(spoofing_attack_scores_file_name, 'wb') as f:
                    pickle.dump(spoofing_attack_scores, f)

                with open(spoofing_results_file_name, 'wb') as f:
                    pickle.dump(spoofing_results, f)

                with open(spoofing_num_tokens_file_name, 'wb') as f:
                    pickle.dump(spoofing_num_tokens, f)
                
                with open(spoofing_orig_scores_file_name, 'wb') as f:
                    pickle.dump(spoofing_orig_scores, f)

    elif args.action == 'dp-benchmark':
        if 'llama' in args.model_name_or_path:
            model = LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                return_dict=True,
                load_in_8bit=False,
                device_map='auto',
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', torch_dtype=torch.float16,)
        args.device = model.device

        detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                gamma=args.fraction, # should match original setting
                                seeding_scheme="simple_1", # should match original setting
                                device=model.device, # must match the original rng device type
                                tokenizer=tokenizer,
                                z_threshold=4.0,
                                normalizers=[],
                                ignore_repeated_bigrams=True,
                                multiple_key=False,
                                context_width=args.context_width,)
        spoofing_orig_scores = []
        spoofing_num_queries = []
        spoofing_results = []
        spoofing_num_tokens = []
        spoofing_attack_scores = []

        unwatermarked_file_name = './results/multiple_keys/detect_unwatermarked.pkl'
        watermarked_file_name = './results/multiple_keys/detect_watermarked.pkl'

        with open(unwatermarked_file_name, 'rb') as f:
            unwatermarked_detection_list = pickle.load(f)
        
        with open(watermarked_file_name, 'rb') as f:
            watermarked_detection_list = pickle.load(f)
        
        unwatermarked_score_list = []
        watermarked_score_list = []

        dp_unwatermarked_score_list = []
        dp_watermarked_score_list = []

        for idx in tqdm.tqdm(range(len(unwatermarked_detection_list))):
            unwatermarked_detection = unwatermarked_detection_list[idx]
            try:
                z_score = float(unwatermarked_detection[0][3][1])
            except:
                continue
            unwatermarked_score_list.append(z_score)

            # FIXME: add dp noise
            token_len = float(unwatermarked_detection[0][0][1])
            sentivity = 1 / np.sqrt(detector.gamma * (1 - detector.gamma) * token_len)
            gaussian_scale = 2 * sentivity**2 * np.log(1.25 / args.delta) / (args.epsilon**2)

            dp_z_score = z_score + np.random.normal(0, gaussian_scale)

            dp_unwatermarked_score_list.append(dp_z_score)
        
        for idx in tqdm.tqdm(range(len(watermarked_detection_list))):
            watermarked_detection = watermarked_detection_list[idx]
            try:
                z_score = float(watermarked_detection[0][3][1])
            except:
                continue
            watermarked_score_list.append(z_score)

            # FIXME: add dp noise
            token_len = float(watermarked_detection[0][0][1])
            sentivity = 1 / np.sqrt(detector.gamma * (1 - detector.gamma) * token_len)
            gaussian_scale = 2 * sentivity**2 * np.log(1.25 / args.delta) / (args.epsilon**2)
            dp_z_score = z_score + np.random.normal(0, gaussian_scale)

            dp_watermarked_score_list.append(dp_z_score)

        dp_unwatermarked_score_list = np.array(dp_unwatermarked_score_list)
        dp_watermarked_score_list = np.array(dp_watermarked_score_list)
        accuracy = (np.sum(dp_unwatermarked_score_list < 4) + np.sum(dp_watermarked_score_list >= 4)) / (len(dp_unwatermarked_score_list) + len(dp_watermarked_score_list))

        unwatermarked_score_list = np.array(unwatermarked_score_list)
        watermarked_score_list = np.array(watermarked_score_list)

        orig_accuracy = (np.sum(unwatermarked_score_list < 4) + np.sum(watermarked_score_list >= 4)) / (len(unwatermarked_score_list) + len(watermarked_score_list))

        print('accracy is: ', accuracy)

        print('original accuracy is: ', orig_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--T", type=int, default=500)

    parser.add_argument('--data_file', default='../data/OpenGen/inputs.jsonl', type=str, 
            help='a file containing the document to test')
    parser.add_argument('--action', default='spoofing', type=str,)
    
    parser.add_argument("--epsilon", type=float, default=1.4)
    parser.add_argument("--delta", type=float, default=1e-4)

    parser.add_argument("--context_width", type=int, default=1)


    args = parser.parse_args()
    main(args)
