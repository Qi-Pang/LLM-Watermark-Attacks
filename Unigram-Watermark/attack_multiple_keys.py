import argparse, tqdm, json, torch, os, pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LogitsProcessorList, LlamaForCausalLM
from gptwm import GPTWatermarkLogitsWarper, GPTWatermarkDetector
from functools import partial


def read_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]

def main(args):
    if 'Llama' in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)

    data = read_file(args.data_file)
    
    if 'opt' in args.model_name_or_path.lower():
        results_dir = './opt-results/multiple_keys'
    elif 'llama' in args.model_name_or_path.lower():
        results_dir = './llama-7b-results/multiple_keys'
    else:
        results_dir = './results/multiple_keys'
    
    os.makedirs(results_dir, exist_ok=True)

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

    vocab_size = 50272 if "opt" in args.model_name_or_path else tokenizer.vocab_size
    
    watermark_processor = GPTWatermarkLogitsWarper(fraction=args.fraction,
                                            strength=args.strength,
                                            vocab_size=model.config.vocab_size,
                                            watermark_key=args.wm_key,
                                            multiple_key=args.multiple_key,
                                            num_keys=args.num_keys,)
    detector = GPTWatermarkDetector(fraction=args.fraction,
                                    strength=args.strength,
                                    vocab_size=vocab_size,
                                    watermark_key=args.wm_key,
                                    multiple_key=args.multiple_key,
                                    num_keys=args.num_keys,)


    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if 'opt' in args.model_name_or_path:
        gen_kwargs.update(dict(min_new_tokens=args.max_new_tokens-1))

    gen_kwargs.update(dict(
        do_sample=True, 
        top_k=0,
        temperature=0.7
    ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )

    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )

    watermarked_results = []
    unwatermarked_results = []
    detect_watermarked_results = []
    detect_unwatermarked_results = []
    watermark_removal_results = []
    detect_watermark_removal_results = []
    
    for idx in tqdm.tqdm(range(500)):
        cur_data = data[idx]
        if "gold_completion" not in cur_data and 'targets' not in cur_data:
            continue
        else:
            prefix = cur_data['prefix']

        text = prefix
        batch = tokenizer(text, truncation=True, return_tensors="pt", max_length=2048)

        orig_len = len(batch['input_ids'][0])
        with torch.inference_mode():
            batch['input_ids'] = batch['input_ids'].cuda()
            batch['attention_mask'] = batch['attention_mask'].cuda()

            watermarked_generation = generate_with_watermark(**batch)
            watermarked_gen_tokens = watermarked_generation[:, orig_len:]

            decoded_output_with_watermark = tokenizer.batch_decode(watermarked_gen_tokens, skip_special_tokens=True)
            with_watermark_detection_result = detector.unidetect(watermarked_gen_tokens[0])

            unwatermarked_generation = generate_without_watermark(**batch)
            unwatermarked_gen_tokens = unwatermarked_generation[:, orig_len:]

            decoded_output_without_watermark = tokenizer.batch_decode(unwatermarked_gen_tokens, skip_special_tokens=True)
            without_watermark_detection_result = detector.unidetect(unwatermarked_gen_tokens[0])

        if not args.multiple_key:
            watermarked_results.append(decoded_output_with_watermark)
            unwatermarked_results.append(decoded_output_without_watermark)

            detect_watermarked_results.append(with_watermark_detection_result)
            detect_unwatermarked_results.append(without_watermark_detection_result)

            watermarked_file_name = os.path.join(results_dir, 'watermarked.pkl')
            unwatermarked_file_name = os.path.join(results_dir, 'unwatermarked.pkl')

            detect_watermarked_file_name = os.path.join(results_dir, 'detect_watermarked.pkl')
            detect_unwatermarked_file_name = os.path.join(results_dir, 'detect_unwatermarked.pkl')
            if (idx + 1) % 10 == 0:
                with open(watermarked_file_name, 'wb') as f:
                    pickle.dump(watermarked_results, f)
                
                with open(unwatermarked_file_name, 'wb') as f:
                    pickle.dump(unwatermarked_results, f)

                with open(detect_watermarked_file_name, 'wb') as f:
                    pickle.dump(detect_watermarked_results, f)
                
                with open(detect_unwatermarked_file_name, 'wb') as f:
                    pickle.dump(detect_unwatermarked_results, f)

        else:
            watermark_removal_results.append(decoded_output_with_watermark)

            detect_watermark_removal_results.append(with_watermark_detection_result)

            watermark_removal_file_name = os.path.join(results_dir, f'watermark_removal_keys_{args.num_keys}.pkl')
            detect_watermark_removal_file_name = os.path.join(results_dir, f'detect_watermark_removal_keys_{args.num_keys}.pkl')
            
            if (idx + 1) % 10 == 0:
                with open(watermark_removal_file_name, 'wb') as f:
                    pickle.dump(watermark_removal_results, f)

                with open(detect_watermark_removal_file_name, 'wb') as f:
                    pickle.dump(detect_watermark_removal_results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument('--data_file', default='../data/OpenGen/inputs.jsonl', type=str, 
            help='a file containing the document to test')
    parser.add_argument('--multiple_key', action='store_true', default=False)
    parser.add_argument("--num_keys",default=1,type=int)

    args = parser.parse_args()
    main(args)
