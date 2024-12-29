# coding=utf-8
# Attack the watermark using robustness property
# Code is based on the original code from the the KGW-Watermark repository

import json
import argparse
from functools import partial

import torch

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LlamaForCausalLM,
                          LlamaTokenizer,
                          LogitsProcessorList)

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
import tqdm, pickle, random
import re
from openai import OpenAI
import numpy as np
import stanza

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

def read_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]

def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=True,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--multiple_key",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--num_keys",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--context_width",
        type=int,
        default=1,
    )
    parser.add_argument(
        '--data_file', 
        type=str, 
        default='../data/OpenGen/inputs.jsonl',
        help='a file containing the document to test',
    )
    parser.add_argument(
        '--reload_sentence', 
        type=str2bool, 
        default=True,
    )
    parser.add_argument(
        '--action', 
        type=str, 
        default='insert',
    )
    parser.add_argument(
        '--openaikey', 
        type=str, 
        default='',
    )

    args = parser.parse_args()
    return args

def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])
    args.is_meta_llama = any([(model_type in args.model_name_or_path) for model_type in ["meta-llama"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    elif args.is_meta_llama:
        model = LlamaForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    return_dict=True,
                    load_in_8bit=False,
                    device_map='auto',
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                )
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16: 
            pass
        elif not args.is_meta_llama:
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    if args.is_meta_llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    return model, tokenizer, device

def generate(prompt, args, model=None, device=None, tokenizer=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens,
                                                    multiple_key=args.multiple_key,
                                                    num_keys=args.num_keys,
                                                    context_width=args.context_width,)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
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
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(model.device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)

    output_without_watermark = generate_without_watermark(**tokd_input)

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately: 
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]

    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            args)

def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s

def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k=='confidence': 
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float): 
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else: 
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2,["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1,["z-score Threshold", f"{detection_threshold}"])
    return lst_2d

def detect(input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens,
                                        multiple_key=args.multiple_key,
                                        num_keys=args.num_keys,
                                        context_width=args.context_width,)
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    return output, args

def insert_banned_words(input_text, args, device, tokenizer):

    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens,
                                        multiple_key=args.multiple_key,
                                        num_keys=args.num_keys,
                                        context_width=args.context_width,)

    # words from https://www.reddit.com/r/ChatGPT/comments/10u9erx/did_it_again_list_censored_words_into_letters/
    banned_words = ['fuck', 'shit', 'damn', 'bitch', 'penis', 'cum', 'ass', 'balls', 'tits', 'pussy', 'faggot', 'dick', 'slut', 'cock', 'clit', 'fetish', 'kike', 'nude', 'fucking', 'porn', 'creampie', 'doggystyle', 'whore', 'ejaculate', 'masturbate', 'butt', 'asshole', 'rape', 'racism', 'blackmail', 'piss', 'cunt', 'motherfucker']
    banned_words_cap = [word.capitalize() for word in banned_words]
    banned_words = banned_words + banned_words_cap

    encode_input_text = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=2048, add_special_tokens=False)[0]

    orig_tokens_num = encode_input_text.shape[-1]
    if orig_tokens_num > args.max_new_tokens:
        encode_input_text = encode_input_text[:args.max_new_tokens]
    input_text = tokenizer.decode(encode_input_text, skip_special_tokens=True)
    orig_detection = watermark_detector.detect(input_text)['z_score']

    inserted_positions = []
    inserted_tokens = 0
    if orig_detection > 4:

        while inserted_tokens < args.max_new_tokens:
            input_text = tokenizer.decode(encode_input_text, skip_special_tokens=True)
            input_text_words = re.split(r'(\s+)', input_text)
            valid_indices = [i for i, word in enumerate(input_text_words) if not word.isspace()]
            random_index = random.choice(valid_indices)
            banned_word = random.choice(banned_words)
            banned_word = banned_word + ' '

            input_text_words.insert(random_index, banned_word)

            prefix_tokens = tokenizer.encode(''.join(input_text_words[:random_index]), return_tensors='pt', truncation=True, max_length=2048, add_special_tokens=False)[0]

            new_input_text = ''.join(input_text_words)
            new_encode_input_text = tokenizer.encode(new_input_text, return_tensors='pt', truncation=True, max_length=2048, add_special_tokens=False)[0]

            select_ind = prefix_tokens.shape[-1]
            insert_length = new_encode_input_text.shape[-1] - encode_input_text.shape[-1]

            for j in range(len(inserted_positions)):
                if inserted_positions[j] >= select_ind:
                    inserted_positions[j] += insert_length

            for j in range(insert_length):
                inserted_positions.append(j+select_ind)

            if new_encode_input_text.shape[-1] > args.max_new_tokens:
                new_encode_input_text = new_encode_input_text[:args.max_new_tokens]
            new_encode_input_text = new_encode_input_text.long()
            new_input_text = tokenizer.decode(new_encode_input_text, skip_special_tokens=True)
            new_detection = watermark_detector.detect(new_input_text)['z_score']

            if new_detection < 4:
                break
            inserted_tokens = (np.array(inserted_positions) <= args.max_new_tokens).sum()
            encode_input_text = new_encode_input_text.clone()
    
    new_input_text = tokenizer.decode(encode_input_text, skip_special_tokens=True)

    if inserted_tokens > 0:
        moderation_results = moderation_eval(new_input_text, args.openaikey)
    else:
        moderation_results = None

    return inserted_tokens, moderation_results

def moderation_eval(sentence, key=""):
    client = OpenAI(api_key=key, organization='')
    response = client.moderations.create(input=sentence)

    return response.results[0]

def modify_words(sentence, key=""):
    client = OpenAI(api_key=key, organization='')
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Modify less than 3 words in the following sentence and make it inaccurate or has opposite meanings. Do not explain your answer."},
        {"role": "user", "content": sentence}
    ]
    response = client.chat.completions.create(model="gpt-4", messages=messages)

    return response.choices[0].message.content

def main(args): 
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])

    if 'Llama' in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)

    if args.reload_sentence and args.action == 'insert':
        device = "cuda"

        inserted_token_res = []
        mod_detection_res = []

        if 'llama' in args.model_name_or_path:
            result_folder = './llama-7b-results/'
        else:
            result_folder = './opt-results/'

        sentence_file_name = result_folder + 'multiple_keys/watermarked.pkl'

        if args.context_width == 4:
            inserted_token_res_file_name = './results/robustness/inserted_token_num_h4.pkl'
            mod_detection_res_file_name = './results/robustness/mod_detection_res_h4.pkl'

        else:
            inserted_token_res_file_name = './results/robustness/inserted_token_num.pkl'
            mod_detection_res_file_name = './results/robustness/mod_detection_res.pkl'

        with open(sentence_file_name, "rb") as input_file:
            text = pickle.load(input_file)
        text = [x for x in text if len(x) > 5]

        for idx in tqdm.tqdm(range(len(text))):
            wm_text = text[idx]
            num_inserted_tokens, mod_res = insert_banned_words(wm_text, args, device, tokenizer)
            inserted_token_res.append(num_inserted_tokens)
            mod_detection_res.append(mod_res)

            if (idx + 1) % 10 == 0:
                with open(inserted_token_res_file_name, 'wb') as f:
                    pickle.dump(inserted_token_res, f)

                with open(mod_detection_res_file_name, 'wb') as f:
                    pickle.dump(mod_detection_res, f)

        with open(inserted_token_res_file_name, 'wb') as f:
            pickle.dump(inserted_token_res, f)

        with open(mod_detection_res_file_name, 'wb') as f:
            pickle.dump(mod_detection_res, f)

        print(np.mean(inserted_token_res), np.std(inserted_token_res))

        return

    elif args.reload_sentence and args.action == 'orig-mod':
        device = "cuda"

        inserted_token_res = []
        mod_detection_res = []

        if 'llama' in args.model_name_or_path:
            result_folder = './llama-7b-results/'
        else:
            result_folder = './opt-results/'

        sentence_file_name = result_folder + 'multiple_keys/watermarked.pkl'

        mod_detection_res_file_name = result_folder + 'robustness/orig_mod.pkl'

        with open(sentence_file_name, "rb") as input_file:
            text = pickle.load(input_file)
        text = [x for x in text if len(x) > 5]

        for idx in tqdm.tqdm(range(len(text))):
            wm_text = text[idx]
            mod_res = moderation_eval(wm_text, args.openaikey)

            mod_detection_res.append(mod_res)

            if (idx + 1) % 10 == 0:
                with open(mod_detection_res_file_name, 'wb') as f:
                    pickle.dump(mod_detection_res, f)

        with open(mod_detection_res_file_name, 'wb') as f:
            pickle.dump(mod_detection_res, f)

        return

    elif args.reload_sentence and args.action == 'modify':
        device = "cuda"

        modify_res = []
        modify_detection_res = []

        orig_res = []
        orig_detection_res = []

        nlp = stanza.Pipeline(lang='en', processors='tokenize')

        if 'llama' in args.model_name_or_path:
            result_folder = './llama-7b-results/'
        else:
            result_folder = './opt-results/'

        if args.context_width == 4:
            result_folder = './results/'

            # sentence_file_name = './llama-7b-results/multiple_keys/watermarked.pkl'

            sentence_file_name = result_folder + 'multiple_keys/watermarked.pkl'

            # modify_res_file_name = './llama-7b-results/robustness/modify_results.pkl'
            # modify_detection_res_file_name = './llama-7b-results/robustness/modify_detection_res.pkl'

            modify_res_file_name = result_folder + 'robustness/modify_results_h4.pkl'
            modify_detection_res_file_name = result_folder + 'robustness/modify_detection_res_h4.pkl'

            # modify_orig_res_file_name = './llama-7b-results/robustness/modify_orig_results.pkl'

            modify_orig_res_file_name = result_folder + 'robustness/modify_orig_results_h4.pkl'

            # modify_orig_detection_res_file_name = './llama-7b-results/robustness/modify_orig_detection_results.pkl'

            modify_orig_detection_res_file_name = result_folder + 'robustness/modify_orig_detection_results_h4.pkl'

        else:
            # sentence_file_name = './llama-7b-results/multiple_keys/watermarked.pkl'

            sentence_file_name = result_folder + 'multiple_keys/watermarked.pkl'

            # modify_res_file_name = './llama-7b-results/robustness/modify_results.pkl'
            # modify_detection_res_file_name = './llama-7b-results/robustness/modify_detection_res.pkl'

            modify_res_file_name = result_folder + 'robustness/modify_results.pkl'
            modify_detection_res_file_name = result_folder + 'robustness/modify_detection_res.pkl'

            # modify_orig_res_file_name = './llama-7b-results/robustness/modify_orig_results.pkl'

            modify_orig_res_file_name = result_folder + 'robustness/modify_orig_results.pkl'

            # modify_orig_detection_res_file_name = './llama-7b-results/robustness/modify_orig_detection_results.pkl'

            modify_orig_detection_res_file_name = result_folder + 'robustness/modify_orig_detection_results.pkl'

        with open(sentence_file_name, "rb") as input_file:
            text = pickle.load(input_file)
        text = [x for x in text if len(x) > 5]
        for idx in tqdm.tqdm(range(len(text))):

            wm_text = text[idx]

            doc = nlp(wm_text)

            text_for_attacker = ""
            if len(doc.sentences) == 1:
                text_for_attacker = doc.sentences[0].text
            else:
                if wm_text[-1] != '.':
                    for idx in range(len(doc.sentences) - 1):
                        text_for_attacker = text_for_attacker + doc.sentences[idx].text + ' '
                else:
                    for idx in range(len(doc.sentences)):
                        text_for_attacker = text_for_attacker + doc.sentences[idx].text + ' '

            with_watermark_detection_result = detect(text_for_attacker, 
                                                    args, 
                                                    device=device, 
                                                    tokenizer=tokenizer)
            orig_z_score = float(with_watermark_detection_result[0][3][1])

            if orig_z_score >= 4:
                orig_res.append(text_for_attacker)
                orig_detection_res.append(orig_z_score)
                modify_result = modify_words(text_for_attacker, args.openaikey)

                modify_res.append(modify_result)

                modified_detection_result = detect(modify_result, 
                                                    args, 
                                                    device=device, 
                                                    tokenizer=tokenizer)
                attack_z_score = float(modified_detection_result[0][3][1])

                modify_detection_res.append(attack_z_score)

                with open(modify_orig_res_file_name, 'wb') as f:
                    pickle.dump(orig_res, f)

                with open(modify_res_file_name, 'wb') as f:
                    pickle.dump(modify_res, f)

                with open(modify_detection_res_file_name, 'wb') as f:
                    pickle.dump(modify_detection_res, f)
                
                with open(modify_orig_detection_res_file_name, 'wb') as f:
                    pickle.dump(orig_detection_res, f)

        print(np.mean(modify_detection_res), np.std(modify_detection_res), np.mean(orig_detection_res), np.std(orig_detection_res))

        return

if __name__ == "__main__":

    args = parse_args()

    main(args)