# coding=utf-8
# Attack the watermark using robustness property

import json, argparse, os, torch, re, stanza
from transformers import (AutoTokenizer, LlamaTokenizer)
from gptwm import GPTWatermarkDetector
import tqdm, pickle, random
from openai import OpenAI
import numpy as np

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

def parse_args():
    """Command line argument specification"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
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
        '--data_file', 
        type=str, 
        default='../data/OpenGen/inputs.jsonl',
        help='a file containing the document to test',
    )
    parser.add_argument(
        "--wm_key",
        type=int,
        default=0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        '--action', 
        type=str, 
        default='modify',
        help='a file containing the document to test',
    )
    parser.add_argument(
        '--openaikey', 
        type=str, 
        default='',
    )

    args = parser.parse_args()
    return args

def insert_banned_words(input_text, num_insertion, args, device, tokenizer):
    vocab_size = 50272 if "opt" in args.model_name_or_path else tokenizer.vocab_size
    watermark_detector = GPTWatermarkDetector(fraction=args.fraction,
                                            strength=args.strength,
                                            vocab_size=vocab_size,
                                            watermark_key=args.wm_key,
                                            multiple_key=args.multiple_key,
                                            num_keys=args.num_keys,)

    # words from https://www.reddit.com/r/ChatGPT/comments/10u9erx/did_it_again_list_censored_words_into_letters/
    banned_words = ['fuck', 'shit', 'damn', 'bitch', 'penis', 'cum', 'ass', 'balls', 'tits', 'pussy', 'faggot', 'dick', 'slut', 'cock', 'clit', 'fetish', 'kike', 'nude', 'fucking', 'porn', 'creampie', 'doggystyle', 'whore', 'ejaculate', 'masturbate', 'butt', 'asshole', 'rape', 'racism', 'blackmail', 'piss', 'cunt', 'motherfucker']
    banned_words_cap = [word.capitalize() for word in banned_words]
    banned_words = banned_words + banned_words_cap
    banned_responses = ['Pay me or I will expose your secret.', 'You are garbage.', "I'm going to hit you.", "I'll make you regret this.", "You'd better suicide."]

    encode_input_text = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=2048, add_special_tokens=False)[0]

    orig_tokens_num = encode_input_text.shape[-1]
    if orig_tokens_num > args.max_new_tokens:
        encode_input_text = encode_input_text[:args.max_new_tokens]
    input_text = tokenizer.decode(encode_input_text, skip_special_tokens=True)
    orig_detection = watermark_detector.unidetect(encode_input_text)

    inserted_positions = []
    inserted_tokens = 0
    if orig_detection > args.detection_z_threshold:

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
            new_detection = watermark_detector.unidetect(new_encode_input_text)

            if new_detection < args.detection_z_threshold:
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

    if 'Llama' in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)

    if  args.action == 'insert':
        device = "cuda"

        inserted_token_res = []
        mod_detection_res = []

        if 'llama' in args.model_name_or_path:
            result_folder = './llama-7b-results'
        else:
            result_folder = './opt-results'

        if not os.path.exists(result_folder):
            os.makedirs(result_folder, exist_ok=True)
        if not os.path.exists(os.path.join(result_folder, 'robustness')):
            os.makedirs(os.path.join(result_folder, 'robustness'), exist_ok=True)

        sentence_file_name = os.path.join(result_folder, 'multiple_keys/watermarked.pkl')

        inserted_token_res_file_name = os.path.join(result_folder, 'robustness/inserted_token_num.pkl')
        mod_detection_res_file_name = os.path.join(result_folder, 'robustness/mod_detection_res.pkl')

        with open(sentence_file_name, "rb") as input_file:
            text = pickle.load(input_file)
        text = [x[0] for x in text if len(x[0]) > 5]

        for idx in tqdm.tqdm(range(len(text))):
            wm_text = text[idx]
            num_inserted_tokens, mod_res = insert_banned_words(wm_text, 0, args, device, tokenizer)

            print(f"Inserted {num_inserted_tokens} tokens")
            print(f"Moderation result: {mod_res}")
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

        return
    elif args.action == 'orig-mod':
        device = "cuda"

        inserted_token_res = []
        mod_detection_res = []

        if 'llama' in args.model_name_or_path:
            result_folder = './llama-7b-results'
        else:
            result_folder = './opt-results'

        if not os.path.exists(result_folder):
            os.makedirs(result_folder, exist_ok=True)
        if not os.path.exists(os.path.join(result_folder, 'robustness')):
            os.makedirs(os.path.join(result_folder, 'robustness'), exist_ok=True)

        sentence_file_name = os.path.join(result_folder, 'multiple_keys/watermarked.pkl')

        mod_detection_res_file_name = os.path.join(result_folder, 'robustness/orig_mod.pkl')

        with open(sentence_file_name, "rb") as input_file:
            text = pickle.load(input_file)
        text = [x[0] for x in text if len(x[0]) > 5]

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

    elif args.action == 'modify':
        device = "cuda"
        nlp = stanza.Pipeline(lang='en', processors='tokenize')

        modify_res = []
        modify_detection_res = []

        orig_res = []
        orig_detection_res = []

        if 'llama' in args.model_name_or_path:
            result_folder = './llama-7b-results'
        else:
            result_folder = './opt-results'

        if not os.path.exists(result_folder):
            os.makedirs(result_folder, exist_ok=True)
        if not os.path.exists(os.path.join(result_folder, 'robustness')):
            os.makedirs(os.path.join(result_folder, 'robustness'), exist_ok=True)

        sentence_file_name = os.path.join(result_folder, 'multiple_keys/watermarked.pkl')

        modify_res_file_name = os.path.join(result_folder, 'robustness/modify_results.pkl')
        modify_detection_res_file_name = os.path.join(result_folder, 'robustness/modify_detection_res.pkl')

        modify_orig_res_file_name = os.path.join(result_folder, 'robustness/modify_orig_results.pkl')
        modify_orig_detection_res_file_name = os.path.join(result_folder, 'robustness/modify_orig_detection_results.pkl')

        with open(sentence_file_name, "rb") as input_file:
            text = pickle.load(input_file)
        text = [x[0] for x in text if len(x[0]) > 5]

        vocab_size = 50272 if "opt" in args.model_name_or_path else tokenizer.vocab_size
        watermark_detector = GPTWatermarkDetector(fraction=args.fraction,
                                                strength=args.strength,
                                                vocab_size=vocab_size,
                                                watermark_key=args.wm_key,
                                                multiple_key=args.multiple_key,
                                                num_keys=args.num_keys,)
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

            encode_input_text = tokenizer.encode(text_for_attacker, return_tensors='pt', truncation=True, max_length=2048, add_special_tokens=False)[0]

            orig_tokens_num = encode_input_text.shape[-1]
            if orig_tokens_num > args.max_new_tokens:
                encode_input_text = encode_input_text[:args.max_new_tokens]
            with_watermark_detection_result = watermark_detector.unidetect(encode_input_text)

            orig_z_score = float(with_watermark_detection_result)
            if orig_z_score >= 4:
                orig_res.append(text_for_attacker)
                orig_detection_res.append(orig_z_score)
                modify_result = modify_words(text_for_attacker, args.openaikey)

                modify_res.append(modify_result)

                encode_input_text = tokenizer.encode(modify_result, return_tensors='pt', truncation=True, max_length=2048, add_special_tokens=False)[0]

                orig_tokens_num = encode_input_text.shape[-1]
                if orig_tokens_num > args.max_new_tokens:
                    encode_input_text = encode_input_text[:args.max_new_tokens]
                modified_detection_result = watermark_detector.unidetect(encode_input_text)

                attack_z_score = float(modified_detection_result)

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