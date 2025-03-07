import sys
sys.path.append('./')

import tqdm, json, torch, pickle, argparse, re, random, stanza, os
from transformers import AutoTokenizer, LlamaTokenizer
import numpy as np
from watermarking.detection import phi, fast_permutation_test_query
from watermarking.gumbel.score import gumbel_edit_score
from watermarking.gumbel.key import gumbel_key_func
from openai import OpenAI

def read_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]

def detect(sentence, seed, tokenizer, args, test):
    if isinstance(sentence, str):
        tokens = tokenizer.encode(sentence, return_tensors='pt', truncation=True, max_length=2048)[0]
    else:
        tokens = sentence
    token_len = tokens.shape[-1]
    if token_len > args.m:
        token_len = args.m
    null_id = (token_len - 1) // 5
    cur_k = (null_id + 1) * 5
    if tokens.shape[-1] < cur_k:
        tokens = torch.nn.functional.pad(tokens,(cur_k-tokens.shape[-1],0),"constant",0)
    else:
        tokens = tokens[:args.m]

    pval = test(tokens, seed, cur_k, null_id)
    return pval

def moderation_eval(sentence, key=""):
    client = OpenAI(api_key=key, organization='')
    response = client.moderations.create(input=sentence)

    return response.results[0]

def insert_banned_words(encode_input_text, args, tokenizer, test, seed):

    # words from https://www.reddit.com/r/ChatGPT/comments/10u9erx/did_it_again_list_censored_words_into_letters/
    banned_words = ['fuck', 'shit', 'damn', 'bitch', 'penis', 'cum', 'ass', 'balls', 'tits', 'pussy', 'faggot', 'dick', 'slut', 'cock', 'clit', 'fetish', 'kike', 'nude', 'fucking', 'porn', 'creampie', 'doggystyle', 'whore', 'ejaculate', 'masturbate', 'butt', 'asshole', 'rape', 'racism', 'blackmail', 'piss', 'cunt', 'motherfucker']
    banned_words_cap = [word.capitalize() for word in banned_words]
    banned_words = banned_words + banned_words_cap

    orig_tokens_num = encode_input_text.shape[-1]
    if orig_tokens_num > args.m:
        encode_input_text = encode_input_text[:args.m]
    
    orig_detection = detect(encode_input_text, seed, tokenizer, args, test)

    inserted_positions = []
    inserted_tokens = 0
    if orig_detection < 0.05:

        while inserted_tokens < args.m:
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

            if new_encode_input_text.shape[-1] > args.m:
                new_encode_input_text = new_encode_input_text[:args.m]
            new_encode_input_text = new_encode_input_text.long()
            new_detection = detect(new_encode_input_text, seed, tokenizer, args, test)
            encode_input_text = new_encode_input_text.clone()

            if new_detection > 0.05:
                break
            inserted_tokens = (np.array(inserted_positions) <= args.m).sum()
    
    new_input_text = tokenizer.decode(encode_input_text, skip_special_tokens=True)

    if inserted_tokens > 0:
        moderation_results = moderation_eval(new_input_text, args.openaikey)
    else:
        moderation_results = None

    return inserted_tokens, moderation_results

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

    vocab_size = len(tokenizer)
    if 'opt' in args.model_name_or_path:
        vocab_size = 50272

    dist = lambda x,y : gumbel_edit_score(x,y,gamma=args.gamma)

    test_stat = lambda tokens,n,k,generator,vocab_size,null=False : phi(tokens=tokens,
                                                                        n=n,
                                                                        k=k,
                                                                        generator=generator,
                                                                        key_func=gumbel_key_func,
                                                                        vocab_size=vocab_size,
                                                                        dist=dist,
                                                                        null=null,
                                                                        normalize=False,)
    
    null_results_list = []
    for i in range(14):
        cur_id = (i + 1) * 5
        file_name = './results/null_tests_' + str(cur_id) + '.pkl'
        null_results = pickle.load(open(file_name, "rb"))
        null_results = torch.sort(torch.tensor(null_results)).values
        null_results_list.append(null_results.clone())
    test = lambda tokens,seed,k,null_id : fast_permutation_test_query(tokens,
                                                    vocab_size,
                                                    args.n,
                                                    k,
                                                    seed,
                                                    test_stat,
                                                    null_results_list,
                                                    null_id,
                                                    args)

    model_type = 'llama-7b' if 'llama' in args.model_name_or_path else 'opt' if 'opt' in args.model_name_or_path else None
    if not model_type:
        return

    base_path = f'./{model_type}-results'

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(f'{base_path}/robustness', exist_ok=True)

    if args.action == 'insert':
        sentence_file_name = f'{base_path}/multiple_keys/1_key.pkl'
        inserted_token_res_file_name = f'{base_path}/robustness/inserted_token_num.pkl'
        mod_detection_res_file_name = f'{base_path}/robustness/mod_detection_res.pkl'

        inserted_token_res = []
        mod_detection_res = []

        torch.manual_seed(args.wm_key)
        seeds = torch.randint(2**32, (args.T,))

        with open(sentence_file_name, "rb") as input_file:
            loaded_data = pickle.load(input_file)
        text = loaded_data['watermark']['tokens']

        for idx in tqdm.tqdm(range(text.shape[0])):
            wm_text = text[idx]
            num_inserted_tokens, mod_res = insert_banned_words(wm_text, args, tokenizer, test, seeds[idx])
            if num_inserted_tokens > 0:
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
    
    elif args.action == 'orig-mod':
        mod_detection_res = []

        sentence_file_name = f'{base_path}/multiple_keys/1_key.pkl'
        mod_detection_res_file_name = f'{base_path}/robustness/orig_mod.pkl'

        torch.manual_seed(args.wm_key)
        seeds = torch.randint(2**32, (args.T,))

        with open(sentence_file_name, "rb") as input_file:
            loaded_data = pickle.load(input_file)
        text = loaded_data['watermark']['tokens']

        for idx in tqdm.tqdm(range(text.shape[0])):
            wm_text = text[idx]
            input_text = tokenizer.decode(wm_text, skip_special_tokens=True)

            mod_res = moderation_eval(input_text, args.openaikey)
            mod_detection_res.append(mod_res)

            if (idx + 1) % 10 == 0:
                with open(mod_detection_res_file_name, 'wb') as f:
                    pickle.dump(mod_detection_res, f)

        with open(mod_detection_res_file_name, 'wb') as f:
            pickle.dump(mod_detection_res, f)

    elif args.action == 'modify':
        inserted_token_res = []
        mod_detection_res = []

        sentence_file_name = f'{base_path}/multiple_keys/1_key.pkl'

        modify_res_file_name = f'{base_path}/robustness/modify_results.pkl'
        modify_detection_res_file_name = f'{base_path}/robustness/modify_detection_res.pkl'

        modify_orig_res_file_name = f'{base_path}/robustness/modify_orig_results.pkl'

        modify_orig_detection_res_file_name = f'{base_path}/robustness/modify_orig_detection_results.pkl'

        torch.manual_seed(args.wm_key)
        seeds = torch.randint(2**32, (args.T,))

        nlp = stanza.Pipeline(lang='en', processors='tokenize')

        with open(sentence_file_name, "rb") as input_file:
            loaded_data = pickle.load(input_file)
        text = loaded_data['watermark']['tokens']

        orig_res = []
        orig_detection_res = []
        modify_res = []
        modify_detection_res = []


        for idx in tqdm.tqdm(range(text.shape[0])):
            wm_text = text[idx]
            wm_text = tokenizer.decode(wm_text, skip_special_tokens=True)

            doc = nlp(wm_text)
            text_for_attacker = ""
            if len(doc.sentences) == 1:
                text_for_attacker = doc.sentences[0].text
            else:
                if wm_text[-1] != '.':
                    for nlp_idx in range(len(doc.sentences) - 1):
                        text_for_attacker = text_for_attacker + doc.sentences[nlp_idx].text + ' '
                else:
                    for nlp_idx in range(len(doc.sentences)):
                        text_for_attacker = text_for_attacker + doc.sentences[nlp_idx].text + ' '

            text_for_attacker_encode = tokenizer.encode(text_for_attacker, return_tensors='pt', truncation=True, max_length=2048, add_special_tokens=True)[0]

            orig_detection = detect(text_for_attacker_encode, seeds[idx], tokenizer, args, test)

            if orig_detection < 0.05:
                orig_res.append(text_for_attacker)
                orig_detection_res.append(orig_detection)
                modify_result = modify_words(text_for_attacker, args.openaikey)

                modify_res.append(modify_result)

                encode_modify_result = tokenizer.encode(modify_result, return_tensors='pt', truncation=True, max_length=2048, add_special_tokens=False)[0]

                modify_tokens_num = encode_modify_result.shape[-1]
                if modify_tokens_num > args.max_new_tokens:
                    encode_modify_result = encode_modify_result[:args.max_new_tokens]
                modified_detection_result = detect(encode_modify_result, seeds[idx], tokenizer, args, test)

                modify_detection_res.append(modified_detection_result)

                with open(modify_orig_res_file_name, 'wb') as f:
                    pickle.dump(orig_res, f)

                with open(modify_res_file_name, 'wb') as f:
                    pickle.dump(modify_res, f)

                with open(modify_detection_res_file_name, 'wb') as f:
                    pickle.dump(modify_detection_res, f)
                
                with open(modify_orig_detection_res_file_name, 'wb') as f:
                    pickle.dump(orig_detection_res, f)
            
        print(np.mean(modify_detection_res), np.std(modify_detection_res), np.mean(orig_detection_res), np.std(orig_detection_res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=70)
    parser.add_argument('--data_file', default='../data/OpenGen/inputs.jsonl', type=str)
    parser.add_argument('--action', default='insert', type=str)

    parser.add_argument('--gamma', default=0.0, type=float)
    parser.add_argument('--load_nulltest', default="./results/null_tests.pkl", type=str)
    parser.add_argument('--m', default=70, type=int)
    parser.add_argument('--k', default=0, type=int)
    parser.add_argument('--n', default=256, type=int)
    parser.add_argument('--T', default=500, type=int)
    parser.add_argument('--openaikey', default='', type=str)

    args = parser.parse_args()

    if args.k == 0: 
        args.k = args.m # k is the block size (= number of tokens)
    else:
        args.k = args.k
    args.max_new_tokens = args.m
    main(args)
