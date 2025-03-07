from time import time
import sys, os
sys.path.append('./')
import torch, pickle, copy, argparse, json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from watermarking.generation import generate, generate_rnd
from watermarking.detection import phi,fast_permutation_test
from watermarking.gumbel.score import gumbel_score,gumbel_edit_score
from watermarking.gumbel.sampler import gumbel_sampling
from watermarking.gumbel.key import gumbel_key_func

def read_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]

results = defaultdict(dict)

parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--model_name_or_path',default="meta-llama/Llama-2-7b-hf",type=str)
parser.add_argument('--seed',default=0,type=int)
parser.add_argument('--batch_size',default=1,type=int)
parser.add_argument('--m',default=70,type=int)
parser.add_argument('--k',default=0,type=int)
parser.add_argument('--n',default=256,type=int)
parser.add_argument('--T',default=500,type=int)
parser.add_argument('--prompt_tokens',default=50,type=int)
parser.add_argument('--buffer_tokens',default=20,type=int)
parser.add_argument('--n_runs',default=5000,type=int)
parser.add_argument('--max_seed',default=100000,type=int)
parser.add_argument('--offset', action='store_true')
parser.add_argument('--norm',default=1,type=int)
parser.add_argument('--gamma',default=0.0,type=float)
parser.add_argument('--edit',action='store_true')
parser.add_argument('--truncate_vocab',default=8,type=int)
parser.add_argument('--multiple_key', action='store_true')
parser.add_argument("--num_keys",default=1,type=int)
parser.add_argument('--load_nulltest',default="./results/null_tests_70.pkl",type=str)
parser.add_argument('--data_file', type=str, default='../data/OpenGen/inputs.jsonl')

args = parser.parse_args()
results['args'] = copy.deepcopy(args)

t0 = time()
torch.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if 'llama' in args.model_name_or_path:
    model = LlamaForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    return_dict=True,
                    load_in_8bit=False,
                    device_map='auto',
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                )
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)

else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map='auto',)

vocab_size = model.get_output_embeddings().weight.shape[0]
eff_vocab_size = vocab_size - args.truncate_vocab
print(f'Loaded the model (t = {time()-t0} seconds)')

dataset = load_dataset("c4", "realnewslike", split="train", streaming=True)

T = args.T                  # number of prompts/generations
n_batches = int(np.ceil(T / args.batch_size)) # number of batches
prompt_tokens = args.prompt_tokens      # minimum prompt length
new_tokens = args.m     # number of tokens to generate
buffer_tokens = args.buffer_tokens
if args.k == 0: 
    k = args.m # k is the block size (= number of tokens)
else:
    k = args.k     
n = args.n     # watermark key length

# this is the "key" for the watermark
# for now each generation gets its own key
torch.manual_seed(args.seed)
seeds = torch.randint(2**32, (T,))

generate_watermark = lambda prompt,seed : generate(model,
                                                    prompt,
                                                    vocab_size,
                                                    n,
                                                    new_tokens+buffer_tokens,
                                                    seed,
                                                    gumbel_key_func,
                                                    gumbel_sampling,
                                                    random_offset=args.offset,
                                                    args=args)

if args.edit is True:
    dist = lambda x,y : gumbel_edit_score(x,y,gamma=args.gamma)
else:
    dist = lambda x,y : gumbel_score(x,y)
test_stat = lambda tokens,n,k,generator,vocab_size,null=False : phi(tokens=tokens,
                                                                    n=n,
                                                                    k=k,
                                                                    generator=generator,
                                                                    key_func=gumbel_key_func,
                                                                    vocab_size=vocab_size,
                                                                    dist=dist,
                                                                    null=null,
                                                                    normalize=False)

data = read_file(args.data_file)

generator = torch.Generator()

null_results = pickle.load(open(args.load_nulltest, "rb"))

null_results = torch.sort(torch.tensor(null_results)).values

test = lambda tokens,seed : fast_permutation_test(tokens,
                                                  vocab_size,
                                                  n,
                                                  k,
                                                  seed,
                                                  test_stat,
                                                  null_results)



t1 = time()

prompts = []
itm = 0
while itm < T:
    cur_data = data[itm]
    if "gold_completion" not in cur_data and 'targets' not in cur_data:
        continue
    else:
        prefix = cur_data['prefix']
    text = prefix

    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
    prompts.append(tokens)

    itm += 1

results['prompts'] = copy.deepcopy(prompts)

null_samples = []
watermarked_samples = []
for idx in tqdm(range(args.T)):
    if not args.multiple_key:
        null_samples.append(generate_rnd(prompts[idx].unsqueeze(0),new_tokens+buffer_tokens,model)[:,prompts[idx].shape[-1]:])
    watermarked_samples.append(generate_watermark(prompts[idx].unsqueeze(0), seeds[idx:idx+1])[:,prompts[idx].shape[-1]:])
if not args.multiple_key:
    null_samples = torch.vstack(null_samples)
    results['null']['tokens'] = copy.deepcopy(null_samples)
    null_samples = torch.clip(null_samples,max=eff_vocab_size-1)

watermarked_samples = torch.vstack(watermarked_samples)
results['watermark']['tokens'] = copy.deepcopy(watermarked_samples)
watermarked_samples = torch.clip(watermarked_samples,max=eff_vocab_size-1)

print(f'Generated samples in (t = {time()-t1} seconds)')

pvals_watermark = []
pvals_null = []
pbar = tqdm(total=T)
for itm in range(T):
    if not args.multiple_key:
        null_sample = null_samples[itm]
        null_sample = tokenizer.decode(null_sample, skip_special_tokens=True)
        null_sample = tokenizer.encode(null_sample,
                                    return_tensors='pt',
                                    truncation=True,
                                    max_length=2048)[0]
        if len(null_sample) < new_tokens + 1:
            null_sample = torch.nn.functional.pad(null_sample,(new_tokens-len(null_sample),0),"constant",0)
        else:
            null_sample = null_sample[1:new_tokens+1]
        pval = test(null_sample, seeds[itm])
        pvals_null.append(pval)

    watermarked_sample = watermarked_samples[itm]
    watermarked_sample = tokenizer.decode(watermarked_sample, skip_special_tokens=True)
    watermarked_sample = tokenizer.encode(watermarked_sample,
                                          return_tensors='pt',
                                          truncation=True,
                                          max_length=2048)[0]
    if len(watermarked_sample) < new_tokens + 1:
        watermarked_sample = torch.nn.functional.pad(watermarked_sample,(new_tokens-len(watermarked_sample),0),"constant",0)
    else:
        watermarked_sample = watermarked_sample[1:new_tokens+1]
    pval = test(watermarked_sample, seeds[itm])
    pvals_watermark.append(pval)

    pbar.update(1)

pbar.close()
print(f'Ran the experiment (t = {time()-t1} seconds)')

results['watermark']['pvals'] = torch.tensor(pvals_watermark)
if not args.multiple_key:
    results['null']['pvals'] = torch.tensor(pvals_null)

print('watermarked', results['watermark']['pvals'])
if not args.multiple_key:
    print('nonwatermarked', results['null']['pvals'])

if 'llama' in args.model_name_or_path:
    result_folder = './llama-7b-results'
else:
    result_folder = './opt-results'

if not os.path.exists(result_folder):
    os.makedirs(result_folder, exist_ok=True)
if not os.path.exists(os.path.join(result_folder, 'multiple_keys')):
    os.makedirs(os.path.join(result_folder, 'multiple_keys'), exist_ok=True)

file_name = str(args.num_keys) + '_key.pkl'

pickle.dump(results, open(os.path.join(result_folder, 'multiple_keys', file_name), "wb"))
