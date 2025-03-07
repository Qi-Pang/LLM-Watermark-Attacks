import torch
from collections import Counter
import numpy as np

def generate(model,prompts,vocab_size,n,m,seeds,key_func,sampler,random_offset=True,args=None):
    batch_size = len(prompts)

    generator = torch.Generator()
    xis,pis = [],[]
    for seed in seeds:
        generator.manual_seed(int(seed))
        xi,pi = key_func(generator,n,vocab_size)
        xis.append(xi.unsqueeze(0))
        pis.append(pi.unsqueeze(0))
    xis = torch.vstack(xis)
    pis = torch.vstack(pis)

    # deliberately not controlling this randomness with the generator
    if random_offset:
        offset = torch.randint(n,size=(batch_size,))
    else:
        offset = torch.zeros(size=(batch_size,),dtype=torch.int64)
    if args is not None and args.multiple_key:
        offsets = torch.randint(n, size=(args.num_keys, batch_size,))

    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1], dim=-1).cpu()

        if args is not None and args.multiple_key:
            # freq_tokens = []
            freq_tokens_counter = []
            for j in range(args.num_keys):
                cur_offset = offsets[j]
                cur_tokens = sampler(probs, pis, xis[torch.arange(batch_size),(cur_offset.squeeze()+i)%n])
                # freq_tokens.append(cur_tokens)
                freq_tokens_counter.append(cur_tokens.clone().numpy())
            freq_tokens_counter = np.array(freq_tokens_counter).squeeze(-1).T
            tokens = cur_tokens.clone()
            for batch_id in range(freq_tokens_counter.shape[0]):
                frequencies = Counter(freq_tokens_counter[batch_id])
                result_freq = [frequencies[item] for item in freq_tokens_counter[batch_id]]
                max_freq = max(result_freq)
                select_key = result_freq.index(max_freq)
                selected_token = freq_tokens_counter[batch_id][select_key]
                # print(selected_token)
                # print(tokens[0][batch_id])
                tokens[batch_id][0] = selected_token
            tokens = tokens.to(model.device)
        else:
            tokens = sampler(probs, pis, xis[torch.arange(batch_size),(offset.squeeze()+i)%n]).to(model.device)

        inputs = torch.cat([inputs, tokens], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()

def generate_query(model,prompts,vocab_size,n,m,seeds,key_func,sampler,random_offset=True,args=None):
    batch_size = len(prompts)

    generator = torch.Generator()
    xis,pis = [],[]
    for seed in seeds:
        generator.manual_seed(int(seed))
        xi,pi = key_func(generator,n,vocab_size)
        xis.append(xi.unsqueeze(0))
        pis.append(pi.unsqueeze(0))
    xis = torch.vstack(xis)
    pis = torch.vstack(pis)

    # deliberately not controlling this randomness with the generator
    if random_offset:
        offset = torch.randint(n,size=(batch_size,))
    else:
        offset = torch.zeros(size=(batch_size,),dtype=torch.int64)
    if args is not None and args.multiple_key:
        offsets = torch.randint(n, size=(args.num_keys, batch_size,))

    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1], dim=-1).cpu()

        tokens = sampler(probs, pis, xis[torch.arange(batch_size),(offset.squeeze()+i)%n]).to(model.device)

        inputs = torch.cat([inputs, tokens], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()

# generate unwatermarked completions of token length m given list of prompts
def generate_rnd(prompts,m,model):
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1], dim=-1)
        
        tokens = torch.multinomial(probs,1)
        inputs = torch.cat([inputs, tokens], dim=1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
    
    return inputs.detach().cpu()
