import torch

def gumbel_sampling(probs,pi,xi):
    return torch.argmax(xi ** (1/torch.gather(probs, 1, pi)),axis=1).unsqueeze(-1)

def gumbel_query(probs,pi,xi):
    new_probs = xi ** (1/torch.gather(probs, 1, pi))
    new_probs = new_probs / torch.sum(new_probs, axis=1)
    return new_probs