# No Free Lunch in LLM Watermarking: Trade-offs in Watermarking Design Choices

----

This repository contains the implementation of NeurIPS 2024 paper, [*No Free Lunch in LLM Watermarking: Trade-offs in Watermarking Design Choices.*](https://openreview.net/pdf?id=rIOl7KbSkv])


----

For the watermarking schemes, we borrow the code from their original implementations [KGW](https://github.com/jwkirchenbauer/lm-watermarking), [Unigram](https://github.com/XuandongZhao/Unigram-Watermark), and [Exp](https://github.com/jthickstun/watermark).

For the watermark stealing attack, we use their [original implementation](https://github.com/eth-sri/watermark-stealing), where the attacker gathers approximately 2.2 million tokens to estimate the watermark pattern and subsequently launches spoofing attacks.
The attacker operates under the assumption of access to the unwatermarked token distribution.

Detailed instructions for running our attacks on each watermarking scheme are provided in the respective directories: [KGW-Watermark](./KGW-Watermark/), [Unigram-Watermark](./Unigram-Watermark/), and [Exp-Watermark](./Exp-Watermark/).

----
#### Environment Setup
To set up the environment, follow these steps:
```bash
conda create -n "llmwmatt" python=3.9.0
conda activate llmwmatt
pip install -r requirements.txt

```

----
#### Current Progress
* **KGW Watermark**: Code is mostly cleaned and ready for use.
* **Unigram, Exp, and Watermark Stealing**: Code cleaning is in progress. Stay tuned for updates.
----

#### Reference

This repository builds on the following implementations:

* https://github.com/jwkirchenbauer/lm-watermarking

* https://github.com/XuandongZhao/Unigram-Watermark

* https://github.com/jthickstun/watermark

* https://github.com/eth-sri/watermark-stealing

----

#### Citation

```bash
@inproceedings{
pang2024no,
title={No Free Lunch in {LLM} Watermarking: Trade-offs in Watermarking Design Choices},
author={Qi Pang and Shengyuan Hu and Wenting Zheng and Virginia Smith},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=rIOl7KbSkv}
}
```