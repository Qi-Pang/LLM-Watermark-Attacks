# No Free Lunch in LLM Watermarking: Trade-offs in Watermarking Design Choices

----

Implementation of NeurIPS 24 paper. [*No Free Lunch in LLM Watermarking: Trade-offs in Watermarking Design Choices.*](https://openreview.net/pdf?id=rIOl7KbSkv])


----

Attacks on the [Unigram](https://github.com/XuandongZhao/Unigram-Watermark) watermark.

----

### Watermark removal attacks exploiting the use of multiple watermark keys.

**Baseline detection results (no multiple keys)**

Run the following command to obtain watermark detection results without multiple keys. The results will be saved in `./opt-results/multiple_keys` folder.
For the LLAMA-2-7B model, pass `meta-llama/Llama-2-7b-hf` to the `--model_name_or_path` argument. Results for this model will be stored in the `./llama-7b-results/multiple_keys` folder.
```bash
python attack_multiple_keys.py --model_name_or_path facebook/opt-1.3b
```

**Attacks exploiting multiple keys**

To simulate watermark removal attacks using multiple keys, run the following command. You can adjust the `--num_keys` parameter to specify the number of keys used.
```bash
python attack_multiple_keys.py --model_name_or_path facebook/opt-1.3b --multiple_key --num_keys 7
```

----

### Watermark spoofing attacks exploiting the robustness property.

**Attack using toxic token insertion**

This attack involves inserting toxic tokens into the text while preserving the watermark. Use the following command, ensuring you provide your OpenAI API key via the `--openaikey` argument:
```bash
python attack_robustness.py --model_name_or_path facebook/opt-1.3b --action insert --openaikey 'YOUR_OPENAI_API_KEY'
```

**Attack using inaccurate content modification**

This attack modifies the text to introduce inaccuracies while preserving the watermark. Use the following command with your OpenAI API key:
```bash
python attack_robustness.py --model_name_or_path facebook/opt-1.3b --action modify --openaikey 'YOUR_OPENAI_API_KEY'
```

*Note: Perform the removal attack exploiting multiple keys first before running these spoofing attacks.*

----

### Attacks exploiting the public detection APIs

**Watermark removal attack exploiting the public detection API**

This attack attempts to remove watermarks using public detection APIs:
```bash
python attack_query.py --model_name_or_path facebook/opt-1.3b --action removal
```

**Watermark spoofing attack exploiting the public detection API**

This attack generates text that mimics watermarked content:
```bash
python attack_query.py --model_name_or_path facebook/opt-1.3b --action spoofing
```

**Evaluating Differential Privacy (DP) Defense**

To evaluate the effectiveness of a DP-based defense mechanism, use the following command:
```bash
python attack_query.py --model_name_or_path facebook/opt-1.3b --action spoofing-defense
```

Benchmark the detection accuracy of the original watermarked and unwatermarked sentences:
```bash
python attack_query.py --model_name_or_path facebook/opt-1.3b --action dp-benchmark
```