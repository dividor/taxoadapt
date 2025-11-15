from transformers import pipeline
from vllm import LLM, SamplingParams
import numpy as np
import torch
import os
from tqdm import tqdm
from dotenv import load_dotenv
from vllm.sampling_params import GuidedDecodingParams
from api.llm_provider import get_llm_provider

# Load environment variables
load_dotenv()


# map each term in text to word_id
def get_vocab_idx(split_text: str, tok_lens):

	vocab_idx = {}
	start = 0

	for w in split_text:
		# print(w, start, start + len(tok_lens[w]))
		if w not in vocab_idx:
			vocab_idx[w] = []

		vocab_idx[w].extend(np.arange(start, start + len(tok_lens[w])))

		start += len(tok_lens[w])

	return vocab_idx

def get_hidden_states(encoded, data_idx, model, layers, static_emb):
	"""Push input IDs through model. Stack and sum `layers` (last four by default).
	Select only those subword token outputs that belong to our word of interest
	and average them."""
	with torch.no_grad():
		output = model(**encoded)

	# Get all hidden states
	states = output.hidden_states
	# Stack and sum all requested layers
	output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

	# Only select the tokens that constitute the requested word

	for w in data_idx:
		static_emb[w] += output[data_idx[w]].sum(dim=0).cpu().numpy()

def chunkify(text, token_lens, length=512):
	chunks = [[]]
	split_text = text.split()
	count = 0
	for word in split_text:
		new_count = count + len(token_lens[word]) + 2 # 2 for [CLS] and [SEP]
		if new_count > length:
			chunks.append([word])
			count = len(token_lens[word])
		else:
			chunks[len(chunks) - 1].append(word)
			count = new_count
	
	return chunks

def constructPrompt(args, init_prompt, main_prompt):
	if (args.llm == 'gpt'):
		return [
            {"role": "system", "content": init_prompt},
            {"role": "user", "content": main_prompt}]
	else:
		return init_prompt + "\n\n" + main_prompt

def initializeLLM(args):
	args.client = {}

	args.client['vllm'] = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", tensor_parallel_size=4, gpu_memory_utilization=0.95, 
						   max_num_batched_tokens=4096, max_num_seqs=1000, enable_prefix_caching=True)

	if args.llm == 'gpt':
		# Use unified LLM provider - reads from LLM_PROVIDER in .env (required)
		args.client[args.llm] = get_llm_provider()
	
	return args

def promptGPT(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
	outputs = []
	provider = args.client['gpt']
	
	for messages in tqdm(prompts):
		# Use the unified provider interface
		response = provider.chat(
			messages=messages,
			json_mode=json_mode,
			max_new_tokens=max_new_tokens,
			temperature=temperature,
			top_p=top_p
		)
		outputs.append(response)
	return outputs

def promptLlamaVLLM(args, prompts, schema=None, max_new_tokens=1024, temperature=0.1, top_p=0.99):
    if schema is None:
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    else:
        guided_decoding_params = GuidedDecodingParams(json=schema.model_json_schema())
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, 
                                    guided_decoding=guided_decoding_params)
    generations = args.client['vllm'].generate(prompts, sampling_params)
    
    outputs = []
    for gen in generations:
        outputs.append(gen.outputs[0].text)

    return outputs

def promptLLM(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
	if args.llm == 'gpt':
		return promptGPT(args, prompts, schema, max_new_tokens, json_mode, temperature, top_p)
	else:
		return promptLlamaVLLM(args, prompts, schema, max_new_tokens, temperature, top_p)
	