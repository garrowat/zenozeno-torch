import tensorflow as tf
from transformers import *

# This file includes code which was modified from https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
tokenizer = AutoTokenizer.from_pretrained("gpt2", eos_token=".")
model = AutoModelWithLMHead.from_pretrained("./models", cache_dir=None, from_tf=False, state_dict=None)

sequence = f"Donald Trump recently"

tf.random.set_random_seed(0)

input = tokenizer.encode(sequence, return_tensors="pt")
sample_outputs = model.generate(
  input,
  do_sample=True,
  max_length=150,
  top_k=40,
  top_p=0.9,
  temperature=0.7,
  eos_token_id=tokenizer.eos_token_id,
  num_return_sequences=3,
)

for i, sample_output in enumerate(sample_outputs):
  print("{}: {}.".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))