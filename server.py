# AI Stuff
import tensorflow as tf
from transformers import *

# Flask stuff
from flask import Flask, url_for, make_response
from flask_classful import FlaskView, route
from flask_cors import CORS, cross_origin

import json

app = Flask(__name__)
CORS(app)

tokenizer = AutoTokenizer.from_pretrained("gpt2", eos_token=".")
model = AutoModelWithLMHead.from_pretrained("./models", cache_dir=None, from_tf=False, state_dict=None)

sequence = f"A good life"

input = tokenizer.encode(sequence, return_tensors="pt")
sample_outputs = model.generate(
  input,
  do_sample=True,
  max_length=100,
  top_k=40,
  top_p=0.9,
  temperature=0.7,
  eos_token_id=tokenizer.eos_token_id,
  num_return_sequences=3,
)

for i, sample_output in enumerate(sample_outputs):
  print("{}: {}.".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))