# Flask stuff
from flask import Flask, url_for, make_response
from flask_classful import FlaskView, route
from flask_cors import CORS, cross_origin

import json

from transformers import GPT2LMHeadModel, AutoTokenizer

app = Flask(__name__)
CORS(app)

def output_json(data, code, headers=None):
    content_type = 'application/json'
    dumped = json.dumps(data)
    if headers:
        headers.update({'Content-Type': content_type})
    else:
        headers = {'Content-Type': content_type}
    response = make_response(dumped, code, headers)
    return response

tokenizer = AutoTokenizer.from_pretrained("gpt2", eos_token=".")
model = GPT2LMHeadModel.from_pretrained("./models", cache_dir=None, from_tf=False, state_dict=None)

def generateQuotes(input):
  input = tokenizer.encode(input, return_tensors="pt")
  outputs = model.generate(
    input,
    do_sample=True,
    max_length=100,
    top_k=40,
    top_p=0.9,
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id,
    num_return_sequences=3,
  )
  return outputs;

class QuoteView(FlaskView):
  route_base = '/zenozeno'
  representations = {'application/json': output_json}

  @route('/quote/<input>')
  def get_proverb(self, input):
      input = str(input)
      quotes = []
      for sample_output in generateQuotes(input):
        quotes.append(tokenizer.decode(sample_output, skip_special_tokens=True))
      return {"quotes": quotes}

QuoteView.register(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0')