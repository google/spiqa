"""LLM Log-Likelihood Scoring for OpenAI GPT models.

Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import json
import tqdm
import argparse
import glob
import numpy as np
from llmlogscore.llmlogscore import OpenAIClient

_SUFFIXES_TO_SCORE = [' yes', ' yeah']
_COMPLEMENT_SUFFIXES = [' no']

parser = argparse.ArgumentParser(description='Evaluate on Qasa/Qasper.')
parser.add_argument('--response_root', type=str, help='Response Root path.')
parser.add_argument('--openai_api_key', type=str, help='OpenAI API key.')
args = parser.parse_args()

_PROMPT = 'You are given a question, ground-truth answer, and a candidate answer. Question: <question> \nGround-truth answer: <GT> \nCandidate answer: <answer> \n\
Is the semantic meaning of the ground-truth and candidate answers similar? Answer in one word - Yes or No.'

def calculate_all_metrics(_RESPONSE_ROOT, client, _PROMPT):
  score = 0
  all = 0
  failed_parsing = 0
  no_samples = 0

  for paper_response in tqdm.tqdm(glob.glob(_RESPONSE_ROOT + '/*.json')):
    with open(paper_response, 'r') as f:
      saved_results = json.load(f)

    for _, value in saved_results.items():

      image_response = value['response']
      gt = value['answer']
      question = value['question']

      flag = 0

      for referred_figure, answer in image_response.items():

        if 'no' in answer[0].lower():
          no_samples += 1
          continue

        else:
            all += 1
            prompt_current = _PROMPT.replace('<question>', question).replace('<GT>', gt).replace('<answer>', answer[1])
            response, prob_yes = client.call_openai_with_score(
            prompt=prompt_current,
            suffixes=_SUFFIXES_TO_SCORE,
            complement_suffixes=_COMPLEMENT_SUFFIXES,
            output_prefix=''
            )
            score += prob_yes

      if flag == 0: ## For a questions, if the model says No for every referred image, then we consider the case as a failure
        all += 1

  print('Printing Metric ..')
  print('Metric: ', score/all)
  print("Examples with Failed Parsing: {}".format(failed_parsing))
  print("all: ", all)
  print("no_samples: ", no_samples)

client = OpenAIClient(
    model_name='gpt-4o',
    api_key=args.openai_api_key,
    json_output_path='./saved_output_l3score/',
)

calculate_all_metrics(args.response_root, client, _PROMPT)
