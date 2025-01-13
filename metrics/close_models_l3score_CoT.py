# Copyright 2024 Google LLC
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""CoT QA evaluation"""


import json
import tqdm
import argparse
import re
import os
import ast
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


  for paper_response in tqdm.tqdm(os.listdir(_RESPONSE_ROOT)):
    with open(os.path.join(_RESPONSE_ROOT, paper_response), 'r') as f:
      saved_results = json.load(f)

    for key, value in saved_results.items():
      if len(value['response'].split('The answer is')) == 2:
        answer = value['response'].split('The answer is')[-1]
      elif len(value['response'].split('The answer to the question is')) == 2:
        answer = value['response'].split('The answer to the question is')[-1]
      else:
        answer = value['response']
      question = value['question']
      try:
        gt = value['answer']
        if type(answer) != str:
          answer = ''
        all += 1
        prompt_current = _PROMPT.replace('<question>', question).replace('<GT>', gt).replace('<answer>', answer)
        response, prob_yes = client.call_openai_with_score(
        prompt=prompt_current,
        suffixes=_SUFFIXES_TO_SCORE,
        complement_suffixes=_COMPLEMENT_SUFFIXES,
        output_prefix=''
        )
        score += prob_yes
      except:
        failed_parsing += 1
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



image = re.compile(r'Image \b([0-9]|10|11|12|13|14|15)\b', flags=re.IGNORECASE)
image_2 = re.compile(r"'Image': \b([0-9]|10|11|12|13|14|15)\b", flags=re.IGNORECASE)

def acc_top_1(_RESPONSE_ROOT):
  correct, all = 0, 0
  failed_parsing = 0
  for paper_response in os.listdir(_RESPONSE_ROOT):
    with open(os.path.join(_RESPONSE_ROOT, paper_response), 'r') as f:
      saved_results = json.load(f)

    for key, value in saved_results.items():
      value['response'] = value['response'].split('The answer is:')[0]
      try:
        image_response = ast.literal_eval(value['response'])
        answer = image_response['Image']
        gt = value['referred_figures_indices']
        if answer in gt:
          correct += 1
          all += 1
        else:
          all += 1
      except:
        try:
          answer = int(image.findall(value['response'])[0])
          gt = value['referred_figures_indices']
          if answer in gt:
            correct += 1
            all += 1
          else:
            all += 1
        except:
          try:
            answer = int(image_2.findall(value['response'])[0])
            gt = value['referred_figures_indices']
            if answer in gt:
              correct += 1
              all += 1
            else:
              all += 1
          except:
            failed_parsing += 1
            all += 1

  print("Retrieval Accuracy: {}".format(correct/all))
  print("Examples with Failed Parsing: {}".format(failed_parsing))

acc_top_1(_RESPONSE_ROOT=args.response_root)
