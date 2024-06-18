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


from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import json
import random
from PIL import Image
import os
import argparse


qasa_filtered_annotations_path = '../../../datasets/test-B/SPIQA_testB.json'

with open(qasa_filtered_annotations_path, "r") as f:
  qasa_data = json.load(f)


_QASA_IMAGE_ROOT = "../../../datasets/test-B/SPIQA_testB_Images"

def prepare_inputs(paper, question_idx):
    all_figures = list(paper['all_figures_tables'].keys())
    referred_figures = list(set(paper['referred_figures_tables'][question_idx]))
    answer = paper['composition'][question_idx]

    referred_figures_captions = []
    for figure in referred_figures:
        referred_figures_captions.append(paper['all_figures_tables'][figure])

    return answer, all_figures, referred_figures, referred_figures_captions


_PROMPT_1 = "Caption: <caption>. Is the input image and caption helpful to answer the following question. Answer in one word - Yes or No. Question: <question>. "
_PROMPT_2 = "Caption: <caption>. Please provide a brief answer to the following question after looking into the input image and caption. Question: <question>."


def infer_instructblip(qasa_data, args):

  processor = InstructBlipProcessor.from_pretrained(args.model_id)
  model = InstructBlipForConditionalGeneration.from_pretrained(args.model_id, load_in_4bit=True, torch_dtype=torch.float16)

  _RESPONSE_ROOT = args.response_root
  os.makedirs(_RESPONSE_ROOT, exist_ok=True)

  for paper_id, paper in sorted(qasa_data.items(), key=lambda x: random.random()):
    if os.path.exists(os.path.join(_RESPONSE_ROOT, str(paper_id) + '_response.json')):
      continue
    response_paper = {}

    try:
      for question_idx, question in enumerate(paper['question']):

        answer, all_figures, referred_figures, referred_figures_captions = prepare_inputs(paper, question_idx)

        answer_dict = {}

        for _idx, figure in enumerate(referred_figures):
          
          caption = referred_figures_captions[_idx]
              
          instructblip_prompt_1 = _PROMPT_1.replace('<caption>', caption).replace('<question>', question)
          instructblip_prompt_2 = _PROMPT_2.replace('<caption>', caption).replace('<question>', question)
            
          image = Image.open(os.path.join(_QASA_IMAGE_ROOT, figure))
          image = image.resize((args.image_resolution, args.image_resolution))
            
          inputs_1 = processor(images=image, text=instructblip_prompt_1, return_tensors="pt").to(device="cuda", dtype=torch.float16)

          # autoregressively generate an answer
          outputs_1 = model.generate(
                  **inputs_1,
                  num_beams=1,
                  max_new_tokens=50,
                  min_length=1,
                  top_p=0.9,
                  repetition_penalty=1.5,
                  length_penalty=1.0,
                  temperature=1,
          )
          outputs_1[outputs_1 == 0] = 2 # this line can be removed once https://github.com/huggingface/transformers/pull/24492 is fixed
          generated_text_1 = processor.batch_decode(outputs_1, skip_special_tokens=True)[0].strip()

          inputs_2 = processor(images=image, text=instructblip_prompt_2, return_tensors="pt").to(device="cuda", dtype=torch.float16)

          # autoregressively generate an answer
          outputs_2 = model.generate(
                  **inputs_2,
                  num_beams=5,
                  max_new_tokens=256,
                  min_length=1,
                  top_p=0.9,
                  repetition_penalty=1.5,
                  length_penalty=1.0,
                  temperature=1,
          )
          outputs_2[outputs_2 == 0] = 2 # this line can be removed once https://github.com/huggingface/transformers/pull/24492 is fixed
          generated_text_2 = processor.batch_decode(outputs_2, skip_special_tokens=True)[0].strip()
          

          answer_dict.update({figure: [generated_text_1, generated_text_2]})
      
          print(answer_dict[figure])
          print('-----------------')

        question_key = paper['question_key'][question_idx]
        response_paper.update({question_key: {'question': question, 'response': answer_dict,
                                              'referred_figures_names': referred_figures, 'answer': answer}})   

    except:
        print('Error in generating.')
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", load_in_4bit=True, torch_dtype=torch.float16)
        continue

    with open(os.path.join(_RESPONSE_ROOT, str(paper_id) + '_response.json'), 'w') as f:
      json.dump(response_paper, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate on Qasa/Qasper.')
    parser.add_argument('--model_id', type=str, default="Salesforce/instructblip-vicuna-7b", help='InstructBLIP Model id.')
    parser.add_argument('--response_root', type=str, help='Response Root path.')
    parser.add_argument('--image_resolution', type=int, help='Response Root path.')
    args = parser.parse_args()
    
    
    infer_instructblip(qasa_data, args)
