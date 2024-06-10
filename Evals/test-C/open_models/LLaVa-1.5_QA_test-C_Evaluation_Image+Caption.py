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
"""
### Example Usage - CUDA_VISIBLE_DEVICES=7 python LLaVa-1.5_QA_Qasa_Evaluation_Image_only.py \ 
### --model_id llava-hf/llava-1.5-7b-hf --image_resolution 224 --response_root ../Responses/Test
"""
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch
import json
import random
from PIL import Image
import os
import argparse

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

qasper_filtered_annotations_path = '../Datasets/QASPER/qasper_0520.json'
with open(qasper_filtered_annotations_path, "r") as f:
  qasper_data = json.load(f)

_QASPER_IMAGE_ROOT = "../Datasets/QASPER/qasper_figures/all"

def prepare_inputs(paper, question_idx):
    all_figures = [element['file'] for element in paper['figures_and_tables']]
    referred_figures = list(set(paper['referred_figures_tables'][question_idx]))
    all_figures_captions_dict = {}
    for element in paper['figures_and_tables']:
        all_figures_captions_dict.update({element['file']: element['caption']})
    
    if paper['answer'][question_idx]['free_form_answer'] != '': 
        answer = paper['answer'][question_idx]['free_form_answer']
    elif paper['answer'][question_idx]['yes_no'] != None:
        if paper['answer'][question_idx]['yes_no'] == False:
            answer = 'No'
        elif paper['answer'][question_idx]['yes_no'] == True:
            answer = 'Yes'
        else:
            raise ValueError
    else:
        raise ValueError
    
    referred_figures_captions = []
    for figure in referred_figures:
        referred_figures_captions.append(all_figures_captions_dict[figure])

    return answer, all_figures, referred_figures, referred_figures_captions


_PROMPT = ["""USER: <image>\n Caption: <caption> Is the input image and caption helpful to answer the following question. Answer in one word - Yes or No. Question: <question>.\nASSISTANT:""", 
           """USER: <image>\n Caption: <caption> Please provide a brief answer to the following question after looking into the input image and caption. Question: <question>.\nASSISTANT:"""]


def infer_llava(qasper_data, args):

  model_id = args.model_id
  processor = AutoProcessor.from_pretrained(model_id)
  model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

  _RESPONSE_ROOT = args.response_root
  os.makedirs(_RESPONSE_ROOT, exist_ok=True)

  for paper_id, paper in sorted(qasper_data.items(), key=lambda x: random.random()):
    if os.path.exists(os.path.join(_RESPONSE_ROOT, str(paper_id) + '_response.json')):
      continue
    response_paper = {}

    try:
      for question_idx, question in enumerate(paper['question']):

        answer, all_figures, referred_figures, referred_figures_captions = prepare_inputs(paper, question_idx)

        answer_dict = {}

        for _idx, figure in enumerate(referred_figures):
          
          caption = referred_figures_captions[_idx]

          contents = [_PROMPT[0].replace('<caption>', caption).replace('<question>', question),
                      _PROMPT[1].replace('<caption>', caption).replace('<question>', question)]
          
          image = Image.open(os.path.join(_QASPER_IMAGE_ROOT, paper['arxiv_id'], figure))
          image = image.resize((args.image_resolution, args.image_resolution))
          inputs = processor(contents, [image, image], padding=True, return_tensors="pt").to("cuda")

          output = model.generate(**inputs, max_new_tokens=100)
          generated_text = processor.batch_decode(output, skip_special_tokens=True)

          answer_dict.update({figure: [generated_text[0].split("ASSISTANT:")[-1], generated_text[1].split("ASSISTANT:")[-1]]})
        
          print(answer_dict[figure])
          print('-----------------')

        question_key = paper['question_key'][question_idx]
        response_paper.update({question_key: {'question': question, 'response': answer_dict,
                                            'referred_figures_names': referred_figures, 'answer': answer}})   

    except:
      print('Error in generating.')
      processor = AutoProcessor.from_pretrained(model_id)
      model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
      continue

    with open(os.path.join(_RESPONSE_ROOT, str(paper_id) + '_response.json'), 'w') as f:
      json.dump(response_paper, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate on Qasa/Qasper.')
    parser.add_argument('--model_id', type=str, default='llava-hf/llava-1.5-7b-hf', help='Huggingface Model id.')
    parser.add_argument('--response_root', type=str, help='Response Root path.')
    parser.add_argument('--image_resolution', type=int, help='Response Root path.')
    args = parser.parse_args()
    
    
    infer_llava(qasper_data, args)
