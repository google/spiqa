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


from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
import torch
import json
import random
from PIL import Image
import os
import argparse


qasper_filtered_annotations_path = '../../../datasets/test-C/SPIQA_testC.json'
with open(qasper_filtered_annotations_path, "r") as f:
  qasper_data = json.load(f)


_QASPER_IMAGE_ROOT = "../../../datasets/test-C/SPIQA_testC_Images"

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

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids


_PROMPT_1 = "Is the input image and caption helpful to answer the following question. Answer in one word - Yes or No. Caption: <caption>. Question: <question>. "
_PROMPT_2 = "Please provide a brief answer to the following question after looking into the input image and caption. Caption: <caption>. Question: <question>."

def apply_prompt_template(prompt):
    s = (
            '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
            f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n'
        )
    return s 


def infer_xgenmm(qasper_data, args):

    model_name_or_path = "Salesforce/blip3-phi3-mini-instruct-r-v1"
    model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = model.update_special_tokens(tokenizer)
    model = model.cuda()

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
                    
                    blip3_prompt_1 = _PROMPT_1.replace('<caption>', caption).replace('<question>', question)
                    blip3_prompt_2 = _PROMPT_2.replace('<caption>', caption).replace('<question>', question)
                
                    image = Image.open(os.path.join(_QASPER_IMAGE_ROOT, paper['arxiv_id'], figure)).convert('RGB')
                    image = image.resize((args.image_resolution, args.image_resolution))

                    inputs_1 = image_processor([image], return_tensors="pt", image_aspect_ratio='anyres')
                    prompt_1 = apply_prompt_template(blip3_prompt_1)
                    language_inputs_1 = tokenizer([prompt_1], return_tensors="pt")
                    inputs_1.update(language_inputs_1)
                    inputs_1 = {name: tensor.cuda() for name, tensor in inputs_1.items()}
                    generated_text_1 = model.generate(**inputs_1, image_size=[image.size],
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    do_sample=False, max_new_tokens=768, top_p=None, num_beams=1,
                                                    stopping_criteria = [EosListStoppingCriteria()],
                                                    )
                    prediction_1 = tokenizer.decode(generated_text_1[0], skip_special_tokens=True).split("<|end|>")[0]


                    
                    inputs_2 = image_processor([image], return_tensors="pt", image_aspect_ratio='anyres')
                    prompt_2 = apply_prompt_template(blip3_prompt_2)
                    language_inputs_2 = tokenizer([prompt_2], return_tensors="pt")
                    inputs_2.update(language_inputs_2)
                    inputs_2 = {name: tensor.cuda() for name, tensor in inputs_2.items()}
                    generated_text_2 = model.generate(**inputs_2, image_size=[image.size],
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    do_sample=False, max_new_tokens=768, top_p=None, num_beams=1,
                                                    stopping_criteria = [EosListStoppingCriteria()],
                                                    )
                    prediction_2 = tokenizer.decode(generated_text_2[0], skip_special_tokens=True).split("<|end|>")[0]      
                    
                    answer_dict.update({figure: [prediction_1, prediction_2]})
            
                    print(answer_dict[figure])
                    print('-----------------')

                question_key = paper['question_key'][question_idx]
                response_paper.update({question_key: {'question': question, 'response': answer_dict,
                                              'referred_figures_names': referred_figures, 'answer': answer}})   

        except:
            print('Error in generating.')
            model_name_or_path = "Salesforce/blip3-phi3-mini-instruct-r-v1"
            model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
            image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
            tokenizer = model.update_special_tokens(tokenizer)
            model = model.cuda()
            continue

        with open(os.path.join(_RESPONSE_ROOT, str(paper_id) + '_response.json'), 'w') as f:
            json.dump(response_paper, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate on Qasa/Qasper.')
    parser.add_argument('--response_root', type=str, help='Response Root path.')
    parser.add_argument('--image_resolution', type=int, help='Response Root path.')
    args = parser.parse_args()
    
    
    infer_xgenmm(qasper_data, args)
