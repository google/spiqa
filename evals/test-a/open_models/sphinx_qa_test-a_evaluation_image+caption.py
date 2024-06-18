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


from SPHINX import SPHINXModel
from PIL import Image
import torch
import json
import random
import os
import argparse
from transformers.utils.quantization_config import BitsAndBytesConfig
from quant import quantize


testA_filtered_annotations_path = '../../../datasets/test-A/SPIQA_testA.json'
with open(testA_filtered_annotations_path, "r") as f:
  testA_data = json.load(f)

def prepare_inputs(paper, question_idx):
    all_figures = list(paper['all_figures'].keys())
    referred_figures = [paper['qa'][question_idx]['reference']]
    answer = paper['qa'][question_idx]['answer']

    referred_figures_captions = []
    for figure in referred_figures:
        referred_figures_captions.append(paper['all_figures'][figure]['caption'])

    return answer, all_figures, referred_figures, referred_figures_captions


_PROMPT_1 = "Caption: <caption> Is the input image and caption helpful to answer the following question. Answer in one word - Yes or No. Question: <question>. "
_PROMPT_2 = "Caption: <caption> Please provide a brief answer to the following question after looking into the input image and caption. Question: <question>."


def infer_sphinx(testA_data, args):

    if args.image_resolution == 224:
        _testA_IMAGE_ROOT = "../../../datasets/test-A/SPIQA_testA_Images_224px"
    else:
        raise NotImplementedError

    model = SPHINXModel.from_pretrained(pretrained_path=args.model_path, with_visual=True, device="cpu")
    quantization_config = BitsAndBytesConfig.from_dict(
            config_dict={
                "load_in_8bit": False,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            },
            return_unused_kwargs=False,
        )
    quantize(model, quantization_config)
    model.cuda()
    model.eval()
    
    _RESPONSE_ROOT = args.response_root
    os.makedirs(_RESPONSE_ROOT, exist_ok=True)

    for paper_id, paper in sorted(testA_data.items(), key=lambda x: random.random()):
        if os.path.exists(os.path.join(_RESPONSE_ROOT, str(paper_id) + '_response.json')):
            continue
        response_paper = {}

        try:
            for question_idx, qa in enumerate(paper['qa']):

                question = qa['question']

                answer, all_figures, referred_figures, referred_figures_captions = prepare_inputs(paper, question_idx)

                answer_dict = {}

                for _idx, figure in enumerate(referred_figures):

                    caption = referred_figures_captions[_idx]
                    
                    sphinx_prompt_1 = _PROMPT_1.replace('<caption>', caption).replace('<question>', question)
                    sphinx_prompt_2 = _PROMPT_2.replace('<caption>', caption).replace('<question>', question)

                    qas_1 = [[sphinx_prompt_1, None]]
                
                    image = Image.open(os.path.join(_testA_IMAGE_ROOT, paper['paper_id'], figure))

                    generated_text_1 = model.generate_response(qas_1, image, max_gen_len=256, temperature=0.9, top_p=0.5, seed=0)

                    qas_2 = [[sphinx_prompt_2, None]]
                    generated_text_2 = model.generate_response(qas_2, image, max_gen_len=256, temperature=0.9, top_p=0.5, seed=0)
                    
                    answer_dict.update({figure: [generated_text_1, generated_text_2]})
            
                    print(answer_dict[figure])
                    print('-----------------')

                response_paper.update({question_idx: {'question': question, 'response': answer_dict,
                                                'referred_figures_names': referred_figures, 'answer': answer}})   

        except:
            print('Error in generating.')
            model = SPHINXModel.from_pretrained(pretrained_path=args.model_path, with_visual=True, device="cpu")
            quantization_config = BitsAndBytesConfig.from_dict(
                    config_dict={
                        "load_in_8bit": False,
                        "load_in_4bit": True,
                        "bnb_4bit_quant_type": "nf4",
                    },
                    return_unused_kwargs=False,
                )
            quantize(model, quantization_config)
            model.cuda()
            model.eval()
            continue

        with open(os.path.join(_RESPONSE_ROOT, str(paper_id) + '_response.json'), 'w') as f:
            json.dump(response_paper, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate on Qasa/Qasper.')
    parser.add_argument('--response_root', type=str, help='Response Root path.')
    parser.add_argument('--image_resolution', type=int, help='Response Root path.')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint.')
    args = parser.parse_args()
    
    
    infer_sphinx(testA_data, args)