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


from transformers import AutoModel, AutoTokenizer
import torch
import json
import random
from PIL import Image
import os
import argparse


def auto_configure_device_map(num_gpus):
    # visual_encoder 算4层
    # internlm_model.model.embed_tokens 占用1层
    # norm 和 lm_head 占用1层
    # transformer.layers 占用 32 层
    # 总共34层分配到num_gpus张卡上
    num_trans_layers = 32
    per_gpu_layers = 38 / num_gpus

    device_map = {
        'vit': 0,
        'vision_proj': 0,
        'model.tok_embeddings': 0,
        'model.norm': num_gpus - 1,
        'output': num_gpus - 1,
    }

    used = 3
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'model.layers.{i}'] = gpu_target
        used += 1

    return device_map


qasa_filtered_annotations_path = '../../../datasets/test-B/SPIQA_testB.json'
with open(qasa_filtered_annotations_path, "r") as f:
    qasa_data = json.load(f)

def prepare_inputs(paper, question_idx):
    all_figures = list(paper['all_figures_tables'].keys())
    referred_figures = list(set(paper['referred_figures_tables'][question_idx]))
    answer = paper['composition'][question_idx]

    referred_figures_captions = []
    for figure in referred_figures:
        referred_figures_captions.append(paper['all_figures_tables'][figure])

    return answer, all_figures, referred_figures, referred_figures_captions


_PROMPT_1 = "<ImageHere> Caption: <caption> Is the input image and caption helpful to answer the following question. Answer in one word - Yes or No. Question: <question>. "
_PROMPT_2 = "<ImageHere> Caption: <caption> Please provide a brief answer to the following question after looking into the input image and caption. Question: <question>."


def infer_internlmxc(qasa_data, args):

    if args.image_resolution == 224:
        _QASA_IMAGE_ROOT = "../../../datasets/test-B/SPIQA_testB_Images_224px"
    else:
        raise NotImplementedError

    # init model and tokenizer
    model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True).eval()
    if args.dtype == 'fp16':
        model.half().cuda()
    elif args.dtype == 'fp32':
        model.cuda()

    if args.num_gpus > 1:
        from accelerate import dispatch_model
        device_map = auto_configure_device_map(args.num_gpus)
        model = dispatch_model(model, device_map=device_map)

    tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True)

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
                    
                    internlmxc_prompt_1 = _PROMPT_1.replace('<caption>', caption).replace('<question>', question)
                    internlmxc_prompt_2 = _PROMPT_2.replace('<caption>', caption).replace('<question>', question)
                
                    image = os.path.join(_QASA_IMAGE_ROOT, figure)

                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            generated_text_1, _ = model.chat(tokenizer, query=internlmxc_prompt_1, image=image, history=[], do_sample=False)


                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            generated_text_2, _ = model.chat(tokenizer, query=internlmxc_prompt_2, image=image, history=[], do_sample=False)

                    
                    answer_dict.update({figure: [generated_text_1, generated_text_2]})
            
                    print(answer_dict[figure])
                    print('-----------------')

                question_key = paper['question_key'][question_idx]
                response_paper.update({question_key: {'question': question, 'response': answer_dict,
                                                'referred_figures_names': referred_figures, 'answer': answer}})   

        except:
            print('Error in generating.')
            # init model and tokenizer
            model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True).eval()
            if args.dtype == 'fp16':
                model.half().cuda()
            elif args.dtype == 'fp32':
                model.cuda()

            if args.num_gpus > 1:
                from accelerate import dispatch_model
                device_map = auto_configure_device_map(args.num_gpus)
                model = dispatch_model(model, device_map=device_map)

            tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True)
            continue

        with open(os.path.join(_RESPONSE_ROOT, str(paper_id) + '_response.json'), 'w') as f:
            json.dump(response_paper, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate on Qasa/Qasper.')
    parser.add_argument('--response_root', type=str, help='Response Root path.')
    parser.add_argument('--image_resolution', type=int, help='Response Root path.')
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--dtype", default='fp16', type=str)
    args = parser.parse_args()
    
    
    infer_internlmxc(qasa_data, args)