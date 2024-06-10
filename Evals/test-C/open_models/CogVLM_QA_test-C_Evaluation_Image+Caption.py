from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch
import json
import random
from PIL import Image
import os
import argparse


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


_PROMPT_1 = "Caption: <caption>. Is the input image and caption helpful to answer the following question. Answer in one word - Yes or No. Question: <question>. "
_PROMPT_2 = "Caption: <caption>. Please provide a brief answer to the following question after looking into the input image and caption. Question: <question>."

def infer_llava(qasper_data, args):

    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to('cuda').eval()

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
                    
                    cogvlm_prompt_1 = _PROMPT_1.replace('<caption>', caption).replace('<question>', question)
                    cogvlm_prompt_2 = _PROMPT_2.replace('<caption>', caption).replace('<question>', question)
                
                    image = Image.open(os.path.join(_QASPER_IMAGE_ROOT, paper['arxiv_id'], figure)).convert('RGB')
                    image = image.resize((args.image_resolution, args.image_resolution))

                    inputs_1 = model.build_conversation_input_ids(tokenizer, query=cogvlm_prompt_1, history=[], images=[image])  # chat mode
                    inputs_1 = {
                        'input_ids': inputs_1['input_ids'].unsqueeze(0).to('cuda'),
                        'token_type_ids': inputs_1['token_type_ids'].unsqueeze(0).to('cuda'),
                        'attention_mask': inputs_1['attention_mask'].unsqueeze(0).to('cuda'),
                        'images': [[inputs_1['images'][0].to('cuda').to(torch.bfloat16)]],
                    }
                    gen_kwargs = {"max_length": 2048, "do_sample": False}

                    with torch.no_grad():
                        outputs_1 = model.generate(**inputs_1, **gen_kwargs)
                        outputs_1 = outputs_1[:, inputs_1['input_ids'].shape[1]:]
                        generated_text_1 = tokenizer.decode(outputs_1[0])


                    inputs_2 = model.build_conversation_input_ids(tokenizer, query=cogvlm_prompt_2, history=[], images=[image])  # chat mode
                    inputs_2 = {
                        'input_ids': inputs_2['input_ids'].unsqueeze(0).to('cuda'),
                        'token_type_ids': inputs_2['token_type_ids'].unsqueeze(0).to('cuda'),
                        'attention_mask': inputs_2['attention_mask'].unsqueeze(0).to('cuda'),
                        'images': [[inputs_2['images'][0].to('cuda').to(torch.bfloat16)]],
                    }
                    gen_kwargs = {"max_length": 2048, "do_sample": False}

                    with torch.no_grad():
                        outputs_2 = model.generate(**inputs_2, **gen_kwargs)
                        outputs_2 = outputs_2[:, inputs_2['input_ids'].shape[1]:]
                        generated_text_2 = tokenizer.decode(outputs_2[0])


                    answer_dict.update({figure: [generated_text_1, generated_text_2]})
            
                    print(answer_dict[figure])
                    print('-----------------')

                question_key = paper['question_key'][question_idx]
                response_paper.update({question_key: {'question': question, 'response': answer_dict,
                                              'referred_figures_names': referred_figures, 'answer': answer}})   

        except:
            print('Error in generating.')
            tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
            model = AutoModelForCausalLM.from_pretrained(
                'THUDM/cogvlm-chat-hf',
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to('cuda').eval()
            continue

        with open(os.path.join(_RESPONSE_ROOT, str(paper_id) + '_response.json'), 'w') as f:
            json.dump(response_paper, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate on Qasa/Qasper.')
    parser.add_argument('--response_root', type=str, help='Response Root path.')
    parser.add_argument('--image_resolution', type=int, help='Response Root path.')
    args = parser.parse_args()
    
    
    infer_llava(qasper_data, args)