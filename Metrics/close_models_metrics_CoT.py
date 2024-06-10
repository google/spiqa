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
import argparse
from pycocotools.coco import COCO
from bert_score import BERTScorer
from pycocoevalcap_spiqa.eval import COCOEvalCap
import os
import re, ast

parser = argparse.ArgumentParser(description='Evaluate on Qasa/Qasper.')
parser.add_argument('--response_root', type=str, help='Response Root path.')
args = parser.parse_args()

def save_result(result, result_dir, filename, remove_duplicate='', is_gt=False):
    final_result_file = os.path.join(result_dir, f'{filename}.json')

    if remove_duplicate:
        result_new = []
        id_list = []
        for res in result:
            if res[remove_duplicate] not in id_list:
                id_list.append(res[remove_duplicate])
                result_new.append(res)
        result = result_new

    if is_gt:
        images = []
        for res in result:
            images.append({"id": res["id"]})
        result = dict(annotations=result, images=images)

    json.dump(result, open(final_result_file, 'w'))
    print(f'result file saved to {final_result_file}')

    return final_result_file


def calculate_all_metrics(_RESPONSE_ROOT, scorer):
  BERTScore_F1 = 0
  all = 0
  failed_parsing = 0
  counter = 0

  pycocoeval_like_pred = []
  pycocoeval_like_gt = []

  for paper_response in os.listdir(_RESPONSE_ROOT):
    with open(os.path.join(_RESPONSE_ROOT, paper_response), 'r') as f:
      saved_results = json.load(f)

    for key, value in saved_results.items():
      if len(value['response'].split('The answer is')) == 2:
        answer = value['response'].split('The answer is')[-1]
      elif len(value['response'].split('The answer to the question is')) == 2:
        answer = value['response'].split('The answer to the question is')[-1]
      else:
        answer = value['response']
      try:
        gt = value['answer']
        if type(answer) != str:
          answer = ''
        pycocoeval_like_pred.append({"image_id": counter, "caption": answer})
        pycocoeval_like_gt.append({"image_id": counter, "id": counter, "caption": gt})
        _, _, F1 = scorer.score([answer], [gt])
        BERTScore_F1 += F1
        all += 1
      except:
        pycocoeval_like_pred.append({"image_id": counter, "caption": ''})
        pycocoeval_like_gt.append({"image_id": counter, "id": counter, "caption": gt})
        failed_parsing += 1
        all += 1
      counter += 1

  pycocoeval_pred_file = save_result(pycocoeval_like_pred, '.', 'pycocoeval_pred') # remove_duplicate='image_id'
  pycocoeval_gt_file = save_result(pycocoeval_like_gt, '.', 'pycocoeval_gt', is_gt=True) # remove_duplicate='image_id'

  coco = COCO(pycocoeval_gt_file)
  coco_result = coco.loadRes(pycocoeval_pred_file)

  # create coco_eval object by taking coco and coco_result
  coco_eval = COCOEvalCap(coco, coco_result)
  # evaluate results
  coco_eval.evaluate(eval_metrics=['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr'])

  print(".......Printing results.......")
  for metric, score in coco_eval.eval.items():
    print(f'{metric}: {score:.3f}')
  print("BERTScore F1: ", BERTScore_F1 / all)
  print("Examples with Failed Parsing: {}".format(failed_parsing))
  print('------------------------------')

scorer = BERTScorer(model_type='bert-base-uncased')
calculate_all_metrics(_RESPONSE_ROOT=args.response_root, scorer=scorer)



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
