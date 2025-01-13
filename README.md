# SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers

[**SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers**](https://arxiv.org/abs/2407.09413)    
[Shraman Pramanick](https://shramanpramanick.github.io/), [Rama Chellappa](https://engineering.jhu.edu/faculty/rama-chellappa/), [Subhashini Venugopalan](https://vsubhashini.github.io/)     
NeurIPS D&B 2024               
[Paper](https://arxiv.org/abs/2407.09413) | [SPIQA Dataset](https://huggingface.co/datasets/google/spiqa)

> **TL;DR:** we introduce SPIQA (**S**cientific **P**aper **I**mage **Q**uestion **A**nswering), the first large-scale QA dataset specifically designed to interpret complex figures and tables within the context of scientific research articles across various domains of computer science.

<img src="SPIQA_Tasks.png" alt="SPIQA_Tasks" style="zoom:67%;" />

## 📢 News

- [Jan, 2025] We update instructions to run metric evaluation scripts for all three tasks.
- [Sept, 2024] SPIQA has been accepted for publication at [NeurIPS 2024](https://neurips.cc/Conferences/2024) in the Datasets and Benchmarks track.
- [July, 2024] We update instructions to run evaluation with different baselines on all three tasks, and release the [responses by baselines](https://drive.google.com/drive/folders/1Y_27zme95jz9cH1UA8cphlRKhi3afwtA?usp=sharing) to fully reproduce the reported numbers.
- [July, 2024] [SPIQA Paper](https://arxiv.org/abs/2407.09413) is now up on arXiv.
- [June, 2024] [SPIQA](https://huggingface.co/datasets/google/spiqa) is now live on Hugging Face🤗.

## 📝 TODOs

- [x] Instructions to run metric computation scripts.
- [x] Starter code snippet for L3Score.
- [x] Release responses by baselines to fully reproduce the reported numbers.
- [x] Instructions to run evaluation. 

## 📁 Repository Structure

The contents of this repository are structured as follows:

```bash
spiqa
    ├── evals
        ├── Evaluation of all open- and closed-source models on test-A
        ├── Evaluation of all open- and closed-source models on test-B 
        └── Evaluation of all open- and closed-source models on test-C
    ├── metrics
        └── Computation of BLEU, ROUGE, CIDEr, METEOR, BERTScore and L3Score
    
```
Each directory contains different python scripts to evaluate various models on three different tasks and compute metrics.

## 🗄️ Dataset

[SPIQA](https://huggingface.co/datasets/google/spiqa) is publicly available on Hugging Face🤗.

#### Dataset Use and Starter Snippets

##### Downloading the Dataset to Local

We recommend the users to download the metadata and images to their local machine. 

- Download the whole dataset (all splits).
```bash
from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/spiqa", repo_type="dataset", local_dir='.') ### Mention the local directory path
```

- Download specific file.
```bash
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="google/spiqa", filename="test-A/SPIQA_testA.json", repo_type="dataset", local_dir='.') ### Mention the local directory path
```

##### Questions and Answers from a Specific Paper in test-A
```bash
import json
testA_metadata = json.load(open('test-A/SPIQA_testA.json', 'r'))
paper_id = '1702.03584v3'
print(testA_metadata[paper_id]['qa'])
```

##### Questions and Answers from a Specific Paper in test-B
```bash
import json
testB_metadata = json.load(open('test-B/SPIQA_testB.json', 'r'))
paper_id = '1707.07012'
print(testB_metadata[paper_id]['question']) ## Questions
print(testB_metadata[paper_id]['composition']) ## Answers
```

##### Questions and Answers from a Specific Paper in test-C
```bash
import json
testC_metadata = json.load(open('test-C/SPIQA_testC.json', 'r'))
paper_id = '1808.08780'
print(testC_metadata[paper_id]['question']) ## Questions
print(testC_metadata[paper_id]['answer']) ## Answers
```

## 🧪 Evaluation

##### Setting up Conda Environment

We use [conda-pack](https://conda.github.io/conda-pack/) to share the required environment for every baseline model for its greater portability. First, start with downloading the [environment tars](http://www.cis.jhu.edu/~shraman/SPIQA/conda_envs_spiqa.tar.gz).
```bash
wget http://www.cis.jhu.edu/~shraman/SPIQA/conda_envs_spiqa.tar.gz
tar -xvzf conda_envs_spiqa.tar.gz && rm conda_envs_spiqa.tar.gz
```
Activate individual envs as follows. In the following snippet, we show an example for running the Gemini 1.5 Pro model. 
```bash
mkdir -p gemini_env
tar -xzf envs/gemini.tar.gz -C gemini_env
source gemini_env/bin/activate
```

- ##### Evaluate Gemini 1.5 Pro for Direct QA with Figures and Tables on test-A

For running the closed-weight models, first provide the API key from corresponding accounts. For example, to run Gemini, fill in the api_key in the scripts `genai.configure(api_key=<Your_API_Key>)`.
```bash
cd evals/test-a/closed_models/
python gemini_qa_test-a_evaluation_image+caption.py --response_root <path_to_save_responses> --image_resolution -1 --model_id gemini-1.5-pro
```

- ##### Evaluate Gemini 1.5 Pro for Direct QA with Full Paper on test-A
```bash
cd evals/test-a/closed_models/
python gemini_qa_test-a_evaluation_image+caption+full_text.py --response_root <path_to_save_responses> --image_resolution -1 --model_id gemini-1.5-pro
```

- ##### Evaluate Gemini 1.5 Pro for CoT QA on test-A
```bash
cd evals/test-a/closed_models/
python gemini_cot_qa_test-a_evaluation_image+caption.py --response_root <path_to_save_responses> --image_resolution -1 --model_id gemini-1.5-pro
```

We list the URLs/Model IDs of all baselines in the [MODEL Zoo](./Model_MOO.md). The names of the various scripts clearly indicate the respective tasks, baseline settings, and evaluation splits.

**NOTE:** To run the SPHINX-v2 baseline model, clone the [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory) github repository, create an environment following the [installation guidelines](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX#installation), and download the [SPHINX-v2-1k](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX/SPHINX-v2-1k) checkpoint.

## ✅ Reproducible Results

To reproduce the results reported in our [paper](https://arxiv.org/abs/2407.09413), we provide the outputs of all open- and closed-source models [here](https://drive.google.com/drive/folders/1Y_27zme95jz9cH1UA8cphlRKhi3afwtA?usp=sharing). Please find the instructions for the metric computation below.

## 💡 Starter Code Snippet for L3Score

```bash
from metrics.llmlogscore.llmlogscore import OpenAIClient

client = OpenAIClient(
    model_name='gpt-4o',
    api_key=<openai_api_key>,
    json_output_path='./saved_output_l3score/',
)

_PROMPT = 'You are given a question, ground-truth answer, and a candidate answer. Question: <question> \nGround-truth answer: <GT> \nCandidate answer: <answer> \n\
Is the semantic meaning of the ground-truth and candidate answers similar? Answer in one word - Yes or No.'
_SUFFIXES_TO_SCORE = [' yes', ' yeah']
_COMPLEMENT_SUFFIXES = [' no']

question = 'Where is Niagara falls located?'
gt = 'Niagara Falls is located on the border between the United States and Canada, specifically between New York State and Ontario Province.'
candidate_answer = 'Niagara Falls is situated on the Niagara River, which connects Lake Erie to Lake Ontario, \
and lies on the international border between the United States (New York State) and Canada (Ontario Province).'

prompt_current = _PROMPT.replace('<question>', question).replace('<GT>', gt).replace('<answer>', candidate_answer)
response, prob_yes = client.call_openai_with_score(
            prompt=prompt_current,
            suffixes=_SUFFIXES_TO_SCORE,
            complement_suffixes=_COMPLEMENT_SUFFIXES,
            output_prefix=''
            )

print('L3Score: ', prob_yes)
#### >>> L3Score: 0.9999999899999982

wrong_answer = 'Niagara Falls is located on the border between the United States and Mexico, specifically between New York State and Ontario Province.'

prompt_current = _PROMPT.replace('<question>', question).replace('<GT>', gt).replace('<answer>', wrong_answer)
response, prob_yes = client.call_openai_with_score(
            prompt=prompt_current,
            suffixes=_SUFFIXES_TO_SCORE,
            complement_suffixes=_COMPLEMENT_SUFFIXES,
            output_prefix=''
            )

print('L3Score: ', prob_yes)
#### >>> L3Score: 3.653482080241728e-08
```

## 📊 Metric Computation

To compute the reported metrics, start with downloading our metric computation script from [here](https://drive.google.com/file/d/1CeYvOoB1jMEhEgJ_W2dm1OThMXptKCU6/view?usp=sharing) which is slightly modified from its [original source](https://github.com/tylin/coco-caption/tree/master/pycocoevalcap).

```bash
gdown --fuzzy https://drive.google.com/file/d/1CeYvOoB1jMEhEgJ_W2dm1OThMXptKCU6/view?usp=sharing
tar -xzvf pycocoevalcap_spiqa.tar.gz && rm -rf pycocoevalcap_spiqa.tar.gz
```

We provide different script to compute the metrics for open- and closed-courced models. We show an example on how to run each diffrent script here. 

##### Direct QA with Figures and Tables for Open-source Models

We provide two different scripts, one for computing exiting metrics (METEOR, ROUGE-L, CIDEr, BERTScore, BLEU), and the other for computing L3Score. Since we use GPT-4o to compute L3Score, this script requires an openai API key.

```bash
## Here we show example to compute the performance of LLaVA-1.5-7B on test-A
cd metrics
python open_models_metrics.py --response_root ../model_responses/test-a_responses/test-a_llava_qa/llava_image+caption_size_224/
python open_models_l3score.py --response_root ../model_responses/test-a_responses/test-a_llava_qa/llava_image+caption_size_224/ --openai_api_key <openai_api_key>
```

##### Direct QA with Figures and Tables for Close-source Models

Similarly for close-source models, we provide two different scripts.

```bash
## Here we show example to compute the performance of Gemini 1.5 Flash on test-A
cd metrics
python close_models_metrics.py --response_root ../model_responses/test-a_responses/test-a_gemini_qa/gemini_1-5_flash_image+caption_size_full/
python close_models_l3score.py --response_root ../model_responses/test-a_responses/test-a_gemini_qa/gemini_1-5_flash_image+caption_size_full/ --openai_api_key <openai_api_key>
```

##### Direct QA with Full paper

We only report the performance of close-source models for direct QA with full paper, and the same scripts can be used for computing metrics on this task.

```bash
## Here we show example to compute the performance of Gemini 1.5 Flash on test-B
cd metrics
python close_models_metrics.py --response_root ../model_responses/test-b_responses/test-b_gemini_qa/gemini_1-5_flash_image+caption+text_size_full/
python close_models_l3score.py --response_root ../model_responses/test-b_responses/test-b_gemini_qa/gemini_1-5_flash_image+caption+text_size_full/ --openai_api_key <openai_api_key>
```

##### CoT QA with Figures and Tables

We provide additional scripts for evaluation on CoT QA task. We only report the performance of close-source models for CoT QA.

```bash
## Here we show example to compute the performance of Gemini 1.5 Flash on test-B
cd metrics
python close_models_metrics_CoT.py --response_root ../model_responses/test-b_responses/test-b_gemini_qa/gemini_1-5_flash_cot_qa_image+caption_size_full/
python close_models_l3score_CoT.py --response_root ../model_responses/test-b_responses/test-b_gemini_qa/gemini_1-5_flash_cot_qa_image+caption_size_full/ --openai_api_key <openai_api_key>
```

Note: Due to changes in OpenAI's GPT-4o checkpoint, the L3Score values reported in the paper might differ from those you get when running our metric computation scripts. However, our underlying algorithm for L3Score stays the same, and can be used with any language model.

## ✉️ Contact

This repository is created and maintained by [Shraman](https://shramanpramanick.github.io/) and [Subhashini](https://vsubhashini.github.io/). Questions and discussions are welcome via spraman3@jhu.edu and vsubhashini@google.com.

## 🙏 Acknowledgements

We evaluate six different open source models on SPIQA: [LLaVA 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf), [InstructBLIP](https://huggingface.co/Salesforce/instructblip-vicuna-7b), [XGen-MM](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1), [InternLM-XC](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b), [SPHINX-v2](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX) and [CogVLM](https://huggingface.co/THUDM/cogvlm-chat-hf). We thank the respective authors for releasing the model weights. We are grateful to the colleagues in the Science Assistant team at Google Research for valuable discussions and support to our project.

## 📄 License

SPIQA evaluation code and library for L3Score in this Github repository are licensed under a [APACHE 2.0 License](./LICENSE).

## 🎓 Citing SPIQA

```
@article{pramanick2024spiqa,
  title={SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers},
  author={Pramanick, Shraman and Chellappa, Rama and Venugopalan, Subhashini},
  journal={NeurIPS},
  year={2024}
}
```
