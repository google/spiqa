# SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers

[**SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers**]()    
[Shraman Pramanick](https://shramanpramanick.github.io/), [Rama Chellappa](https://engineering.jhu.edu/faculty/rama-chellappa/), [Subhashini venugopalan](https://vsubhashini.github.io/)                
arXiv, 2024               
[Paper](https://arxiv.org/abs/2407.09413) | [SPIQA Dataset](https://huggingface.co/datasets/google/spiqa) | Project Page (Coming Soon)

> **TL;DR:** we introduce SPIQA (**S**cientific **P**aper **I**mage **Q**uestion **A**nswering), the first large-scale QA dataset specifically designed to interpret complex figures and tables within the context of scientific research articles across various domains of computer science.

<img src="SPIQA_Tasks.png" alt="SPIQA_Tasks" style="zoom:67%;" />

## 📢 News

- [July, 2024] [SPIQA Paper](https://arxiv.org/abs/2407.09413) is now up on arXiv.
- [June, 2024] [SPIQA](https://huggingface.co/datasets/google/spiqa) is now live on Hugging Face🤗.

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
Each directory contains different python scripts to evaluate various models and compute different metrics.

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

## ✉️ Contact
This repository is created and maintained by [Shraman](https://shramanpramanick.github.io/) and [Subhashini](https://vsubhashini.github.io/). Questions and discussions are welcome via spraman3@jhu.edu and vsubhashini@google.com.

## 🙏 Acknowledgements


## 📄 License

SPIQA is licensed under a [APACHE 2.0 License](./LICENSE).

## 🎓 Citing SPIQA

```
@article{pramanick2024spiqa,
  title={SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers},
  author={Pramanick, Shraman and Chellappa, Rama and Venugopalan, Subhashini},
  journal={arXiv preprint arXiv:2407.09413},
  year={2024}
}
```

