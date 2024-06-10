# SPIQA: A Dataset for Multimodal Question Answering in Scientific Papers


The SPIQA dataset will be released on Huggingface and can be downloaded from:

The data set is licensed under [CC-BY
4.0](http://creativecommons.org/licenses/by/4.0/)

Structure of testA metadata:

```
{
  "1611.04684v1": {
    "paper_id": "1611.04684v1",
    "all_figures": {
      “Figure Name": {
        "caption": Caption,
        "content_type": figure/table,
        "figure_type": type of figure
      },
    },
    "qa": [
      {
        "question": Question,
        "answer": Answer,
        "explanation": Rationale,
        "reference": Reference Image
      }
    ]
  }

```

