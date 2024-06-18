"""LLM Log-Likelihood Scoring for OpenAI GPT models.

Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# LLM Log-Likelihood Scoring for OpenAI GPT models.
#Uses the top-5 log probabilities of the model to score a prediction
#as similar or not to a given ground truth answer.S

import datetime
import json
import logging
import os
import pathlib
from typing import Any, Optional, Tuple, List, Dict

import numpy as np
import openai
import scipy


NEGATIVE_INF = -1000.0
logger = logging.getLogger()


def renormalize_score(yes_score: float, no_score: float) -> float:
  """Corrects the score by applying log-softmax normalization."""
  return np.exp(yes_score - scipy.special.logsumexp([no_score, yes_score]))


def _normalize(text: str) -> str:
  """Remove white space and lower case for normalized comparisons."""
  return text.strip().lower()


def score_openai(
    response: Dict[str, Any],  # openai.ChatCompletion
    suffixes: List[str],
    complement_suffixes: List[str],
) -> float:
  """Returns renormalized prob(suffix) based on top-5 logprobs."""
  assert len(response['choices']) == 1, 'More than 1 choice.'
  response = response['choices'][0]

  # Sanity checks.
  if 'logprobs' not in response:
    raise ValueError('No logprobs found.')
  if 'content' not in response['logprobs']:
    raise ValueError('No content found.')
  if not response['logprobs']['content']:
    raise ValueError('Content is empty.')
  if 'top_logprobs' not in response['logprobs']['content'][0]:
    raise ValueError('No top_logprobs found.')

  top_answers_logprobs = response['logprobs']['content'][0]['top_logprobs']

  # Identify the suffix and complement_suffix if each of them exist.
  # Additionally, extract the corresponding logprob.
  # -- First, search for the suffix.
  suffix_logprob = NEGATIVE_INF
  complement_logprob = NEGATIVE_INF
  suffix_index = -1
  complement_suffix_index = -1

  normalized_suffixes = [_normalize(suffix) for suffix in suffixes]
  normalized_complement_suffixes = [
      _normalize(complement_suffix) for complement_suffix in complement_suffixes
  ]

  # Iterate over the top-n logprobs to find the suffix and complement_suffix.
  # The logprobs are already sorted in descending order, so we break once we
  # find the match with the highest logprob.
  for i, token_logprob in enumerate(top_answers_logprobs):
    if _normalize(token_logprob['token']) in normalized_suffixes:
      suffix_logprob = token_logprob['logprob']
      suffix_index = i
      break

  for i, token_logprob in enumerate(top_answers_logprobs):
    if _normalize(token_logprob['token']) in normalized_complement_suffixes:
      complement_suffix_index = i
      complement_logprob = token_logprob['logprob']
      break

  logger.info(
      'Found: Suffix index: %d, complement_suffix_index: %d',
      suffix_index,
      complement_suffix_index,
  )

  # None of the suffixes or complement_suffixes were found in the output.
  # So score is 0.0.
  if suffix_index == -1 and complement_suffix_index == -1:
    return 0.0

  # Both suffix and complement_suffix were found in the output!
  # This indicates model is ambiguous and there's high prob of both.
  if suffix_index != -1 and complement_suffix_index != -1:
    return renormalize_score(
        yes_score=suffix_logprob, no_score=complement_logprob
    )

  # If only one of the suffix or complement_suffix was found in the output,
  # then we want to find the logprob of the reciprocal (i.e. the item that was
  # not found). To find the reciprocal we get the
  # max of (lowest top-5 logprobs, remaining log prob after summing the top-5).
  # This is equivalent to identifying the
  # min of (lowest prob token in top-5, remaining prob after summing top-5)
  lowest_logprob = top_answers_logprobs[-1]['logprob']
  lowest_token_prob = np.exp(lowest_logprob)
  sum_probs = sum([
      np.exp(token_logprob['logprob']) for token_logprob in top_answers_logprobs
  ])
  remaining_prob = 1 - sum_probs
  min_prob = min(lowest_token_prob, remaining_prob)
  if min_prob < 1e-8:
    min_prob = 1e-8
  reciprocal_logprob = np.log(min_prob)

  if suffix_index != -1:
    exclude_score = suffix_logprob
    include_score = reciprocal_logprob
  elif complement_suffix_index != -1:
    exclude_score = reciprocal_logprob
    include_score = complement_logprob
  else:
    raise ValueError('Not the case where suffix or complement suffix is found.')

  return renormalize_score(yes_score=exclude_score, no_score=include_score)


class OpenAIClient:
  """A proxy to query a OpenAI's API."""

  def __init__(
      self,
      model_name: str,
      api_key: str,
      json_output_path: Optional[str],
  ):
    """Initializes a OpenAIClient.

    Args:
      model_name: The name of the OpenAI model to use (e.g. 'gpt-4').
      api_key: OpenAI API key string.
      json_output_path: If not None, the path to the directory to write JSON
        output to.
    """

    openai.api_key = api_key
    self._model_name = model_name
    if json_output_path:
      self._json_output_path = pathlib.Path(json_output_path)
      if not self._json_output_path.exists():
        self._json_output_path.mkdir(parents=True, exist_ok=True)
    self._timeout = 60

  def call_openai(
      self,
      prompt: str,
      output_prefix: Optional[str],
      max_decode_steps: int = 1024,
      temperature: float = 0.0,
  ) -> str:
    """Call OpenAI chat completion API; save and return the response."""
    message = [{'role': 'user', 'content': prompt}]
    response = openai.chat.completions.create(
        model=self._model_name,
        messages=message,
        temperature=temperature,
        max_tokens=max_decode_steps,
        top_p=1,
        timeout=self._timeout * 10,
    )
    response_json = response.model_dump_json()
    response = json.loads(response_json)
    assert len(response['choices']) == 1
    if not output_prefix:
      output_prefix = ''
    if self._json_output_path:
      timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
      filename = os.path.join(
          self._json_output_path, f'gpt4_{output_prefix}_{timestamp}.json'
      )
      with open(filename, 'w') as f:
        response['input_prompt'] = message
        json.dump(response, f)
    text_response = response['choices'][0]['message']['content']
    return text_response

  def call_openai_with_score(
      self,
      prompt: str,
      suffixes: List[str],
      output_prefix: Optional[str],
      max_decode_steps: int = 1024,
      temperature: float = 0.0,
      complement_suffixes: Optional[List[str]] = None,
  ) -> Tuple[str, float]:
    """Call OpenAI."""
    message = [{'role': 'user', 'content': prompt}]
    if not output_prefix:
      output_prefix = ''
    assert suffixes, 'Please supply a suffix token to score the output.'
    # response = openai.ChatCompletion.create(
    response = openai.chat.completions.create(
        model=self._model_name,
        messages=message,
        temperature=temperature,
        max_tokens=max_decode_steps,
        top_p=1,
        logprobs=True,
        top_logprobs=5,  # This is the largest value allowed by OpenAI.
        frequency_penalty=0,
        presence_penalty=0,
        timeout=self._timeout * 10,
    )
    response_json = response.model_dump_json()
    response = json.loads(response_json)
    assert len(response['choices']) == 1
    if self._json_output_path:
      timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
      filename = os.path.join(
          self._json_output_path, f'gpt4_{output_prefix}_{timestamp}.json'
      )
      with open(filename, 'w') as f:
        response['input_prompt'] = message
        json.dump(response, f)
    text_response = response['choices'][0]['message']['content']
    if complement_suffixes is None:
      complement_suffixes = []
    score = score_openai(
        response, suffixes=suffixes, complement_suffixes=complement_suffixes
    )
    return text_response, score