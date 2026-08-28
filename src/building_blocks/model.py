"""
Copyright 2023 Wenyue Hua

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

__author__ = "Wenyue Hua"
__copyright__ = "Copyright 2023, WarAgent"
__date__ = "2023/11/28"
__license__ = "Apache 2.0"
__version__ = "0.0.1"

import requests
import openai
from openai import OpenAI
from utils import *

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Candidate ':free' model ids, sorted by context length descending. Populated
# once per run by _load_free_model_candidates(). Some of these may turn out
# to be unusable via the plain chat completions API (e.g. restricted to
# "agentic harnesses"), so get_daily_free_model() can advance past them.
_free_model_candidates = None
_free_model_index = 0

def _load_free_model_candidates():
    global _free_model_candidates
    if _free_model_candidates is not None:
        return

    response = requests.get("{}/models".format(OPENROUTER_BASE_URL))
    response.raise_for_status()
    models = response.json()["data"]
    free_models = [m for m in models if m["id"].endswith(":free")]
    if not free_models:
        raise RuntimeError("No free (':free') models are currently available on OpenRouter.")

    _free_model_candidates = sorted(free_models, key=lambda m: m["context_length"], reverse=True)

def get_daily_free_model():
    """
    Resolve the free OpenRouter model to use for this run: the ':free'
    model with the largest context length among those currently offered.
    Cached after the first call so a single run always talks to the same
    model, even across retries. If that model turns out to be unusable
    (see advance_to_next_free_model), the next-largest candidate is used.
    """
    _load_free_model_candidates()
    chosen = _free_model_candidates[_free_model_index]
    print("Using OpenRouter free model of the day: {} (context length: {})".format(chosen["id"], chosen["context_length"]))
    return chosen["id"]

def advance_to_next_free_model():
    """
    Move past the current free model candidate (e.g. because OpenRouter
    rejected it as unusable) and fall back to the next-largest one.
    """
    global _free_model_index
    _free_model_index += 1
    if _free_model_index >= len(_free_model_candidates):
        raise RuntimeError("No usable free (':free') models left on OpenRouter after exhausting all candidates.")

def generate_action(prompt, round):
    MAX_RETRIES = 10

    candidate_start_tokens = [
        "Final Action List in JSON:",\
        "Final Action List in JSON format:",\
        "Final Actions in JSON format:",\
        "Final Action List in JSON format\n",\
        "Final Action List:",\
        "Final Action List\n",\
        "Final Actions:",\
        "Summarize Analysis\n",\
        "Actions to Perform:\n",\
    ]

    plan = run_openrouter(prompt)

    n = 0
    while n < MAX_RETRIES:
        if n > 0:
            print('Generated result cannot be parsed. Retrying generation of plan for {} times...'.format(n))
        try:
            assert "{" in plan
            if plan.count('{') == 1 and plan.count('}') == 1:
                start_token = "{"
                end_token = "}"
                assert start_token in plan and end_token in plan
                start_token_index = plan.index(start_token)
                end_token_index = plan.index(end_token)+1
            else:
                has_start_token = False
                for start_token in candidate_start_tokens:
                    if start_token in plan:
                        assert plan.count(start_token) == 1
                        has_start_token = True
                        start_token_index = plan.index(start_token) + len(start_token)
                        end_token_index = [i for i, c in enumerate(plan) if c == '}'][-1]+1
                        break
                if not has_start_token:
                    raise ValueError("Cannot find start token")
            
            final_json_string = plan[start_token_index:end_token_index]
            thought_process = plan.replace(final_json_string, '')
            final_json_string = plan[start_token_index:end_token_index].strip().rstrip("\n").strip()
            # very often occurring bug: "null" but without quotes
            final_json_string = final_json_string.replace(' null ', ' "null" ')
            final_json_string = re_format_to_json(final_json_string)
            dictionary = parse_dict_string(final_json_string)

            # dictionary may include empty list, so remove such keys; also change back "null" to None
            for k,v in dictionary.items():
                if v == []:
                    dictionary.pop(k)
                if v == "null":
                    dictionary[k] = None

            # "Wait without Action" should only occur alone
            if round == 0:
                if len(dictionary)>=2:
                    if 'Wait without Action' in dictionary:
                        dictionary.pop('Wait without Action')
            else:
                assert 'responding_actions' in dictionary and 'new_actions' in dictionary
                action_length = 0
                for k,v in dictionary.items():
                    for k2,v2 in v.items():
                        if k2 == 'null':
                            dictionary[k].pop(k2)
                for k,v in dictionary.items():
                    action_length += len(v)
                if action_length >= 2:
                    for k,v in dictionary.items():
                        if 'Wait without Action' in v:
                            dictionary[k].pop('Wait without Action') 
            break
        except:
            n += 1
            if n >= MAX_RETRIES:
                # raise Exception("Maximum retries reached")
                print("Maximum retries reached, no action generated.")
                if round == 0:
                    dictionary = {'Wait without Action':None}
                    thought_process = 'There is nothing special I need to do'
                else:
                    dictionary = {'responding_actions': {}, 'new_actions': {'Wait without Action':None}}
                    thought_process = 'There is nothing special I need to do'
                break
            plan = run_openrouter(prompt)

    return thought_process, dictionary


def run_openrouter(text_prompt, temperature: float = 0):
    open_router_key = os.environ["OPENROUTER_API_KEY"]
    client = OpenAI(api_key=open_router_key, base_url=OPENROUTER_BASE_URL)
    while True:
        try:
            response = client.chat.completions.create(
              model=get_daily_free_model(),
              messages=[
                {"role": "user", "content": text_prompt},
              ],
              temperature=temperature,
            )
            break
        except openai.PermissionDeniedError as e:
            print("Free model rejected by OpenRouter ({}), falling back to the next candidate...".format(e))
            advance_to_next_free_model()
    resp = response.choices[0].message.content
    resp = resp.replace("""```json""", '').replace("""```""", '')
    return resp


# llm_lingua = PromptCompressor()
# def compress_prompt(prompt):
#     target_token=2000
#     compressed_prompt = llm_lingua.compress_prompt(
#         prompt.split("\n"),
#         instruction="",
#         question="",
#         target_token=target_token,
#         condition_compare=True,
#         condition_in_question='after',
#         rank_method='llmlingua',
#         use_sentence_level_filter=False,
#         context_budget="+100",
#         dynamic_context_compression_ratio=0.4, # enable dynamic_context_compression_ratio
#         reorder_context="sort"
#     )['compressed_prompt']
#     return compressed_prompt
