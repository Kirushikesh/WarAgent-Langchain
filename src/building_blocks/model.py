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

from langchain_google_genai import ChatGoogleGenerativeAI
from utils import *

# def generate_action(prompt, model, round):
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

    # plan = run_api(model, prompt)
    plan = run_api(prompt)
    
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
            # plan = run_api(model, prompt)
            plan = run_api(prompt)

    return thought_process, dictionary   


def run_api(prompt, max_tokens_to_sample: int = 100000, temperature: float = 0):
    print("Use Gemini to generate action plan...")
    plan = run_model(prompt,temperature=temperature)
    return plan

def run_model(text_prompt, max_tokens_to_sample = None, temperature: float = 0):
    print(text_prompt)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=temperature,
        max_tokens=max_tokens_to_sample,
        timeout=None,
        max_retries=2,
        # other params...
    )
    messages = [
        ("human",text_prompt)
    ]

    resp = llm.invoke(messages).content
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
