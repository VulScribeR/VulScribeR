import openai
import re
import concurrent.futures
import threading
import time
from rag import SearchItem, ScoredCodeSnippet
from typing import List, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" # the device to load the model onto


place_holders = {1: "$MY_UNIQUE_PLACE_HOLDER_1", 2: "$MY_UNIQUE_PLACE_HOLDER_2",
                 3: "$MY_UNIQUE_PLACE_HOLDER_3", 4: "$MY_UNIQUE_PLACE_HOLDER_4", 5: "$MY_UNIQUE_PLACE_HOLDER_5"}

templates = {"base": """
Here are two code snippets specified below, modify code snippet No.2 in a way that it includes the logic of code No.1:
Code Snippet 1: "$MY_UNIQUE_PLACE_HOLDER_1"

Code Snippet 2: "$MY_UNIQUE_PLACE_HOLDER_2"

Put the generated code inside ```C ```
""",
             "extra": """
Here are two code snippets specified below, modify code snippet No.2 in a way that it includes the logic of code No.1:
Code Snippet 1: "$MY_UNIQUE_PLACE_HOLDER_1"

Code Snippet 2: "$MY_UNIQUE_PLACE_HOLDER_2"

Note that the following lines from the first code snippet have a high priority to be in the final modified code:

Lines separated by /~/ : "$MY_UNIQUE_PLACE_HOLDER_3"

Put the generated code inside ```C ```. (Do not include comments)
""", "with_example_vardef": """Here are three code snippets (each containing a function) specified below, look how code snippet AA is changed into BB, introduce these changes to code snippet CC.

Code Snippet AA: "$MY_UNIQUE_PLACE_HOLDER_1"

Code Snippet BB: "$MY_UNIQUE_PLACE_HOLDER_2"

Code Snippet CC: "$MY_UNIQUE_PLACE_HOLDER_3"

Note that the following lines from code snippet CC have a high priority to be in the final modified code:

Lines separated by /~/ : "$MY_UNIQUE_PLACE_HOLDER_4"

Put the generated code inside ```C ```”, and note that the output should be single function with a similar name and parameters as the code snippet CC, and the return statements should be placed after the logic of the code. $MY_UNIQUE_PLACE_HOLDER_5
""",
"vardef": "(Don't Forget to add these variables to function's input arguments: $MY_UNIQUE_PLACE_HOLDER_1)",

"without_example_vardef": """Here are two code snippets specified below, modify code snippet BB in a way that it includes the logic of code AA:

Code Snippet AA: "$MY_UNIQUE_PLACE_HOLDER_1"

Code Snippet BB: "$MY_UNIQUE_PLACE_HOLDER_2"

Note that the following lines from the code snippet AA have a high priority to be in the final modified code:

Lines separated by /~/ : "$MY_UNIQUE_PLACE_HOLDER_3"

Put the generated code inside ```C ``` and note that the output should be single function with a similar function definition as code snippet BB, and the return statements should be placed after the logic of the code. $MY_UNIQUE_PLACE_HOLDER_4""",
"ext":"""Here's two code snippets AA and BB, each including a function. Add some parts of the logic of AA to BB in a way that the result would include all of BB but is not equal to BB. You can add variables to BB to produce a more correct code.

Code Snippet AA: "$MY_UNIQUE_PLACE_HOLDER_1"

Code Snippet BB: "$MY_UNIQUE_PLACE_HOLDER_2"

Put the generated code inside ```C ``` and note that the final result should a function that takes all the input args of BB and more if required. (Do not include comments)
""", "ext_wl":
"""Here's two code snippets AA and BB, each including a function. Add some parts of the logic of AA to BB in a way that the result would include important lines of BB with some parts from AA. You can add variables to BB to produce a more correct code.
                  
Code Snippet AA: "$MY_UNIQUE_PLACE_HOLDER_1"

Code Snippet BB: "$MY_UNIQUE_PLACE_HOLDER_2"
                  
The following lines from BB are important and should be included in the final result, the rest may be changed or even removed if they have no relation to these lines: (Lines are separated via /~/)

"$MY_UNIQUE_PLACE_HOLDER_3"

Put the generated code inside ```C ``` and note that the final result should a function that takes all the input args of BB and more if required. (Do not include comments)
""", "mutation_naive": """Here's a code snippets including a function. Except for the important lines mentioned below and the function's name and input variables, mutate the rest of the code so it is different from the original while keeping the original semantics.
                  
Code Snippet: "$MY_UNIQUE_PLACE_HOLDER_1"
                  
The following lines are important and should be included in the final result, the rest may be changed or even removed if they have no relation to these lines: (Lines are separated via /~/)

"$MY_UNIQUE_PLACE_HOLDER_2"

Put the generated code inside ```C ```. (Do not include comments)
""", "mutation": """Here's a code snippet including a function. Except for the important lines mentioned below, mutate the rest of the code snippet so it is different from the original while keeping the original semantics.
To do so you can use one or more of the following rules:

Rules: 1-> Replace the local variables’ identifiers with new non-repeated identifiers
2-> Replace the for statement with an semantic-equivalent while statement, and vice versa
3-> Change the assignment x++ into x=x+1 or x=x+1 into x+=1
4-> Merge the declaration statements into a single composite declaration statement
5-> Divide the composite declaration statement into separated declaration statements
6-> Switch the two code blocks in the if statement and the corresponding else statement
7-> Change a single if statement into a conditional expression statement.
8-> Change a conditional expression statement into a single if statement
9-> Divide a infix expression into two expressions whose values are stored in temporary variables
10-> Divide a if statement with a compound condition (∧ , ∨, or ¬) into two nested if statements
11-> Switch the places of two adjacent statements in a code block, where the former statement has no shared variable with the latter statement
12-> Replace the if-continue statement in a loop block with if-else statement
13-> Switch the two expressions on both sides of the infix expression whose operator is =
14-> Switch the two string objects on both sides of an equality expression
15-> Divide a pre-fix or post-fix expression into two expressions whose values are stored in temporary variables
16-> Transform the ‘‘Switch-Case’’ statement into the corresponding ‘‘If-Else’’ statement.

Code Snippet: "$MY_UNIQUE_PLACE_HOLDER_1"
                  
The following lines are important and should be included in the final result, but they can still be changed using only the first 5 rules, the rest may be changed using any of the rules or can even be removed if they have no relation to these lines: (Lines are separated via /~/)

"$MY_UNIQUE_PLACE_HOLDER_2"

Put the generated code inside ```C ```. (Do not include comments)
"""  # 2 of the rules were merged for simplicity
}

class CodeQwenPrompter:
    def __init__(self, system_message="You are a helpful assistant.", lock=None) -> None:
        self.messages = None
        self.system_message = system_message
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")
        self.lock = lock
        self.prompt = self._prompt
        if lock != None:
            self.selector = 0
            self.prompt=self._multi_model_prompt
            self.model = AutoModelForCausalLM.from_pretrained("Qwen/CodeQwen1.5-7B-Chat", torch_dtype="auto", device_map="balanced", max_memory={0: "8GB", 1: "8GB", 2: "8GB", 3: "8GB"},cache_dir="./model_cache/")
            self.model2 = AutoModelForCausalLM.from_pretrained("Qwen/CodeQwen1.5-7B-Chat", torch_dtype="auto", device_map="balanced", max_memory={0: "8GB", 1: "8GB", 2: "8GB", 3:"8GB"} ,cache_dir="./model_cache/")
            # self.model3 = AutoModelForCausalLM.from_pretrained("Qwen/CodeQwen1.5-7B-Chat", torch_dtype="auto", device_map="balanced", max_memory={0: "8GB", 1: "8GB", 2: "8GB", 3: "8GB"} ,cache_dir="./model_cache/")
            # self.model4 = AutoModelForCausalLM.from_pretrained("Qwen/CodeQwen1.5-7B-Chat", torch_dtype="auto", device_map="balanced", max_memory={0: "8GB", 1: "8GB", 2: "8GB", 3: "8GB"} ,cache_dir="./model_cache/")
            # self.models = {0:self.model, 1:self.model2, 2: self.model3, 3:self.model4}
            self.models = {0:self.model, 1:self.model2}

        else:
            self.model = AutoModelForCausalLM.from_pretrained("Qwen/CodeQwen1.5-7B-Chat", torch_dtype="auto", device_map="balanced", cache_dir="./model_cache/" , max_memory={0: "15GB", 1: "15GB"})

    def _multi_model_prompt(self, message, clean=True, show=False):
        model = None
        print(self.selector)
        with self.lock:
            model = self.models[self.selector % len(self.models)]
            self.selector += 1
            
                # print(t)
        if self.messages == None or clean:
            self.messages = [
                # system message to set the behavior of the assistant
                {"role": "system", "content": self.system_message},
            ]

        self.messages.append({"role": "user", "content": message})
        
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=4096 # ChatGPT defaults to 4096
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        reply = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.messages.append({"role": "assistant", "content": reply})

        if show:
            print(reply)

        return reply


    # clean initially, it is always dirty after use
    def _prompt(self, message, clean=True, show=False):
        # print(t)
        if self.messages == None or clean:
            self.messages = [
                # system message to set the behavior of the assistant
                {"role": "system", "content": self.system_message},
            ]

        self.messages.append({"role": "user", "content": message})
        
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=4096 # ChatGPT defaults to 4096
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        reply = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.messages.append({"role": "assistant", "content": reply})

        if show:
            print(reply)

        return reply

class ChatGPTPrompter:
    def __init__(self, api_key, system_message="Hi ChatGPT, You are a helpful assistant!") -> None:
        openai.api_key = api_key
        self.messages = None
        self.system_message = system_message

    # clean initially, it is always dirty after use
    def prompt(self, message, clean=True, show=False, t=0.5):
        # print(t)
        if self.messages == None or clean:
            self.messages = [
                # system message to set the behavior of the assistant
                {"role": "system", "content": self.system_message},
            ]
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=self.messages, temperature=t)
            reply = chat_completion["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": reply})

        self.messages.append({"role": "user", "content": message})
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=self.messages)
        reply = chat_completion["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": reply})

        if show:
            print(reply)

        return reply


class PromptUtils:
    @staticmethod
    def extract_code_content(text):
        start_index = text.find('```C') + 4
        end_index = text.find('```', start_index)
        if start_index != -1 and end_index != -1:
            return text[start_index:end_index].strip()
        else:
            print("Sth is wrong!")
            return None

    @staticmethod
    def get_vardef(vul, vulnerable_lines):
        # Step 1: Extract the arguments string from the function definition
        # Using re.DOTALL to allow the dot to match newlines
        args_match = re.search(r'\((.*?)\)', vul, re.DOTALL)
        if args_match:
            args_str = args_match.group(1)
        else:
            return ""

        # Step 2: Split the arguments string into individual arguments
        # This uses a regular expression to split on commas not inside parentheses
        full_args = [arg.strip()
                     for arg in re.split(r',\s*(?![^()]*\))', args_str)]

        # Step 3: Extract argument names, ignoring types
        args = []
        for arg in full_args:
            # Handle cases with pointers and references
            arg_name = arg.split()[-1].replace('*', '').replace('&', '')
            args.append(arg_name)

        # Step 4: Check which arguments are used in vulnerable_lines
        used_args = []
        used_full_args = []
        for i, arg in enumerate(args):
            if re.search(r'\b' + re.escape(arg) + r'\b', vulnerable_lines):
                used_args.append(arg)
                used_full_args.append(full_args[i])

        # Step 5: Format and print the results
        return f"(Don't Forget to add these variables to function's input arguments: {', '.join(full_arg for arg, full_arg in zip(used_args, used_full_args))})"

    @staticmethod
    def get_full_prompt_without_examples_and_vardef_guide(vul, clean, vulnerable_lines):
        vardef = PromptUtils.get_vardef(vul, vulnerable_lines)
        return templates["without_example_vardef"].replace(place_holders[1], vul).replace(place_holders[2], clean).replace(place_holders[3], vulnerable_lines).replace(place_holders[4], vardef)

    @staticmethod
    def get_full_prompt_with_examples_and_vardef_guide(fixed, vul, clean, vulnerable_lines):
        vardef = PromptUtils.get_vardef(vul, vulnerable_lines)
        return templates["with_example_vardef"].replace(place_holders[1], fixed).replace(place_holders[2], vul).replace(place_holders[3], clean).replace(place_holders[4], vulnerable_lines).replace(place_holders[5], vardef)

    @staticmethod
    def get_full_prompt(snippet1, snippet2, vulnerable_lines):
        return templates["extra"].replace(place_holders[1], snippet1).replace(place_holders[2], snippet2).replace(place_holders[3], vulnerable_lines)

    @staticmethod
    def get_ext_wl_prompt(clean, vul, vulnerable_lines):
        return templates["ext_wl"].replace(place_holders[1], clean).replace(place_holders[2], vul).replace(place_holders[3], vulnerable_lines)

    @staticmethod
    def get_mutation_prompt(vul, vulnerable_lines):
        return templates["mutation"].replace(place_holders[1], vul).replace(place_holders[2], vulnerable_lines)

    @staticmethod
    def get_ext_prompt(clean, vul):
        return templates["ext"].replace(place_holders[1], clean).replace(place_holders[2], vul)


    @staticmethod
    def get_prompt(snippet1, snippet2):
        return templates["base"].replace(place_holders[1], snippet1).replace(place_holders[2], snippet2)


class ConcurrencyUtils:

    @staticmethod
    def wait_for_all_futures(futures):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"something's up {e}")

    @staticmethod
    def concurrent_chatgpt_call(prompt_creator_from_searchItem: Callable[[SearchItem], str],
                                index_retreiver: Callable[[SearchItem], str],
                                vul_retriever: Callable[[SearchItem], str],
                                clean_retriever: Callable[[SearchItem], str],
                                vul_lines_retriever: Callable[[SearchItem], str],
                                search_item: SearchItem,
                                indices: List, vuls: List, cleans: List, vul_lines: List, generated: List,
                                lock, llm, retry=2) -> None:
        if retry <= 0:
            return
        
        try:
            prompt = prompt_creator_from_searchItem(search_item)
            code = PromptUtils.extract_code_content(llm.prompt(prompt))
            if code == None:
                print(f"Empty Code response - retry at {retry}")
                ConcurrencyUtils.concurrent_chatgpt_call(prompt_creator_from_searchItem,
                                                     index_retreiver,
                                                     vul_retriever,
                                                     clean_retriever,
                                                     vul_lines_retriever,
                                                     search_item,
                                                     indices, vuls, cleans, vul_lines, generated,
                                                     lock, llm, retry=retry - 1)

            with lock:
                generated.append(code)
                indices.append(index_retreiver(search_item))
                vuls.append(vul_retriever(search_item))
                cleans.append(clean_retriever(search_item))
                vul_lines.append(vul_lines_retriever(search_item))

            print("Success!")

        except (openai.error.APIError, torch.cuda.OutOfMemoryError) as e:
            print(f"ex happend {e}")
            # print(f"Arguments: {e.args}")
            print(f"Exception class: {e.__class__}")
            print(f"Cause: {e.__cause__}")
            print(f"Context: {e.__context__}")
            print(f"API Error - retry at {retry}")
            time.sleep(retry + 1) # maybe the server wasn't available so wait
            ConcurrencyUtils.concurrent_chatgpt_call(prompt_creator_from_searchItem,
                                                     index_retreiver,
                                                     vul_retriever,
                                                     clean_retriever,
                                                     vul_lines_retriever,
                                                     search_item,
                                                     indices, vuls, cleans, vul_lines, generated,
                                                     lock, llm, retry=retry - 1)
        except Exception as e:
            print(f"ex happend {e}")
            # print(f"Arguments: {e.args}")
            print(f"Exception class: {e.__class__}")
            print(f"Cause: {e.__cause__}")
            print(f"Context: {e.__context__}")

    @staticmethod
    def concurrent_prompter(prompt_creator_from_searchItem: Callable[[SearchItem], str],
                            index_retreiver: Callable[[SearchItem], str],
                            vul_retriever: Callable[[SearchItem], str],
                            clean_retriever: Callable[[SearchItem], str],
                            vul_lines_retriever: Callable[[SearchItem], str],
                            search_results: List[SearchItem], llm,
                            target: int = 2000, max_workers=60, is_all=False, resume_from_dataframe=None):
        vuls = []
        cleans = []
        vul_lines = []
        indices = []
        generated = []
        starting_index = 0

        if resume_from_dataframe is not None:
            print("resuming from dataframe")
            for index, row in resume_from_dataframe.iterrows():
                generated.append(str(row['generated']))
                indices.append(str(row['idx']))
                vuls.append(str(row["vul"]))
                vul_lines.append(str(row["vul_lines"]))
                cleans.append(str(row["clean"]))
        
            max_indx = -1
            for idx in indices:
                list_indx = next((j for j, result in enumerate(search_results) if result.best == idx), None)
                if list_indx != None and list_indx > max_indx:
                    max_indx = list_indx

            starting_index = max_indx + 1
        print(f"starting with index: {starting_index}")

        lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executors:
            futures = [executors.submit(
                ConcurrencyUtils.concurrent_chatgpt_call,
                prompt_creator_from_searchItem,
                index_retreiver, vul_retriever,
                clean_retriever, vul_lines_retriever,
                search_results[i], indices, vuls, cleans,
                vul_lines, generated, lock, llm) for i in range(starting_index,target)]

            ConcurrencyUtils.wait_for_all_futures(futures)
            end = target

            while len(generated) < target:
                
                if is_all:
                    break

                remaining = target - len(generated)
                new_limit = end + remaining

                print(f"DEBUG: Done is: {len(generated)}")
                print(f"DEBUG: Last i: {new_limit}")
                futures = [executors.submit(ConcurrencyUtils.concurrent_chatgpt_call,
                                            prompt_creator_from_searchItem,
                                            index_retreiver, vul_retriever,
                                            clean_retriever, vul_lines_retriever,
                                            search_results[i], indices, vuls, cleans,
                                            vul_lines, generated, lock, llm)
                           for i in range(end, new_limit)]
                ConcurrencyUtils.wait_for_all_futures(futures)
                end = new_limit

        return indices, vuls, cleans, vul_lines, generated
