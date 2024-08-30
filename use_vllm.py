import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer
from utils.json_utils import *
from vllm import LLM, SamplingParams

RAG_INCONTEXT_PROMPT = "Find the answer to the question in the given context without using your internal knowledge. \
Provide only essential keywords without explanations or additional details. \
If you can not find the answer, just say \"I don't know\".\
\n\nContext: The Voting Rights Act of 1965 was a landmark piece of federal legislation in the United States that prohibits racial discrimination in voting. \
This act was signed into law by President Lyndon B. Johnson during the height of the Civil Rights Movement. \
It aimed to overcome legal barriers at the state and local levels that prevented African Americans from exercising their right to vote under the 15th Amendment\
\nQuestion: who was the Voting Rights Act of 1965 designed to help\
\nAnswer: African Americans\
\n\nContext: In the midst of the 20th century, amidst geopolitical tensions and scientific breakthroughs, \
the race for space exploration was at its peak. Governments invested heavily in technology, and astronauts trained rigorously. \
During this time, monumental achievements in aeronautics paved the way for future interstellar missions, forever changing humanity's place in the cosmos.\
\nQuestion: which astronauts were part of the Apollo 11 mission that first landed humans on the moon\
\nAnswer: I don't know\
\n\nContext: The process of photosynthesis occurs in the chloroplasts of plant cells, where sunlight is used to convert carbon dioxide and water into glucose and oxygen. \
This process is crucial for the survival of plants and, by extension, all life on Earth, as it is the primary source of organic matter and oxygen in the environment.\
\nQuestion: where does the process of photosynthesis take place in plant cells\
\nAnswer: in the chloroplasts\
\n\nContext: The Inflation Reduction Act was signed into law by President Joe Biden in August 2022. \
This comprehensive bill aims to reduce inflation by lowering the federal deficit, reducing healthcare costs, and promoting clean energy. \
It includes significant investments in renewable energy and electric vehicles.\
\nQuestion: what was the total cost of the Inflation Reduction Act\
\nAnswer: I don't know\
\n\nContext: The Paris Agreement is a landmark international treaty that aims to combat climate change by limiting global warming to well below 2 degrees Celsius compared to pre-industrial levels. \
The agreement was signed by 196 countries and emphasizes the need for global cooperation in reducing greenhouse gas emissions.\
\nQuestion: what is the main goal of the Paris Agreement\
\nAnswer: limiting global warming "

RAG_NAIVE_PROMPT = 'Given the following context:\
\nContext: [context]\
\nAnswer the following question based on the given information without your internal knowledge with one or a few words. If you do not know the answer confidently, just say "I don\'t know".\
Keep in mind to provide only essential keypoints without explanations or additional information.\
If you do not know the answer confidently, just say "I don\'t know".\
\nQuestion: [question]\
\nAnswer:'

LLM_NAIVE_PROMPT = 'Answer the question with one or a few words. If you do not know the answer confidently, just say "I don\'t know".\
Keep in mind to provide only essential keypoints without explanations or additional information.\
\nQuestion: who was the Voting Rights Act of 1965 designed to help\
\nAnswer: African Americans\
\nQuestion: on what date did the first human moon landing occur\
\nAnswer: July 20, 1969\
\nQuestion: what major threat does climate change pose to the Great Barrier Reef\
\nAnswer: rising ocean temperatures'


def gen_message(context, question):
    return [
        {"role": "system", "content": RAG_INCONTEXT_PROMPT},
        {
            "role": "user",
            "content": f"""Context:{context}\nQuestion: {question}\n
            Answer:
            """,
        },
    ]


def val_message(answer1, answer2):
    VAL_PROMPT = f"""
Check if the following answer is at least meaningly similar to the original answer.
Original answer: {answer1}
New answer: {answer2}
Answer only with "yes" or "no".
"""
    return [
        {
            "role": "user",
            "content": VAL_PROMPT.format(answer1=answer1, answer2=answer2),
        },
    ]


if __name__ == "__main__":

    model = LLM("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct"
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=1,
        max_tokens=64,
        stop_token_ids=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
    )
    original_data_path = "data/0824/0824_triviaQA_filtered_dup_questions.json"
    os.makedirs("data/0827", exist_ok=True)
    new_data_path = "data/0827/tqa_rag_incontext.json"
    data = load_json(original_data_path)

    final_list = []
    cnt = 0
    for item in data:
        question = item["question"]
        temp_list = []
        for cf in item["counterfactual"]:
            context = "\n".join(cf["contexts"])
            messages = gen_message(context, question)
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            output = model.generate(
                formatted_prompt, sampling_params=sampling_params
            )
            text = output[0].outputs[0].text

            matched = False
            for answer in cf["answers"]:
                if text.lower() in answer.lower():
                    matched = True
            temp_dict = {
                "original_cf_answers": cf["answers"],
                "new_cf_answers": text,
                "matched": matched,
            }
            temp_list.append(temp_dict)

        cnt += matched
        temp_dict = {
            "index": item["index"],
            "question": item["question"],
            "original_answers": item["answers_in_ctxs"],
            "result": temp_list,
        }
        final_list.append(temp_dict)
        print(f"index {item['index']} done")
    total = (cnt / len(data)) * 100
    print(f"Matched: {total:.2f}%")
    save_json(new_data_path, final_list)
