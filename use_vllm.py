import json

from tqdm import tqdm
from transformers import AutoTokenizer
from utils.json_utils import *
from vllm import LLM, SamplingParams

RAG_INCONTEXT_PROMPT = 'Answer the question based on the given context without your internal knowledge with one or few words. If you do not know the answer, just say "I don\'t know".\
\n\nContext: The Voting Rights Act of 1965 was a landmark piece of federal legislation in the United States that prohibits racial discrimination in voting. \
This act was signed into law by President Lyndon B. Johnson during the height of the Civil Rights Movement. \
It aimed to overcome legal barriers at the state and local levels that prevented African Americans from exercising their right to vote under the 15th Amendment\
\nQuestion: who was the Voting Rights Act of 1965 designed to help\
\nAnswer: African Americans\
\n\nContext: The Apollo 11 mission was a historic spaceflight that first landed humans on the Moon. \
Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle on the lunar surface. \
Armstrong became the first person to step onto the lunar surface six hours after landing on July 20, 1969. Aldrin joined him about 20 minutes later.\
\nQuestion: on what date did the first human moon landing occur\
\nAnswer: July 20, 1969\
\n\nContext: Mount Everest, located in the Himalayas on the border between Nepal and the Tibet Autonomous Region of China, is the highest mountain in the world. \
Its peak reaches an elevation of 8,848 meters (29,029 feet) above sea level. This iconic mountain has attracted climbers from all over the globe, ranging from highly experienced mountaineers to capable climbers willing to hire professional guides. \
Sir Edmund Hillary of New Zealand and Tenzing Norgay, a Sherpa of Nepal, made the first confirmed ascent of Mount Everest on May 29, 1953.\
\nQuestion: who made the first confirmed ascent of Mount Everest\
\nAnswer: Sir Edmund Hillary and Tenzing Norgay\
\n\nContext: The Great Barrier Reef, located off the coast of Queensland, Australia, is the world\'s largest coral reef system. Comprising over 2,900 individual reefs and 900 islands, it stretches over 2,300 kilometers. \
The reef is home to a vast diversity of marine life, including many species of fish, mollusks, and sea turtles. \
Unfortunately, the Great Barrier Reef is facing significant threats from climate change, particularly coral bleaching, \
which occurs when ocean temperatures rise and cause corals to expel the symbiotic algae living in their tissues, leading to a white, "bleached" appearance\
\nQuestion: what major threat does climate change pose to the Great Barrier Reef\
\nAnswer: rising ocean temperatures'


def gen_message(context, question):
    return [
        {"role": "user", "content": RAG_INCONTEXT_PROMPT},
        {
            "role": "user",
            "content": f"Context: {context}\nQuestion: {question}\nAnswer:",
        },
    ]


if __name__ == "__main__":

    model = LLM("meta-llama/Meta-Llama-3-8B-Instruct")
    sampling_params = SamplingParams(temperature=0.1, top_p=1)
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct"
    )

    data = load_json("data/0811_nq_final_clean.json")
    data = data
    cnt = 0
    final_list = []
    for item in data:
        new_list = []
        new_set = set()
        question = item["question"]
        for context in item["paraphrase"]:
            messages = gen_message(context, question)
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            output = model.generate(
                formatted_prompt, sampling_params=sampling_params
            )
            text = output[0].outputs[0].text
            if text[-1] == ".":
                text = text[:-1]
            new_set.add(text)
            temp_dict = {
                "context": context,
                "new_answer": text,
            }
            new_list.append(temp_dict)
        item["paraphrase"] = new_list
        item["new_answer"] = list(new_set)

        new_dict = {
            "index": item["index"],
            "question": item["question"],
            "answers": item["answers"],
            "new_answers": list(new_set),
            "paraphrase": item["paraphrase"],
            "counterfactual": item["counterfactual"],
            "irrelevant": item["irrelevant"],
        }
        final_list.append(new_dict)
        cnt += len(new_set)
        print(f"item {item['index']} done")
    print(cnt / len(data))
    save_json("data/0811_paraphrased_with_ans.json", final_list)
