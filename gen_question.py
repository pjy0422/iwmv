from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.json_utils import load_json, save_json
from gen_para import *


@dataclass
class Paraphrase(BaseModel):
    question: str
    answer: str


def get_system_prompt() -> str:
    return f"""
    """


def get_user_prompt(context) -> str:
    return f"""
    Generate a short and closed form question  and corresponding answer based on the following context:
    Context: {context}
    You should return question first, and then the answer.
    Limit length of the answer to four words.
    """


def gen_tuple(
    context: str
) -> Tuple[str, str, Dict[str, Any]]:
    num_pairs = 5
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(context)
    kwargs = {
        "model": "gpt-4o-mini",
        "max_tokens": 2000,
        "top_p": 1,
        "temperature": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": Paraphrase,
    }
    return system_prompt, user_prompt, kwargs


from numpy import random
def main():
    original_data_path = "data/0812_nq_wo_multianswer.json"
    new_data_path = "data/0816_context.json"
    original_data = load_json(original_data_path)
    original_data = original_data[:200]
    new_data = []
    for item in tqdm(original_data):
        paraphrase_list = item['paraphrase']
        rand_para = random.choice(paraphrase_list)
        para_tuple = gen_tuple(rand_para)
        counterfactual_list = item['counterfactual']
        rand_counterfactual = random.choice(counterfactual_list)
        cf_tuple = gen_tuple(rand_counterfactual['contexts'][0])
        handler = OpenaiQueryHandler(system_prompt=para_tuple[0], user_prompt=para_tuple[1], **para_tuple[2])
        para = handler.query_with_schema()
        handler = OpenaiQueryHandler(system_prompt=cf_tuple[0], user_prompt=cf_tuple[1], **cf_tuple[2])
        cf = handler.query_with_schema()
        new_data.append({
            "index": item["index"],
            "question": item["question"],
            "answers": item["answers"],
            "paraphrase": {
                "original_context": rand_para,
                "question": para.question,
                "answer": para.answer
            },
            "counterfactual": {
                "original_context": rand_counterfactual['contexts'][0],
                "question": cf.question,
                "answer": cf.answer
            },  
        })
    save_json(new_data_path, new_data)

if __name__ == "__main__":
    main()
