import json
from pprint import pprint

from tqdm import tqdm
from util import *


def postprocess_dict(text: dict, num_pairs: int):
    assert "result" in text
    text["result"] = text["result"][:num_pairs]
    return text


def check_answer(original_answer, answer):
    original_answer_words = original_answer.lower().split()
    answer_words = set(answer.lower().split())
    any_word_in_answer_words = any(
        word in answer_words for word in original_answer_words
    )
    if any_word_in_answer_words:
        return True
    return False


def query_para(
    question: str, answer: str, num_pairs: int = 9, top_k: int = 3, V: int = 30
):
    para_answer = gen_openai_para_answer(
        question=question, answer=answer, num_pairs=num_pairs, answer_limit=4
    )

    para_answer = str2dict(para_answer)
    para_answer_list = para_answer["answer"]
    para_answer_list = para_answer_list[:num_pairs]
    for item in para_answer_list:
        if not check_answer(answer, item):
            item += ", " + answer

    para_text = gen_openai_para_text(
        question=question,
        original_answer=answer,
        answers=para_answer_list,
        num_pairs=num_pairs,
        top_k=top_k,
        V=V,
    )
    para_text = str2dict(para_text)
    cnt = 0
    for item in para_text["result"]:
        cnt += len(item["text"])
    return para_text, cnt


def query_counterfactual(
    question: str, answer: str, num_pairs: int = 9, top_k: int = 3, V: int = 30
):
    cf_answer = gen_openai_counterfactual_answer(
        question=question, answer=answer, num_pairs=num_pairs, answer_limit=4
    )

    cf_answer = str2dict(cf_answer)
    cf_answer_list = cf_answer["answer"]
    cf_answer_list = cf_answer_list[:num_pairs]

    cf_text = gen_openai_counterfactual_text(
        question=question,
        original_answer=answer,
        answers=cf_answer_list,
        num_pairs=num_pairs,
        top_k=top_k,
        V=V,
    )
    cf_text = str2dict(cf_text)
    cnt = 0
    for item in cf_text["result"]:
        cnt += len(item["text"])
    return cf_text, cnt


def process_pipeline(
    question: str,
    answer: str,
    required_num_pairs: int = 27,
    final_num_pairs: int = 9,
    top_k: int = 3,
    max_repeats: int = 10,
    mode: str = "para",
):
    attempts = 0
    text = None

    for attempts in range(max_repeats):
        pprint(f"Attempt {attempts}")
        try:
            if mode == "para":
                text, cnt = query_para(
                    question=question,
                    answer=answer,
                    num_pairs=final_num_pairs,
                    top_k=top_k,
                    V=30,
                )
            elif mode == "cf":
                text, cnt = query_counterfactual(
                    question=question,
                    answer=answer,
                    num_pairs=final_num_pairs,
                    top_k=top_k,
                    V=50,
                )
            if cnt >= required_num_pairs:
                break
        except json.decoder.JSONDecodeError as e:
            pprint(f"JSONDecodeError on attempt {attempts}: {e}")
            continue
        except SyntaxError as e:
            pprint(f"SyntaxError on attempt {attempts}: {e}")
            continue

    pprint(f"Total attempts: {attempts}")
    if text:
        text = postprocess_dict(text, final_num_pairs)

    return text


if __name__ == "__main__":
    question = "panda is a national animal of which country"
    answer = "China"
    para = process_pipeline(question=question, answer=answer, mode="para")
    pprint(para)
    pprint(len(para["result"]))
    cf = process_pipeline(question=question, answer=answer, mode="cf")
    pprint(cf)
    pprint(len(cf["result"]))
