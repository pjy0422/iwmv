#!/bin/bash
python preprocess.py --dataset triviaqa --data_name triviaqa_sample.json
python gen_cf_answers.py --dataset triviaqa
python gen_cf.py --dataset triviaqa
# Optional: Clean the counterfactual with the following command
python clean_cf.py --dataset triviaqa
python gen_para.py --dataset triviaqa
# Optional: Clean the paragraphs with the following command
python clean_para.py --dataset triviaqa
python postprocess.py --dataset triviaqa