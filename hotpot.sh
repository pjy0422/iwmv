#!/bin/bash
python preprocess.py --dataset hotpot --data_name hotpot_sample.json
python gen_cf_answers.py --dataset hotpot
python gen_cf.py --dataset hotpot
# Optional: Clean the counterfactual with the following command
python clean_cf.py --dataset hotpot
python gen_para.py --dataset hotpot
# Optional: Clean the paragraphs with the following command
python clean_para.py --dataset hotpot
python postprocess.py --dataset hotpot