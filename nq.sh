python preprocess.py --dataset nq --data_name nq_sample.json
python gen_cf_answers.py --dataset nq
python gen_cf.py --dataset nq
python gen_para.py --dataset nq
python postprocess.py --dataset nq