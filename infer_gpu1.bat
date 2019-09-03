call conda activate hyoje
python main.py --command infer -g 1 --alg cycle_3 -l cycle_3_exp1
python main.py --command infer -g 1 --alg cycle_3 -l cycle_3_exp2
python main.py --command infer -g 1 --alg cycle_3 -l cycle_3_exp3
python main.py --command infer -g 1 --alg proto_1 -l proto_exp1
python main.py --command infer -g 1 --alg proto_1 -l proto_exp2
python main.py --command infer -g 1 --alg proto_1 -l proto_exp3
pause