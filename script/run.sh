#!/usr/bin/env bash


#python main.py --n_epochs 10 --gpu 2 --exp_name cycle_2 --alg cycle_1
# python main.py --n_epochs 2000 --gpu 3 --exp_name proto_2 --alg proto_1 &


python main.py --force True --n_epochs 2 --command train -g 0 --alg recommendation_model -n recommendation_model
