#!/bin/bash

# total_neurons = 296991
# core_neurons = 66652
python parallel_training_testing_allen_vscode.py --neurons 66652 --seq_len 500 --loss_core_radius 200 --plot_core_radius 200 --delays 0,0 --train_recurrent --osi_loss_method 'crowd_osi' --osi_cost 10 --rate_cost 10000 --voltage_cost 0 --recurrent_weight_regularization 0 --sync_cost 0.3 --learning_rate 0.001 --n_runs 20 --n_epochs 50 --steps_per_epoch 25 --train_noise --data_dir 'GLIF_network' --results_dir 'Simulation_results' 
# --restore_from 'Simulation_results/v1_66652/b_n0a2/Intermediate_checkpoints'
