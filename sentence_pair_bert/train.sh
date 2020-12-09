for((i=1;i<2;i++));
do

CUDA_VISIBLE_DEVICES=3 python run_glue.py \
--cuda_device 3 \
--model_type bert \
--model_name_or_path /home/mhxia/workspace/BDCI/chinese_wwm_ext_pytorch \
--do_train \
--data_dir /home/mhxia/Semantic_coherence/sentence_pair_bert/data/train_dataset \
--num_labels 2 \
--output_dir /home/mhxia/workspace/my_models/Semantic_coherence/sentence_pair_bert/model1 \
--adversarial None \
--max_seq_length 256 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--dropout 0.1 \
--warmup_steps 0 \
--learning_rate 3e-5 \
--adam_epsilon 1e-8 \
--logging_steps 1942 \
--num_train_epochs 20 \
--evaluate_during_training \
--early_stop 2 \
--overwrite_output_dir \
--weight_decay 0  ;

done


#python train.py \
#--model_type bert \
#--model_name_or_path /home/mhxia/workspace/BDCI/chinese_wwm_ext_pytorch \
#--do_train \
#--do_eval \
#--data_dir /home/mhxia/BD/problem_equivalence_discrimination/data \
#--output_dir ./model_bert_wwm_ext$i \
#--max_seq_length 256 \
#--split_num 1 \
#--lstm_hidden_size 512 \
#--lstm_layers 1 \
#--lstm_dropout 0.1 \
#--eval_steps 200 \
#--per_gpu_train_batch_size 1 \
#--gradient_accumulation_steps 1 \
#--warmup_steps 0 \
#--per_gpu_eval_batch_size 1 \
#--learning_rate 5e-6 \
#--adam_epsilon 1e-6 \
#--weight_decay 0 \
#--train_steps 5000 ;