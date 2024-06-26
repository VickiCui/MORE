TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python src/train_more.py \
    --cold_start_step $INPUT_DROPOUT_STEP \
    --random_p $NOISY_RA_RATIO \
    --tokens_learning_rate $LEARNING_RATE_OF_TASK_PROMPT \
    --output_dir ./res/more \
    --use_image True \
    --image_num $NUMBER \
    --use_text True \
    --text_num $NUMBER \
    --do_pred \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 8 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.05 \
    --max_steps $TOTAL_TRAINING_STEP \
    --random_p $NOISY_RA_RATIO \
    --warmup_ratio 0.01 \
    --logging_strategy steps \
    --logging_steps 1000 \
    --logging_first_step False \
    --save_steps 1000 \
    --save_total_limit 1 \
    --seed 0 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --dataloader_num_workers 4 \
    --load_best_model_at_end True \
    --metric_for_best_model average \
    --predict_with_generate \
    --generation_max_length 32 \
    --generation_num_beams 5 \
    --overwrite_output_dir True \
    --overwrite_cache True \
    --disable_tqdm False 