dataname=('ADHD')
MODEL_PATH=(
    ./output_${dataname}_MSC-Net_full_8/pretrain_checkpoint-799.pth
)

for dataname in "${dataname[@]}"; do
  experiment_name=$(basename "$(dirname "$MODEL_PATH")")

  OUTPUT_DIR=./output_ADHD/${experiment_name}/

  for i in {1..5}; do
    echo "Run $i th project with model: $experiment_name"
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.run \
        --nproc_per_node=2 \
        tools/run_class_finetuning.py \
        --model msc_net_patch1_256 \
        --finetune $MODEL_PATH \
        --nb_classes 2 \
        --output_dir $OUTPUT_DIR \
        --batch_size 64 \
        --lr 5e-4 --update_freq 1 \
        --warmup_epochs 5 --epochs 200 --layer_decay 0.65 --drop_path 0.1 \
        --weight_decay 0.05 \
        --sin_pos_emb \
        --dist_eval \
        --dataset ${dataname} \
        --no_auto_resume \
        --KFolds 5 \
        --seed 3407 \
        --ith $i
  done
done
