dataname='ADHD'
OUTPUT_DIR="./output_ADHD/"
batch_size=8

# ============================ pretraining ============================
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.run \
  --nproc_per_node=2 \
  tools/run_pretraining.py \
  --output_dir ${OUTPUT_DIR} \
  --model msc_base_patch_256 \
  --batch_size ${batch_size} --lr 5e-4 --warmup_epochs 2 --epochs 800 \
  --clip_grad 7.0 --layer_scale_init_value 0.1 \
  --drop_path 0.1 \
  --mask_generator random \
  --decoder_layer_scale_init_value 0.1 \
  --no_auto_resume \
  --save_ckpt_freq 100 \
  --exp_name 'pretrain' \
  --input_size 264 \
  --ratio_mask_patches 0.5 \
  --predictor_depth 2 \
  --decoder_depth 2 \
  --align_loss_weight 2 \
  --dataset ${dataname}