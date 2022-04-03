bash
source activate dgx
nvidia-smi
top -bn 1 -c -i
export OMP_NUM_THREADS=4
# cd ../timm_pretrain
# pip install --upgrade .
# cd ../Swin-Transformer

# python -m torch.distributed.launch --nproc_per_node=5 --master_port=12345  main.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /scratch/cluster/dilin/datasets/imagenet --batch-size 216 --amp-opt-level O0 --accumulation-steps 1 --output /scratch/cluster/cygong/swin_tiny_patch4_multiscalewindow2net --resume /scratch/cluster/cygong/swin_tiny_patch4_multiscalewindow/swin_tiny_patch4_window7_224/default/ckpt.pth
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345  main.py --cfg configs/swin_base_patch4_window7_224.yaml --data-path /scratch/cluster/dilin/datasets/imagenet --batch-size 72 --amp-opt-level O0 --accumulation-steps 4 --output /scratch/cluster/cygong/swin_base_patch4_noshift
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345  main.py --cfg configs/ms_swin_tiny_patch4_window7_224.yaml --data-path /scratch/cluster/dilin/datasets/imagenet --batch-size 108 --amp-opt-level O0 --accumulation-steps 3 --output /scratch/cluster/cygong/swin_tiny_patch4_multiscale_shuffletoken

# python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345  main.py --cfg configs/swin_small_patch4_window7_224.yaml --data-path /scratch/cluster/dilin/datasets/imagenet --batch-size 90 --amp-opt-level O0 --accumulation-steps 3 --output /scratch/cluster/cygong/swin_small_patch4_multiscalewindow_noshift_convproj_res2softmax --resume /scratch/cluster/cygong/swin_small_patch4_multiscalewindow_noshift_convproj_res2softmax/swin_small_patch4_window7_224/default/ckpt.pth #  --resume /scratch/cluster/cygong/swin_tiny_patch4_multiscalewindow_headattn/swin_tiny_patch4_window7_224/default/ckpt.pth # --resume /scratch/cluster/cygong/swin_tiny_patch4_window7_224.pth
# CUDA_VISIBLE_DEVICES=1,2,3,4,7 python -m torch.distributed.launch --nproc_per_node=5 --master_port=12345  main.py --cfg configs/ms_swin_tiny_patch4_window7_224.yaml --data-path /scratch/cluster/dilin/datasets/imagenet --batch-size 60 --amp-opt-level O0 --accumulation-steps 2 --output /scratch/cluster/cygong/swin_tiny_patch4_multiscalewindow_noshift --resume /scratch/cluster/cygong/swin_tiny_patch4_multiscalewindow_noshift/swin_tiny_patch4_window7_224/default/ckpt.pth # --resume /scratch/cluster/cygong/swin_tiny_patch4_window7_224.pth

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345  main.py --cfg configs/ms_swin_tiny_patch4_window7_224.yaml --data-path /scratch/cluster/dilin/datasets/imagenet --batch-size 54  --amp-opt-level O0 --accumulation-steps 1 --output /scratch/cluster/cygong/swin_tiny_patch4_multiscalewindow_noshift_convproj_res2softmax  --resume /scratch/cluster/cygong/swin_tiny_patch4_multiscalewindow_noshift_convproj_res2softmax/swin_tiny_patch4_window7_224/default/ckpt.pt #  --resume /scratch/cluster/cygong/swin_tiny_patch4_window7_224.pth


# CUDA_VISIBLE_DEVICES=1,2,3,4,7
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 distill.py --cfg configs/ms_swin_mobile_patch4_window7_224.yaml --data-path /scratch/cluster/dilin/datasets/imagenet --batch-size 80 --amp-opt-level O0 --accumulation-steps 2 --output /scratch/cluster/cygong/swin_mobile_patch4_window7_kl --resume  /scratch/cluster/cygong/swin_mobile_patch4_window7/swin_mobile_patch4_window7_224/default/ckpt.pth  # --resume /scratch/cluster/cygong/swin_small_patch4_window7_224.pth
