Install Pytorch: https://pytorch.org/get-started/locally/

Install Cuda: https://developer.nvidia.com/cuda-zone

huggingface-cli login

pip install transformers[torch]

python classify.py --model_ckpt microsoft/unixcoder-base-nine --num_epochs 60 --num_warmup_steps 10 --batch_size 8 --learning_rate 5e-4 