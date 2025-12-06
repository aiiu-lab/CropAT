python edit_cli_all_image.py \
    --gpu-id 0 \
    --source-domain 'voc2007' \
    --resolution 512 \
    --steps 50 \
    --cfg-text 7.5 \
    --edit "an image in the style of <domain>" \
    --placeholder-token "<domain>" \
    --placeholder-token-ckpt-path exps/voc2clipart/checkpoint/placeholder_token_steps_39.safetensors \
    --output-dir "target_like/voc2007_to_clipart" \
    --seed 58912

python edit_cli_all_image.py \
    --gpu-id 7 \
    --source-domain 'voc2012' \
    --resolution 512 \
    --steps 50 \
    --cfg-text 7.5 \
    --edit "an image in the style of <domain>" \
    --placeholder-token "<domain>" \
    --placeholder-token-ckpt-path exps/voc2clipart/checkpoint/placeholder_token_steps_39.safetensors \
    --output-dir-name "target_like/voc2012_to_clipart" \
    --seed 58912