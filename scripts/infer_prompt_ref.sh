cd "$(dirname "$0")"
cd ../

export PYTHONPATH=$PYTHONPATH:$PWD
export CUDA_VISIBLE_DEVICES=0

if [[ "$OSTYPE" =~ ^darwin ]]; then
    export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.dylib
fi

python3 infer/infer.py \
    --lrc-path infer/example/test1.lrc \
    --ref-prompt "slow, tenor, cello, male, relaxing, traditional Chinese folk, 50s" \
    --audio-length 132 \
    --repo-id ASLP-lab/DiffRhythm-1_2 \
    --output-dir infer/example/output \
    --chunked \
    --batch-infer-num 5
