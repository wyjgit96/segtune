cd "$(dirname "$0")"
cd ../

export PYTHONPATH=$PYTHONPATH:$PWD
export CUDA_VISIBLE_DEVICES=0

if [[ "$OSTYPE" =~ ^darwin ]]; then
    export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.dylib
fi

python3 infer/infer.py \
    --lrc-path infer/example/edit_cn.lrc \
    --ref-audio-path infer/example/edit_cn.mp3 \
    --audio-length 95 \
    --repo-id ASLP-lab/DiffRhythm-1_2 \
    --output-dir infer/example/output \
    --chunked \
    --edit \
    --ref-song infer/example/edit_cn.mp3 \
    --edit-segments "[[-1,41],[70,-1]]" \
    --batch-infer-num 1