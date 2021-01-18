## End-to-End Mono to Binaural Conversion with Conv-TasNet
<br/>

This project is the result of a research subject in the Bachelor's Degree in Data Science and Engineering in Universitat Politècnica de Catalunya (UPC).

It is an end-to-end approach to mono to binaural conversion, having [2.5D Visual Sound](https://github.com/facebookresearch/2.5D-Visual-Sound) as the baseline and focusing on Conv-TasNet's architecture.

More information can be found in `paper_mono2binaural_tasnet.pdf`.

### Training and Testing
(The code has beed tested under the following system environment: Ubuntu 18.04.5 LTS, CUDA 11.1, Python 3.6.9, PyTorch 1.6.0)
1. Download the [FAIR-Play](https://github.com/facebookresearch/FAIR-Play) dataset.

2. Generate the frames from the mp4 videos with the script `generate_frames.py`.

3. Set relative path to the splits with the script `generate_splits.py`.

4. [OPTIONAL] Preprocess the audio files using reEncodeAudio.py to accelerate the training process.

5. Use the following command to train a model:
```
python3 train.py --hdf5FolderPath /YOUR_CODE_PATH/2.5d_visual_sound/hdf5/ --name mono2binaural --model audioVisual --checkpoints_dir /YOUR_CHECKPOINT_PATH/ --save_epoch_freq 50 --display_freq 10 --save_latest_freq 100 --batchSize 32 --learning_rate_decrease_itr 10 --niter 1000 --lr_visual 0.0001 --lr_audio 0.001 --nThreads 32 --gpu_ids 0,1,2,3,4,5,6,7 --validation_on --validation_freq 100 --validation_batches 50 --tensorboard True --model MODEL_NAME --use_visual_info |& tee -a training.log
```

The MODEL_NAME is either `tasnet` or `audioVisual`. If it does not fit into the gpu, use the `--stepBatchSize` parameter.

6. Use the following command to test your trained mono2binaural model:
```
python3 demo.py --input_audio_path /BINAURAL_AUDIO_PATH --video_frame_path /VIDEO_FRAME_PATH --weights_visual /VISUAL_MODEL_PATH --weights_audio /AUDIO_MODEL_PATH --output_dir_root /YOUT_OUTPUT_DIR/ --input_audio_length 10 --hop_size 0.05 --model MODEL_NAME --use_visual_info
```

7. Use the following command for evaluation:
```
python evaluate.py --results_root /YOUR_RESULTS --normalization True
```


### Acknowlegements
This code is manly based on 2.5 Visual Sound (https://github.com/facebookresearch/2.5D-Visual-Sound).

The Conv-TasNet implementation is based on Demucs (https://github.com/facebookresearch/demucs).


### Licence
The code is CC BY 4.0 licensed, as found in the LICENSE file.
