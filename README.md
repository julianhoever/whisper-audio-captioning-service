# Whisper Audio Captioning Service

This repository contains a simple audio captioning service built using OpenAI's Whisper model. The service accepts audio files via a REST API and returns descriptions of the soundscape.

## Usage
```bash
docker build -t audiocap https://github.com/julianhoever/whisper-audio-captioning-service.git
docker run --rm -it -p 80:80 -v <model_checkpoint>:/checkpoint:ro -e AC_CHECKPOINT=/checkpoint -e AC_ARCHITECTURE=<architecture> audiocap
```
Replace `<model_checkpoint>` with the path to the downloaded model checkpoint directory described below.

## Model Checkpoint Downloads

### Tiny
```bash
git clone https://huggingface.co/MU-NLPC/whisper-tiny-audio-captioning
```
Set `AC_ARCHITECTURE` to `openai/whisper-tiny`.

### Small
```bash
git clone https://huggingface.co/MU-NLPC/whisper-small-audio-captioning
```
Set `AC_ARCHITECTURE` to `openai/whisper-small`.

### Large
```bash
git clone https://huggingface.co/MU-NLPC/whisper-large-v2-audio-captioning
```
Set `AC_ARCHITECTURE` to `openai/whisper-large-v2`.

## API
The service exposes a single endpoint `/generate_caption` that accepts POST requests with an attached audio file.