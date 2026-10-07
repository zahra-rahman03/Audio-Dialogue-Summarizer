# Audio-Dialogue-Summarizer
# Multimodal Speech Transcription & Dialogue Summarization Pipeline

An end-to-end multimodal pipeline that transcribes raw audio into text using OpenAI's Whisper and distills conversational transcripts into concise summaries via Facebook's BART-large-cnn.

## Pipeline Architecture
- **Automatic Speech Recognition (ASR):** `openai/whisper-small` using sliding-window chunking (30s windows) and FP16 batched GPU inference.
- **Abstractive Summarization:** `facebook/bart-large-cnn` utilizing token chunking (400-word blocks) to handle extended context.
- **Interactive UI:** Gradio interface supporting `.wav`/`.mp3` audio uploads, playback, and dual-pane rendering for transcripts and summaries.

## Zero-Shot Benchmark Results (SAMSum Dataset)
Evaluated on conversational dialogue from the SAMSum test set to benchmark zero-shot cross-domain performance:

| Metric | Score | Focus Area |
| :--- | :--- | :--- |
| **ROUGE-1** | **32.26%** | Keyword and named-entity extraction |
| **ROUGE-2** | **9.82%** | Bigram/phrasal agreement |
| **ROUGE-L** | **24.95%** | Structural fluency and longest common subsequence |

## How to Run
Click the **Open in Colab** badge in the notebook, select a **T4 GPU** runtime, and run all cells sequentially.
