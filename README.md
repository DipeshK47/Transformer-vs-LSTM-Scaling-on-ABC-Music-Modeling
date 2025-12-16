# Transformer-vs-LSTM-Scaling-on-ABC-Music-Modeling

This repository contains code and artifacts for a scaling law study on **symbolic music** modeled as **ABC notation** using **character level language models**.

It includes:

- Part 1: Data preprocessing and corpus creation (ABC to character corpus)
- Part 2: Transformer scaling study (tiny to xl)
- Part 3: LSTM (RNN) scaling study and comparison
- Part 4: Best model training, sample generation, and sample evaluation (ABC validity and ABC to MIDI conversion)

## Repository structure

Typical layout (your paths may differ):

- `preprocessing/` data pipeline scripts and statistics
- `part2/` transformer scaling scripts, logs, plots
- `part3/` rnn scaling scripts, logs, plots
- `part4/` best model training, sampling, evaluation

Key scripts mentioned in the report:

- `part2/` Transformer scaling
  - training script that trains five Transformer sizes on a fixed token budget
  - plotting script that produces training curves and scaling law fit plots
- `part3/` RNN scaling
  - `rnn_model.py` defines the LSTM language model family
  - `train_rnn_scaling.py` trains each LSTM size for one epoch under the fixed token budget
  - `stats.py` parses logs and produces scaling plots and summary metrics
- `part4/` Best model and sample evaluation
  - `train_gpt_part4.py` trains the best Transformer (XL) longer
  - sample generation scripts that write `.abc` outputs
  - sample evaluation script that checks syntax and converts to MIDI with `music21`

## Requirements

Recommended:

- Python 3.10+
- CUDA GPU for training (CPU will be extremely slow)

Python packages:

- `torch`
- `numpy`
- `matplotlib`
- `tqdm`
- `music21` (required for ABC to MIDI conversion during sample evaluation)

Example install:

```bash
pip install numpy matplotlib tqdm music21
# install torch for your CUDA version, example:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Data setup

The scripts expect a prepared **character corpus folder** with:

- `train.txt` 
- `val.txt`
- `vocab.json` (character to integer mapping)

The code you shared uses this path:

```text
/scratch/dk5288/data/abc_char_corpus_98_1_1
```

If you are running locally or on a different system, either:

- edit the `DATA_DIR` constant in the scripts, or
- refactor the scripts to accept a `--data_dir` argument.

## Part 2: Transformer scaling study

This experiment trains a family of decoder only Transformers with a fixed training recipe and a fixed token budget per run.

Model sizes:

- tiny
- small
- medium
- large
- xl

The configuration table for the Transformer family is included in the report.

### Run training

Example usage pattern (adjust to your actual script names and paths):

```bash
python train_transformer_scaling.py --model_size tiny
python train_transformer_scaling.py --model_size small
python train_transformer_scaling.py --model_size medium
python train_transformer_scaling.py --model_size large
python train_transformer_scaling.py --model_size xl
```

### Produce plots and scaling summary

```bash
python stats_transformer.py /path/to/transformer_training_log.out
```

Expected outputs include:

- training loss curves across sizes
- validation loss vs parameters scaling plot with power law fit
- a text summary containing the per model metrics and fitted exponent

## Part 3: RNN (LSTM) scaling study

This experiment repeats the scaling study using an LSTM based character language model.

The model is defined in `rnn_model.py` (CharLSTM) and the main training script is `train_rnn_scaling.py`.

Key fixed settings (kept consistent with the Transformer scaling setup):

- character level tokenization
- context length `BLOCK_SIZE = 256`
- batch size `BATCH_SIZE = 64`
- optimizer AdamW, lr schedule with warmup then cosine decay
- fixed training token budget (one epoch definition)

### Run training

```bash
python train_rnn_scaling.py --model_size tiny
python train_rnn_scaling.py --model_size small
python train_rnn_scaling.py --model_size medium
python train_rnn_scaling.py --model_size large
```

This writes:

- checkpoints: `rnn_<size>_best.pt` and `rnn_<size>_final.pt`
- a results CSV: `rnn_scaling_results.csv`

### Parse logs and generate plots

`stats.py` parses the combined training log and produces:

- `rnn_training_loss_curves_all_models.png`
- `rnn_scaling_law_val_loss_vs_params.png`
- `rnn_scaling_analysis_summary.txt`
- `rnn_scaling_extra_metrics.csv`

Example:

```bash
python stats.py /scratch/dk5288/code/my_project/part3/final_model_logs.out
```

## Part 3: Transformer vs RNN comparison

The comparison includes:

- a combined scaling plot (validation loss vs parameter count for both architectures)
- computational efficiency plots (time vs parameters, memory vs parameters)

If you already generated these figures, keep them in a `plots/` folder and reference them in the report.

## Part 4: Best model training, sample generation, and evaluation

### Train the best model

`train_gpt_part4.py` trains the XL Transformer longer using:

- `TARGET_TOKENS = 200_000_000` per epoch
- `EPOCHS = 5`

Run:

```bash
python train_gpt_part4.py --out_dir /scratch/dk5288/models/part4_best
```

Outputs:

- `best_xl.pt` (best validation checkpoint)
- `final_xl.pt` (final checkpoint)

### Generate samples

Generate at least 10 samples:

- unconditional samples (`uncond_*.abc`)
- conditional samples (`cond_*.abc`) prompted by a prefix

Store them under a samples directory, for example:

```text
/scratch/dk5288/code/my_project/50_samples
```

### Evaluate samples

The evaluation script should:

- check ABC syntax validity
- attempt ABC to MIDI conversion using `music21`
- write a per file CSV report
- save converted MIDI files

From your run:

- Total ABC files: 50
- Syntactically valid: 35 (70.00%)
- Converted to MIDI (of all): 25 (50.00%)
- Converted to MIDI (of valid only): 25 (71.43%)

## Reproducing the report figures

Figures referenced in the report:

- Transformer training curves: `150M_training_loss_curves_all_models.png`
- Transformer scaling law plot: `150M_scaling_law_val_loss_vs_params.png`
- RNN training curves: `rnn_training_loss_curves_all_models.png`
- RNN scaling law plot: `rnn_scaling_law_val_loss_vs_params.png`
- Combined scaling plot: `transformer_vs_rnn_scaling.png`
- Combined time vs params: `combined_time_vs_params.png`
- Combined memory vs params: `combined_mem_vs_params.png`

## Notes

- Many scripts have hard coded paths under `/scratch/dk5288/...`. If you move environments, update `DATA_DIR` and output directories accordingly.
- ABC to MIDI conversion depends on valid ABC syntax; failures are expected for partially valid generations.

## License

This project is for coursework and research experimentation. Add a license here if you plan to publish the code publicly.
