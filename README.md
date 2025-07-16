# OASIS
## Quick Start
### Conda Env
```bash
conda create -n oasis python=3.10 -y
conda activate oasis
pip install -r requirements.txt
```
### Evaluation
For OpenAI Embedding model, add `api_key` and `url` in `/src/evaluation/nl2code_models.py`

**NL2code**
```bash
cd scripts
bash evaluation.sh Kwaipilot/OASIS-code-embedding-1.5B "NL2CodeEvaluation_OASIS" 0 # not use valid set
```
**Code2code**
```bash
cd scripts
bash evaluation_code2code.sh Kwaipilot/OASIS-code-embedding-1.5B "Code2CodeEvaluation_OASIS"
```
### Training

Dataset examples can be found in ```/data/training```, Huggingface Dataset is available at.

