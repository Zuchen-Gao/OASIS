# OASIS: Order-Augmented Strategy for Improved Code Search
## Update
- 🤗 [2025/03/12] Our latest Code Embedding Model [OASIS-code-1.5B](https://huggingface.co/Kwaipilot/OASIS-code-1.5B) is now released.
- 📝 [2025/03/12] Our preprint is now available at [OASIS-arxiv](https://arxiv.org/abs/2503.08161).
- 🔥 [2025/05/16] Our paper is accepted as a main conference paper in [ACL2025](https://2025.aclweb.org/program/main_papers/).
- 🤗 [2025/07/XX] All training data for OASIS is now available at [OASIS-53m-dataset](https://huggingface.co/datasets/Kwaipilot/OASIS-53M-dataset). 

## Framework
<p align="center">
<img src="/assets/OASIS-Framework-v2.jpg" alt="image" width="auto" height="auto">
</p>
OASIS begins by using program analysis to enhance prompts for
pairing code with generated docstrings. Then, these pairs are augmented and annotated for similarity, after which
suboptimal labeled negative pairs will be selected with AST and threshold strategies and similarity will be adjusted
subsequently. Finally, these refined similarity labels are used in the optimization of hybrid objective.

## Performance
### NL2Code Search
<p align="center">
<img src="/assets/nl2code.png" alt="image" style="width: 80%;">
</p>

### Code2Code Search
<p align="center">
<img src="/assets/code2code.png" alt="image" style="width: 80%;">
</p>

## Local Results Reproduction
### Conda Environment
```bash
conda create -n oasis python=3.10 -y
conda activate oasis
pip install -r requirements.txt
```
### Evaluation
For OpenAI Embedding model, add `api_key` and `url` in `/src/evaluation/nl2code_models.py`

**NL2Code Search**
```bash
cd scripts
bash evaluation.sh Kwaipilot/OASIS-code-embedding-1.5B "NL2CodeEvaluation_OASIS" 0 # not use valid set
```
**Code2Code Search**
```bash
cd scripts
bash evaluation_code2code.sh Kwaipilot/OASIS-code-embedding-1.5B "Code2CodeEvaluation_OASIS"
```
### Training

Dataset examples can be found in ```/data/training```, Huggingface Dataset is available at [OASIS-53m-dataset](https://huggingface.co/datasets/Kwaipilot/OASIS-53M-dataset).

