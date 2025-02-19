# OASIS
## Quick Start
### Evaluation
For OpenAI Embedding model, add `api_key` and `url` in `/src/evaluation/nl2code_models.py`

**NL2code**
```bash
cd scripts
bash evaluation.sh /path/to/your/model "NL2CodeEvaluationshortmessage" 1 #use_valid set
```
**Code2code**
```bash
cd scripts
bash evaluation_code2code.sh /path/to/your/model "Code2CodeEvaluationshortmessage"
```
**Training**

Change `--train_data_path` to the actual training data path.

```bash
cd scripts
bash fine_tuning.sh
```
