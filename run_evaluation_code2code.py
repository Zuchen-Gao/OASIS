import fire
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.evaluation.code2code import main

def evaluate_on_config(
    model_name_or_path: str = "",
    dataset_dir: str = "./data/evaluation/Code2Code",
    eval_batch_size: int = 2,
):
    in_lang_lang = ["python", "java", "javascript", "typescript", "csharp", "c", "ruby", "php", "go"]
    result_display = {}
    for lang in in_lang_lang:
        result_display[lang] = main(
            model_name_or_path,
            dataset_dir,
            eval_batch_size,
            lang,
            lang,
        )
    logging.info(result_display)
    logging.info([round(result, 4) for result in result_display.values()])
    logging.info("\t".join([str(round(result, 4)) for result in result_display.values()]))
    return result_display

if __name__ == "__main__":
    fire.Fire(evaluate_on_config)