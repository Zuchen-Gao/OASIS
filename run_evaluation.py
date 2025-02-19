import fire
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.evaluation.nl2code import main

eval_data_dir = "./data/evaluation/"

def evaluate_on_config(
    model_name_or_path: str,
    dataset_name: tuple[str] = ("csn", "cosqa", "adv"),
    remain_format: bool = False,
    pooling_strategy: str = "last",
    pooling_layer: int = -1,
    eval_batch_size: int = 256,
    use_valid: bool = False,
):
    
    result_display = {}
    for dataset in dataset_name:
        if dataset == "csn":
            test_data_file = os.path.join(eval_data_dir, "CodeSearchNet/clean_data/dataset/{lang}/test.jsonl" if not use_valid else "CodeSearchNet/clean_data/dataset/{lang}/valid.jsonl")
            candidate_database_file = os.path.join(eval_data_dir, "CodeSearchNet/clean_data/dataset/{lang}/codebase.jsonl")
            for lang in ["python", "java", "javascript", "php", "go", "ruby"]:
                result_display[lang] = main(
                    model_name_or_path,
                    test_data_file.format(lang=lang),
                    candidate_database_file.format(lang=lang),
                    remain_format,
                    pooling_strategy,
                    pooling_layer,
                    eval_batch_size,
                )
        elif dataset == "cosqa":
            test_data_file = os.path.join(eval_data_dir, "CoSQA/clean_data/cosqa-retrieval-test-500.json" if not use_valid else "CoSQA/clean_data/cosqa-retrieval-dev-500.json")
            candidate_database_file = os.path.join(eval_data_dir, "CoSQA/clean_data/code_idx_map_list_merged.json")
            result_display["cosqa"] = main(
                model_name_or_path,
                test_data_file,
                candidate_database_file,
                remain_format,
                pooling_strategy,
                pooling_layer,
                eval_batch_size,
            )
        elif dataset == "adv":
            test_data_file = os.path.join(eval_data_dir, "AdvTest/clean_data/dataset/test.jsonl" if not use_valid else "AdvTest/clean_data/dataset/valid.jsonl")
            candidate_database_file = os.path.join(eval_data_dir, "AdvTest/clean_data/dataset/test.jsonl" if not use_valid else "AdvTest/clean_data/dataset/valid.jsonl")
            result_display["adv"] = main(
                model_name_or_path,
                test_data_file,
                candidate_database_file,
                remain_format,
                pooling_strategy,
                pooling_layer,
                eval_batch_size,
            )
    print(result_display)

    result_list = []
    for result_name in ["cosqa", "adv", "python", "java", "javascript", "php", "go", "ruby"]:
        if result_name in result_display:
            result_list.append(result_display[result_name]["mrr_at_1000"])
    print([round(result, 4) for result in result_list])
    print("\t".join([str(round(result, 4)) for result in result_list]))
    return result_display


if __name__ == "__main__":
    fire.Fire(evaluate_on_config)