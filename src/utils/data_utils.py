import tokenize
import re

import torch.distributed as dist

from io import StringIO

class CSNDataCollator:
    def __init__(self, remain_format: bool = False):
        self.remain_format = remain_format

    def __call__(self, data):
        batch = {}
        batch["url"] = []
        batch["query"] = []
        batch["passage"] = []
        bs = len(data)
        for i in range(bs):
            url = data[i]["url"]
            query = " ".join(data[i]["docstring_tokens"])
            if self.remain_format:
                passage = data[i].get("code_wo_comment", "")
            else:
                passage = " ".join(data[i]["code_tokens"])
            batch["url"].append(url)
            batch["query"].append(query)
            batch["passage"].append(passage)
        return batch

class CoSQADataCollator:
    def __init__(self, remain_format: bool = False):
        self.remain_format = remain_format

    def __call__(self, data):
        batch = {}
        batch["url"] = []
        batch["query"] = []
        batch["passage"] = []
        bs = len(data)
        if "retrieval_idx" in data[0]: # Test
            for i in range(bs):
                url = str(data[i]["retrieval_idx"])
                query = " ".join(data[i]["doc"].split())
                batch["url"].append(url)
                batch["query"].append(query)
        else: # Codebase
            for i in range(bs):
                url = str(data[i]["url"])
                if self.remain_format:
                    passage = data[i]["code_wo_comment"]
                else:
                    passage = " ".join(data[i]["code"].split())
                batch["url"].append(url)
                batch["passage"].append(passage)
        return batch

class AdvTestDataCollator:
    def __init__(self, remain_format: bool = False):
        self.remain_format = remain_format

    def __call__(self, data):
        batch = {}
        batch["url"] = []
        batch["query"] = []
        batch["passage"] = []
        bs = len(data)
        for i in range(bs):
            url = data[i]["url"]
            if self.remain_format:
                query = " ".join(data[i]["docstring_tokens"])
                passage = data[i]["function_wo_comment"]
            else:
                query = " ".join(data[i]["docstring_tokens"])
                passage = " ".join(data[i]["function_tokens"])
            batch["url"].append(url)
            batch["query"].append(query)
            batch["passage"].append(passage)
        return batch

class CodeNetDataCollator:

    def __init__(self, lang: str):
        self.lang = lang

    @staticmethod
    def remove_comments_and_docstrings(source, lang):
        if lang in ['python']:
            io_obj = StringIO(source)
            out = ""
            prev_toktype = tokenize.INDENT
            last_lineno = -1
            last_col = 0
            for tok in tokenize.generate_tokens(io_obj.readline):
                token_type = tok[0]
                token_string = tok[1]
                start_line, start_col = tok[2]
                end_line, end_col = tok[3]
                ltext = tok[4]
                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out += (" " * (start_col - last_col))
                # Remove comments:
                if token_type == tokenize.COMMENT:
                    pass
                # This series of conditionals removes docstrings:
                elif token_type == tokenize.STRING:
                    if prev_toktype != tokenize.INDENT:
                        # This is likely a docstring; double-check we're not inside an operator:
                        if prev_toktype != tokenize.NEWLINE:
                            if start_col > 0:
                                out += token_string
                else:
                    out += token_string
                prev_toktype = token_type
                last_col = end_col
                last_lineno = end_line
            temp = []
            for x in out.split('\n'):
                if x.strip() != "":
                    temp.append(x)
            return '\n'.join(temp)
        elif lang in ['ruby']:
            return source
        else:
            def replacer(match):
                s = match.group(0)
                if s.startswith('/'):
                    return " "  # note: a space and not an empty string
                else:
                    return s

            pattern = re.compile(
                r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                re.DOTALL | re.MULTILINE
            )
            temp = []
            for x in re.sub(pattern, replacer, source).split('\n'):
                if x.strip() != "":
                    temp.append(x)
            return '\n'.join(temp)

    def __call__(self, data):
        batch = {}
        batch["index"] = []
        batch["label"] = []
        batch["func"] = []
        bs = len(data)
        for i in range(bs):
            index = data[i]["index"]
            label = data[i]["label"]
            func = " ".join(self.remove_comments_and_docstrings(data[i]["func"], self.lang).split())
            batch["index"].append(index)
            batch["label"].append(label)
            batch["func"].append(func)
        return batch

class RawEmbeddingTrainingDataCollator:
    def __init__(self, 
        query_column: str = "query", 
        passage_column: str = "passage",
        label_column: str = "label",
        language_column: str = "language",
        gt_column: str= "is_gt",
    ):
        self.query_column = query_column
        self.passage_column = passage_column
        self.label_column = label_column
        self.language_column = language_column
        self.gt_column = gt_column

    """
    To make the input zigzag style, such as [i[0][0], i[0][1], i[1][0], i[1][1], ...], where (i[0][0], o[0][1]) stands for a pair.
    """
    def __call__(self, data):
        batch = {}
        batch["sequences"] = []
        batch["labels"] = []
        batch["language"] = []
        batch["gt"] = []
        bs = len(data)

        for i in range(bs):
            query = data[i][self.query_column]
            passage = data[i][self.passage_column]
            label = data[i].get(self.label_column) # Could be None
            language = data[i][self.language_column]
            gt = data[i].get(self.gt_column) # Could be None
            batch["sequences"].append(query)
            batch["sequences"].append(passage)
            batch["labels"].extend([label] * 2)
            batch["language"].extend([language] * 2)
            batch["gt"].extend([gt] * 2)
        if None in batch["labels"]:
            batch["labels"] = None
        if None in batch["gt"]:
            batch["gt"] = None
        return batch