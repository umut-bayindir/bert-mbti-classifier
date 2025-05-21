import os
import json
from typing import List, Dict

import pandas as pd
from transformers import BertTokenizer


class DataLoader:
    """
    Handles loading and preprocessing of question-answer datasets for MBTI classification.
    """
    def __init__(self, questions_path: str, tokenizer_name: str = 'bert-base-uncased'):
        """
        :param questions_path: Path to the JSON file containing Q/A sets
        :param tokenizer_name: HuggingFace tokenizer identifier
        """
        self.questions_path = questions_path
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def load_questions(self) -> List[Dict]:
        """
        Load the list of questions and possible answers from a JSON file.
        Expects a list of dicts: [{"question": str, "options": [str, ...]}, ...]
        """
        with open(self.questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        return questions

    def preprocess_text(self, texts: List[str]) -> List[List[int]]:
        """
        Tokenize and encode a list of texts using BERT tokenizer.
        :param texts: List of strings to preprocess
        :return: List of token ID lists
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        return encoded['input_ids'].tolist()

    def load_and_preprocess(self) -> Dict[str, any]:
        """
        Full pipeline: load questions, extract text, and return tokenized inputs.
        """
        questions = self.load_questions()
        texts = [q['question'] for q in questions]
        token_ids = self.preprocess_text(texts)
        return {
            'questions': questions,
            'token_ids': token_ids
        }
