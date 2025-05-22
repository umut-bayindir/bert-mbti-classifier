import torch
import pandas as pd
from transformers import BertTokenizer
from model import MBTIClassifier
from explanations import MBTI_EXPLANATIONS


def run_inference(
    questions_file, model_path, tokenizer_name='bert-base-uncased', max_length=128
):
    questions = pd.read_csv(questions_file)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = MBTIClassifier(num_labels=4)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    answers = []
    print("Answer the following questions with 1 (yes) or 0 (no):")
    for idx, row in questions.iterrows():
        while True:
            ans = input(f"Q{row.question_id}: {row.question} ")
            if ans in ('0', '1'):
                answers.append(int(ans))
                break
            print("Please enter 1 or 0.")

    # Prepare inputs
    inputs = tokenizer(
        questions['question'].tolist(),
        padding='max_length', truncation=True,
        max_length=max_length, return_tensors='pt'
    )
    logits = model(inputs['input_ids'], inputs['attention_mask'])
    # Each logit corresponds to one dichotomy: I/E, N/S, T/F, J/P
    preds = torch.argmax(logits, dim=1).tolist()

    # Map preds to letters
    mbti = ''.join([
        'I' if preds[0]==0 else 'E',
        'N' if preds[1]==0 else 'S',
        'T' if preds[2]==0 else 'F',
        'J' if preds[3]==0 else 'P'
    ])

    print(f"\nPredicted MBTI: {mbti}\n")
    print(MBTI_EXPLANATIONS.get(mbti, "No explanation available."))


if __name__ == '__main__':
    run_inference('data/questions.csv', 'models/bert_finetuned/model_epoch3.pt')
```