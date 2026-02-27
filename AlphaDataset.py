import pandas as pd
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
#pip install hf_xet

def get_text_embedding(text):
    if pd.isna(text) or text == "":
        return None

    try:
        if not hasattr(get_text_embedding, 'tokenizer'):
            get_text_embedding.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
            get_text_embedding.model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')

        inputs = get_text_embedding.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length'
        )

        with torch.no_grad():
            outputs = get_text_embedding.model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        return cls_embedding

    except Exception as e:
        print(f"Ошибка обработки текста '{text}': {e}")
        return None


def vectorize_all_text(df):

    df['text_embedding'] = df['text'].apply(get_text_embedding)

    vectorized = df['text_embedding'].notna().sum()
    print(f"\nУспешно векторизовано текстов: {vectorized} из {len(df)}")

    return df



def create_final_dataset(df):
    dataset = []

    for idx, row in df.iterrows():
        if (row['audio_features'] is None or
                row['video_features'] is None or
                row['text_embedding'] is None):
            continue

        record = {
            'text_embedding': row['text_embedding'],

            'f0_mean': row['audio_features']['f0_mean'],
            'f0_std': row['audio_features']['f0_std'],
            'energy_mean': row['audio_features']['energy_mean'],
            'energy_std': row['audio_features']['energy_std'],
            'num_pauses': row['audio_features']['num_pauses'],
            'pause_ratio': row['audio_features']['pause_ratio'],
            'mfcc_mean': row['audio_features']['mfcc_mean'],
            'mfcc_std': row['audio_features']['mfcc_std'],

            'video_features': row['video_features'].numpy(),

            'formal_text_tone': row['formal_text_tone'],
            'true_tone': row['true_tone'],
            'mismatch_type': row['mismatch_type'],

            'video_file': row['video_file'],
            'confidence': row['confidence']
        }

        dataset.append(record)

    print(f"\nСоздан датасет из {len(dataset)} записей")

    return dataset


def split_dataset(dataset):

    train_data, test_data = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42
        #stratify=[d['mismatch_type'] for d in dataset]
    )

    train_data, val_data = train_test_split(
        train_data,
        test_size=0.15,
        random_state=42
        #stratify=[d['mismatch_type'] for d in train_data]
    )

    print(f"Обучающая выборка: {len(train_data)}")
    print(f"Валидационная выборка: {len(val_data)}")
    print(f"Тестовая выборка: {len(test_data)}")

    return train_data, val_data, test_data


class SarcasmDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]

        text_emb = torch.tensor(record['text_embedding'], dtype=torch.float32)

        audio_features = torch.tensor([
            record['f0_mean'],
            record['f0_std'],
            record['energy_mean'],
            record['energy_std'],
            record['num_pauses'],
            record['pause_ratio']
        ], dtype=torch.float32)

        mfcc_features = torch.tensor(
            record['mfcc_mean'] + record['mfcc_std'],
            dtype=torch.float32
        )

        video_features = torch.tensor(record['video_features'], dtype=torch.float32)

        formal_tone = torch.tensor(record['formal_text_tone'], dtype=torch.long)
        true_tone = torch.tensor(record['true_tone'], dtype=torch.long)

        mismatch_types = {
            'саркастически+': -2,
            'саркастически–': 1,
            'ирония': -1,
            'вежливая критика': 2,
            'скрытое одобрение': 3,
            'отсутствие несоответствия': 0
        }
        mismatch_type = torch.tensor(
            mismatch_types.get(record['mismatch_type'], 5),
            dtype=torch.long
        )

        return {
            'text': text_emb,
            'audio': torch.cat([audio_features, mfcc_features]),
            'video': video_features,
            'formal_tone': formal_tone,
            'true_tone': true_tone,
            'mismatch_type': mismatch_type
        }


def create_dataloaders(train_data, val_data, test_data, batch_size=4):
    train_dataset = SarcasmDataset(train_data)
    val_dataset = SarcasmDataset(val_data)
    test_dataset = SarcasmDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset



def main():
    INPUT_PATH = "prepared_data.pkl"
    OUTPUT_DIR = "Al_dataset"

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH, 'rb') as f:
        df = pickle.load(f)

    df = vectorize_all_text(df)

    dataset = create_final_dataset(df)

    train_data, val_data, test_data = split_dataset(dataset)

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
        create_dataloaders(train_data, val_data, test_data)

    with open(f'{OUTPUT_DIR}/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(f'{OUTPUT_DIR}/val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open(f'{OUTPUT_DIR}/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    with open(f'{OUTPUT_DIR}/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(f'{OUTPUT_DIR}/val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)
    with open(f'{OUTPUT_DIR}/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = main()