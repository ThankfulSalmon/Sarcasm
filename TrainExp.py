import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import BertTokenizer, BertModel
import warnings

warnings.filterwarnings('ignore')


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



class TextEncoder(nn.Module):

    def __init__(self, input_dim=768, output_dim=768):
        super(TextEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x


class AudioEncoder(nn.Module):

    def __init__(self, input_dim=38, output_dim=768):
        super(AudioEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class VideoEncoder(nn.Module):

    def __init__(self, input_dim=2304, output_dim=768):
        super(VideoEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.network(x)


class AttentionBottleneck(nn.Module):

    def __init__(self, dim=768, num_bottlenecks=4):
        super(AttentionBottleneck, self).__init__()

        self.bottlenecks = nn.Parameter(torch.randn(num_bottlenecks, dim))

        self.num_heads = 4
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)

        bottlenecks = self.bottlenecks.unsqueeze(0).expand(batch_size, -1, -1)

        q = self.q_proj(bottlenecks)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.num_heads * self.head_dim)

        output = self.out_proj(attn_output)
        output = self.norm(output + bottlenecks)

        return output.mean(dim=1)


class MultimodalSarcasmModel(nn.Module):

    def __init__(self, num_classes=6):
        super(MultimodalSarcasmModel, self).__init__()

        self.text_encoder = TextEncoder()
        self.audio_encoder = AudioEncoder()
        self.video_encoder = VideoEncoder()

        self.text_audio_bottleneck = AttentionBottleneck()
        self.text_video_bottleneck = AttentionBottleneck()
        self.audio_video_bottleneck = AttentionBottleneck()

        self.fusion_layer = nn.Sequential(
            nn.Linear(768 * 3, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, num_classes)
        )

    def forward(self, text, audio, video):
        text_emb = self.text_encoder(text)
        audio_emb = self.audio_encoder(audio)
        video_emb = self.video_encoder(video)


        ta_input = torch.stack([text_emb, audio_emb], dim=1)
        ta_fused = self.text_audio_bottleneck(ta_input)

        tv_input = torch.stack([text_emb, video_emb], dim=1)
        tv_fused = self.text_video_bottleneck(tv_input)

        av_input = torch.stack([audio_emb, video_emb], dim=1)
        av_fused = self.audio_video_bottleneck(av_input)

        combined = torch.cat([ta_fused, tv_fused, av_fused], dim=1)

        output = self.fusion_layer(combined)

        return output


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):

    best_val_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['mismatch_type'].to(device)

            optimizer.zero_grad()

            outputs = model(text, audio, video)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                text = batch['text'].to(device)
                audio = batch['audio'].to(device)
                video = batch['video'].to(device)
                labels = batch['mismatch_type'].to(device)

                outputs = model(text, audio, video)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss / len(train_loader):.4f}, Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss / len(val_loader):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    model.load_state_dict(torch.load('best_model.pth'))
    return model


def train_bimodal_models(train_loader, val_loader, device='cuda'):

    print("Обучение Текст + Аудио")
    model_ta = BimodalModel(modalities=['text', 'audio']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ta.parameters(), lr=1e-4)
    model_ta = train_model(model_ta, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

    print("\nОбучение Текст + Видео")
    model_tv = BimodalModel(modalities=['text', 'video']).to(device)
    optimizer = optim.Adam(model_tv.parameters(), lr=1e-4)
    model_tv = train_model(model_tv, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

    return model_ta, model_tv


class BimodalModel(nn.Module):

    def __init__(self, modalities=['text', 'audio']):
        super(BimodalModel, self).__init__()
        self.modalities = modalities
        self.num_classes = 6

        if 'text' in modalities:
            self.text_encoder = TextEncoder()
        if 'audio' in modalities:
            self.audio_encoder = AudioEncoder()
        if 'video' in modalities:
            self.video_encoder = VideoEncoder()

        input_dim = 768 * len(modalities)

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, self.num_classes)
        )

    def forward(self, text=None, audio=None, video=None):
        embeddings = []

        if 'text' in self.modalities and text is not None:
            embeddings.append(self.text_encoder(text))
        if 'audio' in self.modalities and audio is not None:
            embeddings.append(self.audio_encoder(audio))
        if 'video' in self.modalities and video is not None:
            embeddings.append(self.video_encoder(video))

        combined = torch.cat(embeddings, dim=1)
        return self.fusion(combined)


def evaluate_model(model, test_loader, device='cuda'):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['mismatch_type'].to(device)

            outputs = model(text, audio, video)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nМатрица ошибок:")
    print(conf_matrix)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'labels': all_labels
    }


#def analyze_mismatch_impact(model, test_loader, device='cuda'):
    #model.eval()


def compare_with_baselines(train_loader, val_loader, test_loader, device='cuda'):

    results = {}

    print("\nМодель: только текст")
    text_only = BimodalModel(modalities=['text']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(text_only.parameters(), lr=1e-4)
    text_only = train_model(text_only, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
    results['text_only'] = evaluate_model(text_only, test_loader, device)

    print("\nМодель: текст + аудио")
    ta_model = BimodalModel(modalities=['text', 'audio']).to(device)
    optimizer = optim.Adam(ta_model.parameters(), lr=1e-4)
    ta_model = train_model(ta_model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
    results['text_audio'] = evaluate_model(ta_model, test_loader, device)

    print("\nМодель: текст + видео")
    tv_model = BimodalModel(modalities=['text', 'video']).to(device)
    optimizer = optim.Adam(tv_model.parameters(), lr=1e-4)
    tv_model = train_model(tv_model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
    results['text_video'] = evaluate_model(tv_model, test_loader, device)

    print("\nМодель: текст + аудио + видео с Attention Bottlenecks")
    full_model = MultimodalSarcasmModel().to(device)
    optimizer = optim.Adam(full_model.parameters(), lr=1e-4)
    full_model = train_model(full_model, train_loader, val_loader, criterion, optimizer, num_epochs=15, device=device)
    results['full_model'] = evaluate_model(full_model, test_loader, device)


    for name, metrics in results.items():
        model_name = {
            'text_only': 'Только текст',
            'text_audio': 'Текст + Аудио',
            'text_video': 'Текст + Видео',
            'full_model': 'Предложенная (T+A+V)'
        }
        print(f"{model_name[name]:<30} {metrics['accuracy']:.4f}      {metrics['f1']:.4f}")

    return results, full_model



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{device}")

    with open('Al_dataset/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('Al_dataset/val_data.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open('Al_dataset/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    print(f"Обучающая выборка: {len(train_data)} записей")
    print(f"Валидационная выборка: {len(val_data)} записей")
    print(f"Тестовая выборка: {len(test_data)} записей")

    batch_size = 4

    train_dataset = SarcasmDataset(train_data)
    val_dataset = SarcasmDataset(val_data)
    test_dataset = SarcasmDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    results, best_model = compare_with_baselines(train_loader, val_loader, test_loader, device)

    # Анализ влияния видео
    #analyze_mismatch_impact(best_model, test_loader, device)

    # Сохранение лучшей модели
    torch.save(best_model.state_dict(), 'final_model.pth')
    print("\n✓ Лучшая модель сохранена в 'final_model.pth'")

    # Сохранение результатов
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()