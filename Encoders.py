import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from torchvision.models.video import r3d_18, R3D_18_Weights
import numpy as np

class TextEncoder(nn.Module):

    def __init__(self, pretrained_model='DeepPavlov/rubert-base-cased',
                 output_dim=768, freeze_bert=True):
        super(TextEncoder, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_model)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.output_layer = nn.Linear(768, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Применение выходного слоя
        output = self.dropout(cls_embedding)
        output = self.output_layer(output)

        return output

class AudioEncoder(nn.Module):
    def __init__(self, pretrained_model='facebook/wav2vec2-base',
                 output_dim=768, freeze_wav2vec=True):
        super(AudioEncoder, self).__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model)

        if freeze_wav2vec:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

        self.output_layer = nn.Linear(768, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_values):

        outputs = self.wav2vec(input_values=input_values)

        audio_embedding = outputs.last_hidden_state.mean(dim=1)  # [batch_size, 768]

        output = self.dropout(audio_embedding)
        output = self.output_layer(output)  # [batch_size, output_dim]

        return output

class SlowFastEncoder(nn.Module):
    def __init__(self, output_dim=2304, freeze_backbone=True):
        super(SlowFastEncoder, self).__init__()
        self.slow_backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)

        self.fast_backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.slow_backbone.parameters():
                param.requires_grad = False
            for param in self.fast_backbone.parameters():
                param.requires_grad = False

        self.slow_backbone.fc = nn.Identity()
        self.fast_backbone.fc = nn.Identity()

        self.slow_output = nn.Linear(512, 2048)
        self.fast_output = nn.Linear(512, 256)

        self.output_layer = nn.Linear(2304, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, video_frames_slow, video_frames_fast):

        slow_features = self.slow_backbone(video_frames_slow)  # [batch_size, 512]
        slow_output = self.slow_output(slow_features)  # [batch_size, 2048]

        fast_features = self.fast_backbone(video_frames_fast)  # [batch_size, 512]
        fast_output = self.fast_output(fast_features)  # [batch_size, 256]

        combined = torch.cat([slow_output, fast_output], dim=1)  # [batch_size, 2304]

        output = self.dropout(combined)
        output = self.output_layer(output)  # [batch_size, output_dim]

        return output


class AttentionBottleneck(nn.Module):

    def __init__(self, dim=768, num_bottlenecks=4, num_heads=4):
        super(AttentionBottleneck, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.bottlenecks = nn.Parameter(torch.randn(num_bottlenecks, dim))

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

    def __init__(self, num_classes=6, freeze_encoders=True):
        super(MultimodalSarcasmModel, self).__init__()

        self.text_encoder = TextEncoder(freeze_bert=freeze_encoders)
        self.audio_encoder = AudioEncoder(freeze_wav2vec=freeze_encoders)
        self.video_encoder = SlowFastEncoder(freeze_backbone=freeze_encoders)

        self.text_audio_bottleneck = AttentionBottleneck(dim=768, num_bottlenecks=4)
        self.text_video_bottleneck = AttentionBottleneck(dim=768, num_bottlenecks=4)
        self.audio_video_bottleneck = AttentionBottleneck(dim=768, num_bottlenecks=4)

        self.fusion_layer = nn.Sequential(
            nn.Linear(768 * 3, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, text_input_ids, text_attention_mask,
                audio_input_values,
                video_frames_slow, video_frames_fast):

        text_emb = self.text_encoder(text_input_ids, text_attention_mask)  # [batch_size, 768]
        audio_emb = self.audio_encoder(audio_input_values)  # [batch_size, 768]
        video_emb = self.video_encoder(video_frames_slow, video_frames_fast)  # [batch_size, 2304]

        video_emb_reduced = video_emb[:, :768]

        text_emb_expanded = text_emb.unsqueeze(1)
        audio_emb_expanded = audio_emb.unsqueeze(1)
        video_emb_expanded = video_emb_reduced.unsqueeze(1)

        ta_input = torch.cat([text_emb_expanded, audio_emb_expanded], dim=1)  # [batch_size, 2, 768]
        ta_fused = self.text_audio_bottleneck(ta_input)  # [batch_size, 768]

        tv_input = torch.cat([text_emb_expanded, video_emb_expanded], dim=1)  # [batch_size, 2, 768]
        tv_fused = self.text_video_bottleneck(tv_input)  # [batch_size, 768]

        av_input = torch.cat([audio_emb_expanded, video_emb_expanded], dim=1)  # [batch_size, 2, 768]
        av_fused = self.audio_video_bottleneck(av_input)  # [batch_size, 768]

        combined = torch.cat([ta_fused, tv_fused, av_fused], dim=1)  # [batch_size, 2304]

        output = self.fusion_layer(combined)  # [batch_size, num_classes]

        return output

def prepare_text_input(text, tokenizer, max_length=512):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_attention_mask=True
    )
    return inputs['input_ids'], inputs['attention_mask']


def prepare_audio_input(audio_array, processor, sampling_rate=16000):
    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors='pt'
    )
    return inputs['input_values']


def prepare_video_input(video_frames, num_slow_frames=4, num_fast_frames=32):
    video_tensor = torch.tensor(video_frames, dtype=torch.float32)

    video_tensor = video_tensor / 255.0
    video_tensor = (video_tensor - 0.45) / 0.225

    slow_indices = np.linspace(0, len(video_frames) - 1, num_slow_frames, dtype=int)
    video_frames_slow = video_tensor[slow_indices]

    fast_indices = np.linspace(0, len(video_frames) - 1, num_fast_frames, dtype=int)
    video_frames_fast = video_tensor[fast_indices]

    video_frames_slow = video_frames_slow.unsqueeze(0).permute(0, 4, 1, 2, 3)
    video_frames_fast = video_frames_fast.unsqueeze(0).permute(0, 4, 1, 2, 3)

    return video_frames_slow, video_frames_fast

def main():
    model = MultimodalSarcasmModel(num_classes=6, freeze_encoders=True)
    print("Модель создана")

    text_tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    print("Токенизаторы и процессоры загружены")

    sample_text = "Ну конечно, именно этого мне сегодня не хватало..."
    sample_audio = np.random.randn(16000).astype(np.float32)  # 1 секунда аудио
    sample_video_frames = np.random.randint(0, 255, (64, 224, 224, 3), dtype=np.uint8)

    text_ids, text_mask = prepare_text_input(sample_text, text_tokenizer)
    audio_values = prepare_audio_input(sample_audio, audio_processor)
    video_slow, video_fast = prepare_video_input(sample_video_frames)

    print("Входные данные подготовлены")

    with torch.no_grad():
        output = model(text_ids, text_mask, audio_values, video_slow, video_fast)
        probabilities = F.softmax(output, dim=1)

    print("Модель выполнена")

    print("\nРезультаты предсказания:")
    class_names = ['саркастически+', 'саркастически–', 'ирония',
                   'вежливая критика', 'скрытое одобрение', 'отсутствие несоответствия']

    for i, prob in enumerate(probabilities[0]):
        print(f"{class_names[i]:<25}: {prob.item():.4f}")

    predicted_class = torch.argmax(probabilities, dim=1).item()
    print(f"\nПредсказанный класс: {class_names[predicted_class]}")
    print(f"Вероятность: {probabilities[0][predicted_class].item():.4f}")

if __name__ == "__main__":
    main()