import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import BasicBlock

class ResNetVisualWajah(nn.Module):
    def __init__(self, freeze=True):
        super(ResNetVisualWajah, self).__init__()
        resnet = models.resnet18(pretrained=True)
        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.extra_block = nn.Sequential(
            BasicBlock(
                inplanes=512,
                planes=1024,
                stride=2,
                downsample=nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(1024)
                )
            ),
            BasicBlock(
                inplanes=1024,
                planes=1024,
                stride=1
            )
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.shape
        x = x.view(batch_size * time_steps, C, H, W)
        x = self.feature_extractor(x)
        x = self.extra_block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x.view(batch_size, time_steps, -1)

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):  # x: (B, T, F)
        weights = self.attn(x).squeeze(-1)  # (B, T)
        weights = torch.softmax(weights, dim=1)  # (B, T)
        attended = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, F)
        return attended

class BiLSTMVisualWajah(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, num_layers=2, dropout=0.3):
        super(BiLSTMVisualWajah, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                             bidirectional=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers=num_layers, batch_first=True,
                             bidirectional=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_size * 2, hidden_size,num_layers=num_layers, batch_first=True,
                             bidirectional=True, dropout=dropout)
        
        self.attention = AttentionLayer(hidden_size * 2)
        self.output_size = hidden_size * 2

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        attended = self.attention(x)
        return attended

class DeepfakeClassifierVisualWajah(nn.Module):
    def __init__(self, num_classes=2, freeze_resnet=True):
        super(DeepfakeClassifierVisualWajah, self).__init__()
        self.visual_extractor = ResNetVisualWajah(freeze=freeze_resnet)
        self.bilstm = BiLSTMVisualWajah()
        input_dim = self.bilstm.output_size

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, frames):
        visual_feat_seq = self.visual_extractor(frames)
        visual_embed = self.bilstm(visual_feat_seq)
        out = self.classifier(visual_embed)
        return out

class CNNBiLSTMVisualBibirBranch(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.alexnet(pretrained=False)
        self.cnn = nn.Sequential(*list(base.features.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.cnn(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = x.view(B, T, -1)
        lstm_out, _ = self.lstm(x)
        return lstm_out
    
class CNNBiLSTMAudioBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=256,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(2).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return lstm_out
    
class DeepfakeClassifierVisualBibirAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = CNNBiLSTMVisualBibirBranch()
        self.audio = CNNBiLSTMAudioBranch()
        self.temporal_fusion1 = nn.LSTM(input_size=512+512, hidden_size=256,
                                       num_layers=1, batch_first=True, bidirectional=True)
        self.temporal_fusion2 = nn.LSTM(input_size=512, hidden_size=256,
                                       num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, frames, audio):
        vis_feat = self.visual(frames)
        aud_feat = self.audio(audio)

        min_T = min(vis_feat.size(1), aud_feat.size(1))
        vis_feat = vis_feat[:, :min_T, :]
        aud_feat = aud_feat[:, :min_T, :]

        fused = torch.cat([vis_feat, aud_feat], dim=2)
        fused_out1, _ = self.temporal_fusion1(fused)
        fused_out2, _ = self.temporal_fusion2(fused_out1)
        pooled = fused_out2.mean(dim=1)
        return self.classifier(pooled)
    
def load_visual_models():
    classifierWajah = DeepfakeClassifierVisualWajah()
    classifierWajah.load_state_dict(torch.load("mlapp/ml_model/ResNet18_1024-BiLSTM_3_Wajah_2R1F.pth", map_location='cpu'))
    classifierWajah.eval()

    return classifierWajah

def load_lipsync_models():
    classifierBibirAudio = DeepfakeClassifierVisualBibirAudio()
    classifierBibirAudio.load_state_dict(torch.load("mlapp/ml_model/5_Conv_AlexNet-Bibir_256Custom-Audio_2-BiLSTM_NotPretrained.pth", map_location='cpu'))
    classifierBibirAudio.eval()
    
    return classifierBibirAudio