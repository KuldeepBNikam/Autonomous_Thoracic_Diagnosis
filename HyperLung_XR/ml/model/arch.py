import torch
import torch.nn as nn
import timm


class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()

        # 🔹 CNN Backbone (DenseNet – excellent for X-rays)
        self.cnn = timm.create_model(
            "densenet121",
            pretrained=True,
            features_only=True
        )
        cnn_out_channels = self.cnn.feature_info[-1]["num_chs"]

        # 🔹 Swin Transformer
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )
        swin_out_features = self.swin.num_features

        # 🔹 Feature fusion
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_channels + swin_out_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # CNN path
        cnn_feats = self.cnn(x)[-1]
        cnn_feats = self.pool(cnn_feats).flatten(1)

        # Swin path
        swin_feats = self.swin(x)

        # Fusion
        fused = torch.cat([cnn_feats, swin_feats], dim=1)

        return self.classifier(fused)
