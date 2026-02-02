import torch  #main torch library
import torch.nn as nn # Hepls us build nn
import timm  #Library of pretrained models


class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()   #sets up pytorch internals

        #  CNN Backbone (DenseNet – excellent for X-rays)
        self.cnn = timm.create_model(
            "densenet121",  # CNN-Excellent for medicle images
            pretrained=True,
            features_only=True
        )
        cnn_out_channels = self.cnn.feature_info[-1]["num_chs"] # no of channels of last feature map

        #  Swin Transformer
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )
        swin_out_features = self.swin.num_features

        #  Freeze CNN backbone
        for param in self.cnn.parameters():
            param.requires_grad = False

        #  Freeze Swin Transformer backbone
        for param in self.swin.parameters():
            param.requires_grad = False

        #  Feature fusion
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_channels + swin_out_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def unfreeze_top_layers(self):
        """
        Unfreeze high-level layers for fine-tuning
        """

        #  Unfreeze last DenseNet block
        for name, param in self.cnn.named_parameters():
            if "denseblock4" in name or "norm5" in name:
                param.requires_grad = True

        #  Unfreeze last Swin stage
        for name, param in self.swin.named_parameters():
            if "layers.3" in name:  # last Swin stage
                param.requires_grad = True


    def forward(self, x):
        # CNN feature maps (DO NOT pool yet)
        cnn_feature_maps = self.cnn(x)
        last_conv_map = cnn_feature_maps[-1]   # (B, C, H, W)

        # Pool ONLY for classifier
        cnn_feats = self.pool(last_conv_map).flatten(1)

        swin_feats = self.swin(x)

        fused = torch.cat([cnn_feats, swin_feats], dim=1)
        logits = self.classifier(fused)

        return logits, last_conv_map

