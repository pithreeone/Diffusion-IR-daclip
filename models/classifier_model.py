import torch
import torch.nn as nn
import torchvision.models as models

class DegradationClassifier(nn.Module):
    def __init__(self, num_classes, backbone='resnet18', pretrained=True, prompt_dim=128):
        super().__init__()
        
        # 1. Load ResNet backbone
        if backbone == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 2. Remove the last classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # output shape: (B, 512, 1, 1) or (B, 2048, 1, 1)

        # 3. Determine feature dim (ResNet18/34 → 512, ResNet50 → 2048)
        feat_dim = 512 if '18' in backbone or '34' in backbone else 2048

        # 4. Classification head (for predicting degradation type)
        self.classifier_head = nn.Linear(feat_dim, num_classes)

        # # 5. Optional: projection head for prompt embedding (if you want soft prompt conditioning)
        # self.prompt_proj = nn.Linear(feat_dim, prompt_dim)

    def forward(self, x):
        # Extract features
        feat = self.feature_extractor(x).squeeze(-1).squeeze(-1)  # (B, feat_dim)

        # Predict degradation label probabilities
        logits = self.classifier_head(feat)  # (B, num_classes)
        return logits

        # # Optionally, project feature to prompt space
        # prompt = self.prompt_proj(feat)  # (B, prompt_dim)

        # return logits, prompt
