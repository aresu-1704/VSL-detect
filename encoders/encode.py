import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from encoders.BiLSTM_encoder import VSL_BiLSTM_Encoder
from retrieval.retrieval_preprocess import extract_sequence_from_video


class VSLEncoder:
    def __init__(self, ckpt_path, device="cpu"):
        self.device = torch.device(device)

        self.model = VSL_BiLSTM_Encoder(
            input_dim=201,
            hidden_dim=256,
            num_layers=2,
            emb_dim=256,
            dropout=0.2
        )

        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device)
        )
        self.freeze_encoder()

        self.model.to(self.device)
        self.model.eval()

    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, video_path):
        """
        return: np.ndarray (256,)
        """

        # (60, 201)
        sequence = extract_sequence_from_video(video_path)
        if sequence is None:
            raise RuntimeError("Cannot extract sequence")

        # (1, 60, 201)
        x = torch.tensor(sequence, dtype=torch.float32)
        x = x.unsqueeze(0).to(self.device)

        # (1, 256)
        emb = self.model(x)

        return emb[0].cpu().numpy()