"""
Code to define CLAP-related networks.
Some code inspired from here https://github.com/zhepeiw/clap_curation

Credits:
    * Francesco Paissan 2024
"""

import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, BatchEncoding

def get_model_from_str(s, vs=("alpha", "beta", "t0", "N")):
    def get_var(s, key):
        tmp = s.split("_")
        return tmp[tmp.index(key) + 1]

    verb = "PhiNet initialized with "
    ret = {}
    for k in vs:
        verb += f"{k}={get_var(s, k)} "
        ret[k] = float(get_var(s, k))

    ret["t_zero"] = ret["t0"]
    ret["num_layers"] = ret["N"]
    del ret["t0"]
    del ret["N"]

    return ret

def get_audio_encoder(name: str):
    if name == "Cnn14":
        return Cnn14
    elif "phinet" in name:
        phinet_conf = get_model_from_str(name)
        return PhiNet(input_shape=(1, 640, 64), compatibility=True, **phinet_conf)
    else:
        raise Exception(
            "The audio encoder name {} is incorrect or not supported".format(name)
        )
    
class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x

class Cnn14(nn.Module):
    def __init__(
        self,
        classes_num,
        out_emb,
    ):

        super(Cnn14, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        # out_emb is 2048 for best Cnn14
        self.fc1 = nn.Linear(2048, out_emb, bias=True)
        self.fc_audioset = nn.Linear(out_emb, classes_num, bias=True)

    def forward(self, x, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        """
        # (batch_size, 1, time_steps, mel_bins)

        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x4_out = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x4_out, p=0.2, training=self.training)
        x3_out = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x3_out, p=0.2, training=self.training)
        x2_out = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x2_out, p=0.2, training=self.training)
        x1_out = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x1_out, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {
            "clipwise_output": clipwise_output,
            "embedding": (embedding, x1_out, x2_out, x3_out, x4_out),
        }

        return output_dict
    
class AudioEncoder(nn.Module):
    def __init__(
        self,
        audioenc_name: str,
        d_in: int,
        d_out: int,
        classes_num: int,
    ) -> None:
        super().__init__()

        audio_encoder = get_audio_encoder(audioenc_name)

        if not "phinet" in audioenc_name:
            self.base = audio_encoder(
                classes_num,
                d_in,
            )
        else:
            self.base = audio_encoder

        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = (
            out_dict["embedding"][0],
            out_dict["clipwise_output"],
        )
        projected_vec = self.projection(audio_features)

        return (
            projected_vec,
            out_dict["embedding"][1:],
            audio_classification_output,
        )
        

class CLAP(nn.Module):
    def __init__(
        self,
        # audio
        audioenc_name: str,
        classes_num: int,
        out_emb: int,
        # common
        d_proj: int,
        clap_checkpoint: str,
):
        super().__init__()

        self.audio_encoder = AudioEncoder(
            audioenc_name,
            out_emb,
            d_proj,
            classes_num,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        state_dict = torch.load(clap_checkpoint)["model"]
        self.load_state_dict(self.clean_state_dict(state_dict))
        print("Loaded pretrained CLAP checkpoint.")

    @staticmethod
    def clean_state_dict(state_dict):
        """Removes pre-processing keys from the state-dict."""
        keys_to_remove = []
        for k in state_dict:
            if "spectrogram" in k or "mel" in k:
                keys_to_remove.append(k)
            if "caption" in k: 
                keys_to_remove.append(k)

        for k in keys_to_remove:
            state_dict.pop(
                k,
                None,
            )

        return state_dict

    def forward(self, audio, input_ids, token_type_ids, attention_mask, single=None):
        audio_embed = None
        caption_embed = None

        if not single == "txt":
            audio_embed, _, _ = self.audio_encoder(audio)
            audio_embed = audio_embed / audio_embed.norm(dim=1, keepdim=True)

        if not single == "aud":
            text = BatchEncoding(
                {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                }
            )
            caption_embed = self.caption_encoder(text)
            caption_embed = caption_embed / caption_embed.norm(dim=1, keepdim=True)

        return caption_embed, audio_embed
    

def load_cnn14_audio_encoder(weights_path: str | Path | None = None) -> AudioEncoder:
    """Build a Cnn14 audio encoder and load the optional CLAP checkpoint."""

    audio_encoder = AudioEncoder("Cnn14", 2048, 1024, 527)
    if weights_path is None:
        weights_path = Path(__file__).resolve().parents[2] / "weights" / "clap_encoder.pth"

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"CLAP checkpoint not found at {weights_path}")

    audio_encoder.load_state_dict(torch.load(weights_path, weights_only=False))
    return audio_encoder
