import torch
import torch.nn as nn
import einops
import torch.nn.functional as F
class RSFeatureAggregator(nn.Module):
    in_channels_dict = {
        'base': [768] * (12+1),
        'large': [1024] * (24+1),
        'huge': [1280] * (32+1),
    }

    def __init__(
            self,
            in_channels="large",
            hidden_channels=64,
            out_channels=256,
            select_layers=range(1, 24+1, 2),
    ):
        super().__init__()
        assert isinstance(in_channels, str)
        model_arch = "large"
        self.in_channels = self.in_channels_dict[model_arch]
        self.select_layers = select_layers

        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            self.downconvs.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i_layer], hidden_channels, 1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.hidden_convs = nn.ModuleList()
        for _ in self.select_layers:
            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, inputs):
        #inputs :25*(B,H,W,C) 25layers features
        assert len(inputs) == len(self.in_channels)
        inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in inputs]

        features = []
        for idx, i_layer in enumerate(self.select_layers):
            features.append(self.downconvs[idx](inputs[i_layer]))

        x = None
        for hidden_state, hidden_conv in zip(features, self.hidden_convs):
            if x is not None:
                hidden_state = x + hidden_state
            residual = hidden_conv(hidden_state)
            x = hidden_state + residual
        x = self.fusion_conv(x)
        return x
    
class LLAMANECK(nn.Module):
    def __init__(self, 
                input_channels=1536, ##1.5B=1536  3B=2048  0.5B=896  7B=3584
                out_channels=256,
                num_seq = 4
                ):
        super().__init__()
        self.in_channels = input_channels
        self.out_channels = out_channels
        self.num_seq = num_seq
        self.global_avg_pool = nn.AdaptiveAvgPool1d(self.num_seq)
        self.text_mlp = nn.Sequential(
            nn.Linear(self.in_channels,self.out_channels*8),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels*8,self.out_channels*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels*2,self.out_channels),
        )
    # def forward(self,x):
    #     # format_length = []
    #     # for text in x:
    #     #     x = self.global_avg_pool(text.unsqueeze(dim=0).permute(0,2,1)).permute(0,2,1)
    #     #     format_length.append(x)
    #     # x = torch.cat(format_length,dim=0)
    #     return (self.text_mlp(x[0])).unsqueeze(dim=0)    
    def forward(self,x):
        format_length = []
        for text in x:
            x = self.global_avg_pool(text.unsqueeze(dim=0).permute(0,2,1)).permute(0,2,1)
            format_length.append(x)
        x = torch.cat(format_length,dim=0)
        return self.text_mlp(x) 