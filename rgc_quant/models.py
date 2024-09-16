import torch
import torch.nn as nn

class CellSomaSegmentationModel(nn.Module):
    def __init__(self):
        super(CellSomaSegmentationModel, self).__init__()
        self.dropout_prob = 0.3
        self.num_features = 16
        
        # Encoder
        self.down_1 = nn.Sequential(
            # input size 1*128x128x16
            nn.Conv3d(1, self.num_features, kernel_size=(8,8,2), padding=(3,3,0), stride=(2,2,2)),
            nn.BatchNorm3d(self.num_features),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_prob),
            # output size 8*64x64x8
        )
        self.down_2 = nn.Sequential(
            # input size 8*64x64x8
            nn.Conv3d(self.num_features, 2*self.num_features, kernel_size=(8,8,3), padding=(3,3,1), stride=(2,2,1)),
            nn.BatchNorm3d(2*self.num_features),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_prob),
            # output size 16*32x32x8
        )
        self.down_3 = nn.Sequential(
            # input size 16*32x32x8
            nn.Conv3d(2*self.num_features, 4*self.num_features, kernel_size=(8,8,3), padding=(3,3,1), stride=(2,2,1)),
            nn.BatchNorm3d(4*self.num_features),
            nn.ReLU(),
            # output size 32*16x16x8
        )

        self.double_conv_1 = nn.Sequential(
            # input size 32*16x16x8
            nn.Conv3d(4*self.num_features, 4*self.num_features, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_prob),
            nn.Conv3d(4*self.num_features, 4*self.num_features, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_prob),
            # output size 32*16x16x8
        )

        # Decoder
        self.up_1 = nn.Sequential(
            # input size 64*16x16x8
            nn.ConvTranspose3d(8*self.num_features, 4*self.num_features, kernel_size=(8,8,3), padding=(3,3,1), stride=(2,2,1)),
            nn.BatchNorm3d(4*self.num_features),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_prob),
            # output size 32*32x32x8
        )
        self.up_2 = nn.Sequential(
            # input size 48*32x32x8
            nn.ConvTranspose3d(6*self.num_features, 2*self.num_features, kernel_size=(8,8,3), padding=(3,3,1), stride=(2,2,1)),
            nn.BatchNorm3d(2*self.num_features),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_prob),
            # output size 16*64x64x8
        )
        self.up_3 = nn.Sequential(
            # input size 24*64x64x8
            nn.ConvTranspose3d(3*self.num_features, 1, kernel_size=(8,8,4), padding=(3,3,1), stride=(2,2,2)),
            nn.BatchNorm3d(1),
            nn.Hardtanh(min_val=0, max_val=1),
            # output size 1*128x128x16
        )
    
    def forward(self, x):
        # x shape is 1*64x64x16
        x1 = self.down_1(x)
        # x1 shape is 8*32x32x8
        x2 = self.down_2(x1)
        # x2 shape is 16*16x16x8
        x3 = self.down_3(x2)
        # x3 shape is 32*8x8x8
        x4 = self.double_conv_1(x3)
        # x4 shape is 32*8x8x8
        x5 = torch.cat([x4, x3], dim=1)
        # x5 shape is 64*8x8x8
        x6 = self.up_1(x5)
        # x6 shape is 32*16x16x8
        x7 = torch.cat([x6, x2], dim=1)
        # x7 shape is 48*16x16x8
        x8 = self.up_2(x7)
        # x8 shape is 16*32x32x8
        x9 = torch.cat([x8, x1], dim=1)
        # x9 shape is 24*32x32x16
        x10 = self.up_3(x9)
        # x10 shape is 1*64x64x16

        return x10