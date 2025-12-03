import torch
from torch import nn
from .vae_input_modle import VAEInputModel

from ...utils.augmentations import GaussianSmoothing


class GRUDecoderVAE(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        hidden_dim_vae,
        layer_dim,
        latent_global,
        latent_local,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoderVAE, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_vae = hidden_dim_vae
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.latent_global = latent_global
        self.latent_local = latent_local
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        if self.gaussianSmoothWidth > 0:
            self.gaussianSmoother = GaussianSmoothing(
                neural_dim, 20, self.gaussianSmoothWidth, dim=1
            )
        else:
            self.gaussianSmoother = torch.nn.Identity()
        # apply vae dim reduction
        # self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        # self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        latent_dim = latent_local + latent_global
        self.inputAE = VAEInputModel(input_dim=neural_dim, latent_global=latent_global, latent_local= latent_local, hidden_dim=hidden_dim_vae)
        self.day_scale = nn.Parameter(torch.ones(nDays, latent_dim))
        self.day_shift = nn.Parameter(torch.zeros(nDays, latent_dim)) 
        # for x in range(nDays):
        #     self.dayWeights.data[x, :, :] = torch.eye(neural_dim)


        # Layer normalization 
        norm_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.layerNorm = nn.LayerNorm(norm_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (latent_dim) * self.kernelLen,
            #(neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        self.day_embedding = nn.Embedding(self.nDays, neural_dim).to(device)

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # # Input layers
        # for x in range(nDays):
        #     setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        # for x in range(nDays):
        #     thisLayer = getattr(self, "inpLayer" + str(x))
        #     thisLayer.weight = torch.nn.Parameter(
        #         thisLayer.weight + torch.eye(neural_dim)
        #     )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        device = neuralInput.device
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        
        batch, time, channels = neuralInput.shape

        day_embed = self.day_embedding(dayIdx) #shape (batch, D_day)
        D_day = day_embed.shape[-1] # shape (batch, D_day)
        day_embed = day_embed.unsqueeze(1).expand(batch, time, D_day) 
        neuralInput = torch.cat([neuralInput, day_embed], dim=2) #shape (batch, time, channels + D_day)
        flat = neuralInput.reshape(batch * time, 2*channels) #s
        # Pooled per-sample features for the global encoder: (B, channels)
        global_x = neuralInput.mean(dim=1)

        # Encode
        mean_local, log_variance_local = self.inputAE.encode_local(flat)     # (B*T, latent_local)
        mean_global, log_variance_global = self.inputAE.encode_global(global_x) # (B, latent_global)

        # Reparameterize local latents; ensure flattened shape (B*T, latent_local)
        z_local = self.inputAE.reparameterize(mean_local, log_variance_local, self.training)  # [B*T, latent_local]
        z = z_local
        if mean_global is not None:
            # global z: (B, latent_global) -> expand to (B*T, latent_global)
            z_global = self.inputAE.reparameterize(mean_global, log_variance_global, self.training)
            z_global_exp = z_global.unsqueeze(1).expand(batch, time, -1).reshape(batch * time, -1)
            z = torch.cat([z_local, z_global_exp], dim=1)    # B*T x (latent_local + latent_global)
        dayIdx_exp = dayIdx.unsqueeze(1).repeat(1, time).reshape(-1)  # B*T
        scale = self.day_scale[dayIdx_exp] # B*T x (latent_local + latent_global)
        shift = self.day_shift[dayIdx_exp] # B*T x (latent_local + latent_global)
        z = z * scale + shift  # 
        z = z.reshape(batch, time, self.latent_global+self.latent_local)  # B x T x (latent_local + latent_global)
        recon_flat = self.inputAE.decode(z)#
        transformedNeural = recon_flat.view(batch, time, 2*channels)#
        recon = transformedNeural[ :, :, :channels]  
        self.inputLayerNonlinearity(transformedNeural)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(z, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # normalize
        hid = self.layerNorm(hid)

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out, recon, mean_local, log_variance_local, mean_global, log_variance_global
