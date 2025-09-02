import torch
import torch.nn as nn
# To load from a YAML file, you would use:
# import yaml
# with open("config.yaml", 'r') as f:
#     config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualBlock(nn.Module):
    """
    A residual block for the encoder, composed of Conv1d, GroupNorm, Dropout, and a skip connection.
    GroupNorm with num_groups=1 is used to simulate Layer Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_prob):
        super(ResidualBlock, self).__init__()
        
        # Calculate padding to maintain sequence length for stride=1, or halve it for stride=2
        padding = (kernel_size - 1) // 2
        
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.GroupNorm(1, out_channels), # Layer Normalization
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )
        
        # Skip connection to handle changes in dimensions
        self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) \
            if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.layers(x)
        
        if self.skip_connection:
            identity = self.skip_connection(x)
            
        return out + identity


class ResidualTransposeBlock(nn.Module):
    """
    A symmetrical residual block for the decoder, using ConvTranspose1d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_prob):
        super(ResidualTransposeBlock, self).__init__()
        
        padding = (kernel_size - 1) // 2
        output_padding = stride - 1

        self.layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.GroupNorm(1, out_channels), # Layer Normalization
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )
        
        self.skip_connection = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=output_padding) \
            if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.layers(x)

        if self.skip_connection:
            identity = self.skip_connection(x)
            
        # Ensure the output size matches for the skip connection
        if identity.shape[-1] != out.shape[-1]:
             # Pad identity if there is a mismatch from ConvTranspose1d's output shape behavior
             diff = out.shape[-1] - identity.shape[-1]
             identity = nn.functional.pad(identity, [diff // 2, diff - diff // 2])

        return out + identity


class SymmetricalAutoencoder(nn.Module):
    """
    A symmetrical convolutional autoencoder built from residual blocks.
    The architecture is defined by a configuration dictionary.
    """
    def __init__(self, config):
        super(SymmetricalAutoencoder, self).__init__()
        self.config = config
        
        # --- Build Encoder ---
        encoder_blocks = []
        blocks_config = config['blocks_config']
        for block_cfg in blocks_config:
            encoder_blocks.append(ResidualBlock(block_cfg['in'], block_cfg['out'], block_cfg['kernel'], block_cfg['stride'], config['dropout_prob']))
        self.encoder_conv = nn.Sequential(*encoder_blocks)

        # Calculate flattened size dynamically
        self._calculate_flattened_size()
        
        self.encoder_fc = nn.Linear(self.flattened_size, config['latent_dim'])

        # --- Build Decoder ---
        self.decoder_fc = nn.Linear(config['latent_dim'], self.flattened_size)

        decoder_blocks = []
        # Create decoder blocks in reverse order of encoder
        for block_cfg in reversed(blocks_config):
            decoder_blocks.append(ResidualTransposeBlock(block_cfg['out'], block_cfg['in'], block_cfg['kernel'], block_cfg['stride'], config['dropout_prob']))
        self.decoder_conv = nn.Sequential(*decoder_blocks)

    def _calculate_flattened_size(self):
        """Passes a dummy tensor through the conv encoder to determine the flattened size."""
        dummy_input = torch.randn(1, self.config['input_channels'], self.config['input_seq_len'])
        dummy_output = self.encoder_conv(dummy_input)
        self.flattened_size = int(torch.numel(dummy_output) / dummy_output.shape[0])
        self.unflatten_shape = dummy_output.shape[1:]

    def encode(self, x):
        # Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1) # Flatten
        latent = self.encoder_fc(x)
        return latent

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), *self.unflatten_shape) # Unflatten
        x = self.decoder_conv(x)
        # Permute back to (batch, length, channels)
        reconstructed = x.permute(0, 2, 1)
        return reconstructed

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)


# --- Example Usage ---
if __name__ == '__main__':
    config = {
        'input_channels': 3,
        'input_seq_len': 256,
        'latent_dim': 32,
        'dropout_prob': 0.1,
        'blocks_config': [
            # Encoder blocks: (in_channels, out_channels, kernel_size, stride)
            {'in': 3,  'out': 16, 'kernel': 7, 'stride': 2},
            {'in': 16, 'out': 32, 'kernel': 5, 'stride': 2},
            {'in': 32, 'out': 64, 'kernel': 3, 'stride': 2},
        ]
    }

    model = SymmetricalAutoencoder(config).to(device)
    print("--- Symmetrical Autoencoder Architecture ---")
    print(model)
    print("-" * 60)

    # --- Test with a dummy tensor ---
    DUMMY_BATCH_SIZE = 32
    dummy_input = torch.randn(DUMMY_BATCH_SIZE, config['input_seq_len'], config['input_channels']).to(device)
    
    print(f"Input shape:  {dummy_input.shape}")
    reconstructed_output = model(dummy_input)
    print(f"Output shape: {reconstructed_output.shape}")

    assert dummy_input.shape == reconstructed_output.shape
    print("\nSuccessfully tested the autoencoder. Input and output shapes match.")

    encoded = model.encode(dummy_input)
    print(f"Encoded shape: {encoded.shape}")
