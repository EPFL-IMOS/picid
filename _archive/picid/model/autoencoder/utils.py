import numpy as np
from typing import List, Tuple


def get_autoencoder_summary_and_tags(
    config: dict,
) -> Tuple[str, List[str]]:
    """
    Analyzes a symmetrical autoencoder config and returns a summary string
    and a sorted list of descriptive tags.

    Args:
        config: A dictionary containing the model architecture.

    Returns:
        A tuple containing:
        - A formatted, multi-line string with the model summary.
        - A sorted list of descriptive string tags for the model.
    """
    # --- 1. Extract and Validate Parameters ---
    model_params = config
    kernels = model_params.get("kernels", [])

    if not kernels:
        return "No convolutional layers found in config.", []

    # --- 2. Calculate Encoder Convolutional Properties ---
    num_layers = len(kernels)
    strides = model_params.get("strides", [1] * num_layers)
    dilations = model_params.get("dilations", [1] * num_layers)
    output_channels = model_params.get("output_channels", [])
    input_channels = model_params.get("input_channels", 1)
    input_seq_len = model_params.get("input_seq_len", 256)
    latent_dim = model_params.get("latent_dim", 32)

    # --- MODIFICATION START ---
    # Check if input_channels is a valid integer for parameter calculation
    params_calculable = isinstance(input_channels, int)
    # --- MODIFICATION END ---

    receptive_field, total_stride = 1, 1
    output_size = input_seq_len

    # Initialize parameter variables
    encoder_conv_params = 0
    total_params = 0

    # This loop calculates receptive field and output size regardless of parameter calculability.
    # The parameter calculation part is now inside the conditional block below.
    _current_in_channels = (
        input_channels if params_calculable else 1
    )  # Use a dummy value if not calculable
    for i in range(num_layers):
        k, s, d = kernels[i], strides[i], dilations[i]
        p = (k - 1) * d // 2

        receptive_field += (k - 1) * d * total_stride
        total_stride *= s
        output_size = np.floor((output_size + 2 * p - d * (k - 1) - 1) / s) + 1

    # --- 3. Calculate All Layer Properties (only if possible) ---
    if params_calculable:
        current_in_channels = input_channels
        for i in range(num_layers):
            k = kernels[i]
            cin, cout = current_in_channels, output_channels[i]

            # A simplified approximation of Residual Block parameters
            conv_block_params = ((cin * cout * k) + cout) + (cout * 2)
            residual_params = (
                ((cin * cout * 1) + cout) + (cout * 2) if cin != cout else 0
            )
            encoder_conv_params += conv_block_params + residual_params
            current_in_channels = cout

        flattened_size = int(output_size) * output_channels[-1]
        encoder_fc_params = (flattened_size * latent_dim) + latent_dim
        decoder_fc_params = (latent_dim * flattened_size) + flattened_size
        decoder_conv_params = encoder_conv_params

        total_params = (
            encoder_conv_params
            + decoder_conv_params
            + encoder_fc_params
            + decoder_fc_params
        )

    # --- 4. Generate Tags ---
    tags = {
        f"L:{num_layers*2}",
        f"RF:{receptive_field}",
        f"S:{total_stride}",
        f"latent:{latent_dim}",
    }

    if params_calculable:
        if total_params < 50_000:
            tags.add("params:<50k")
        elif total_params < 250_000:
            tags.add("params:50k-250k")
        else:
            tags.add("params:>250k")
    else:
        tags.add("params:N/A")  # Add a tag indicating params are not available

    # --- 5. Build the Summary String ---

    # Conditionally format the total_params string
    total_params_str = f"{total_params:,}" if params_calculable else "N/A"
    summary_lines = [
        " 🤖 Symmetrical Autoencoder Summary 🤖 ",
        "-" * 40,
        f"Conv Layers (Encoder/Decoder):  {num_layers} / {num_layers}",
        f"Latent Dimension:               {latent_dim}",
        f"Total Parameters:               {total_params_str}",
        "-" * 40,
        f"Receptive Field:                {receptive_field} steps",
        f"Total Stride:                   {total_stride}x",
        "-" * 40,
        "Example Shape Transform:",
        f"Input: {input_seq_len} -> Encoded: {int(output_size)} -> Output: {input_seq_len}",
        "-" * 40,
    ]

    summary_string = "\n".join(summary_lines)

    return summary_string, sorted(list(tags))


if __name__ == "__main__":
    model_config = {
        "input_channels": 3,
        "input_seq_len": 256,
        "latent_dim": 32,
        "dropout_prob": 0.2,
        "output_channels": [16, 32, 64],
        "kernels": [5, 5, 5],
        "strides": [2, 2, 2],
        "dilations": [1, 1, 1],
    }

    summary_table, model_tags_list = get_autoencoder_summary_and_tags(model_config)

    print("--- 📋 Full Autoencoder Summary ---")
    print(summary_table)
    print("\n--- 🏷️  Generated Tags (as a List) ---")
    print(model_tags_list)
