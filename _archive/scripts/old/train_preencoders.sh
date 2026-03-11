#!/bin/bash
max_epochs=5
check_val_every_n_epoch=1
device=1
datasets=(unibo pronostia nb14 xjtu_sy)
BATCH_SIZES=(1024)

# ========================
# ✅ The long architecture strings are now GONE.
# We just define the names of the YAML files.
# ========================
ARCHS_FOR_SEQ_LEN_64=(
  "five_layers_rf_65"
  "five_layers_rf_67"
  "ten_layers_rf_64"
)

# The path is now simple and correct.
ARCH_FILE="configs/model/residual_autoencoder/block_configs/five_layers_rf_65.yaml"

# ✅ This line is correct and will now work.
ARCH_JSON=$(yq -o=json -c . "${ARCH_FILE}")
echo "Loaded architecture: ${ARCH_JSON}"

# Define ARCHS_FOR_SEQ_LEN_128=(...) for other sequence lengths if needed

# ========================
# Define valid seq_lens per dataset
# ========================
declare -A DATASET_SEQ_MAP
DATASET_SEQ_MAP["unibo"]="64 128 256"
DATASET_SEQ_MAP["nb14"]="64 128" # Example with multiple lengths

# ========================
# Loop
# ========================
for dataset in "${!DATASET_SEQ_MAP[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    for seq in ${DATASET_SEQ_MAP[$dataset]}; do

      # Determine which array of architecture NAMES to use
      declare -n arch_name_list="ARCHS_FOR_SEQ_LEN_${seq}"

      # Loop through the architecture names
      for arch_name in "${arch_name_list[@]}"; do
        echo "--- Running $dataset with seq_len=$seq, arch=$arch_name ---"
        python picid/run.py \
          experiment=${dataset}/reconstruction/residual_autoencoder \
          trainer.max_epochs=${max_epochs} \
          trainer.check_val_every_n_epoch=${check_val_every_n_epoch} \
          datamodule.train_batch_size=${batch_size} \
          datamodule.val_batch_size=${batch_size} \
          datamodule.test_batch_size=${batch_size} \
          task_definition.seq_len=$seq \
          trainer.devices=[${device}] \
          cache.use_cache_after_loading=True \
          cache.use_cache_after_transfroms=True \
          model.block_configs=${arch_name}

      done
    done
  done
done

# #!/bin/bash
# max_epochs=5
# check_val_every_n_epoch=1
# device=1
# BATCH_SIZES=(1024)


# # ========================
# # NEW: Map dataset names to their input channel count
# # ⚠️ IMPORTANT: You must replace these numbers with the correct values for your datasets.
# # ========================
# declare -A DATASET_INPUT_CHANNELS
# DATASET_INPUT_CHANNELS["unibo"]=2
# DATASET_INPUT_CHANNELS["pronostia"]=2
# DATASET_INPUT_CHANNELS["nb14"]=17
# DATASET_INPUT_CHANNELS["xjtu_sy"]=2

# # ========================
# # Define architecture strings
# # ========================
# FIVE_LAYER_STRIDE_RF_67='[{"in": __INPUT_CHANNELS__, "out": 16, "kernel": 7, "stride": 2, "dilation": 1, "padding": "same"}, {"in": 16, "out": 32, "kernel": 7, "stride": 2, "dilation": 1, "padding": "same"}, {"in": 32, "out": 64, "kernel": 5, "stride": 1, "dilation": 1, "padding": "same"}, {"in": 64, "out": 64, "kernel": 5, "stride": 1, "dilation": 1, "padding": "same"}, {"in": 64, "out": 64, "kernel": 5, "stride": 1, "dilation": 1, "padding": "same"}]'
# FIVE_LAYER_DILATED_RF_65='[{"in": __INPUT_CHANNELS__, "out": 16, "kernel": 5, "stride": 1, "dilation": 1, "padding": "same"}, {"in": 16, "out": 32, "kernel": 5, "stride": 1, "dilation": 2, "padding": "same"}, {"in": 32, "out": 64, "kernel": 5, "stride": 1, "dilation": 4, "padding": "same"}, {"in": 64, "out": 64, "kernel": 5, "stride": 1, "dilation": 8, "padding": "same"}, {"in": 64, "out": 64, "kernel": 5, "stride": 1, "dilation": 1, "padding": "same"}]'
# TEN_LAYER_STRIDE_RF_64='[{"in": __INPUT_CHANNELS__, "out": 16, "kernel": 10, "stride": 1, "dilation": 1, "padding": "same"}, {"in": 16, "out": 32, "kernel": 7, "stride": 1, "dilation": 1, "padding": "same"}, {"in": 32, "out": 64, "kernel": 7, "stride": 1, "dilation": 1, "padding": "same"}, {"in": 64, "out": 64, "kernel": 7, "stride": 1, "dilation": 1, "padding": "same"}, {"in": 64, "out": 64, "kernel": 7, "stride": 1, "dilation": 1, "padding": "same"}, {"in": ${model.input_channels}, "out": 16, "kernel": 7, "stride": 1, "dilation": 1, "padding": "same"}, {"in": 16, "out": 32, "kernel": 7, "stride": 1, "dilation": 1, "padding": "same"}, {"in": 32, "out": 64, "kernel": 7, "stride": 1, "dilation": 1, "padding": "same"}, {"in": 64, "out": 64, "kernel": 7, "stride": 1, "dilation": 1, "padding": "same"}, {"in": 64, "out": 64, "kernel": 7, "stride": 1, "dilation": 1, "padding": "same"}]'
# # Define architectures for other seq_lens here as needed
# # ARCH_SHALLOW_VAL="..."
# # ARCH_DEEP_VAL="..."
# # ARCH_WIDE_VAL="..."

# # ========================
# # FIX 1: Store architectures for each seq_len in separate arrays.
# # ========================
# ARCHS_64=(
#   "$FIVE_LAYER_STRIDE_RF_67"
#   "$FIVE_LAYER_DILATED_RF_65"
#   "$TEN_LAYER_STRIDE_RF_64"
# )
# # ARCHS_128=("$ARCH_SHALLOW_VAL" "$ARCH_DEEP_VAL")
# # ARCHS_256=("$ARCH_WIDE_VAL")

# # ========================
# # Define valid seq_lens per dataset
# # ========================
# declare -A DATASET_SEQ_MAP
# DATASET_SEQ_MAP["unibo"]="64"
# DATASET_SEQ_MAP["nb14"]="64" # Simplified for example
# # DATASET_SEQ_MAP["nb14"]="64 128 256"
# # DATASET_SEQ_MAP["xjtu_sy"]="128 256"
# # DATASET_SEQ_MAP["pronostia"]="64 128"

# # ========================
# # Loop
# # ========================
# for dataset in "${!DATASET_SEQ_MAP[@]}"; do
#   for batch_size in "${BATCH_SIZES[@]}"; do
#     for seq in ${DATASET_SEQ_MAP[$dataset]}; do

#       # FIX 2: Create a temporary variable to hold the correct array name.
#       declare -n arch_list="ARCHS_${seq}" # nameref to the correct array

#       # FIX 3: Loop through the array properly.
#       for arch in "${arch_list[@]}"; do

#         # --- MODIFIED Section ---
#         # 1. Get the number of input channels for the current dataset
#         in_channels=${DATASET_INPUT_CHANNELS[$dataset]}

#         # 2. Replace the placeholder with the actual number
#         final_arch_string="${arch//__INPUT_CHANNELS__/$in_channels}"



#         echo "--- Running $dataset with seq_len=$seq ---"
#         python picid/run.py \
#           experiment=${dataset}/reconstruction/residual_autoencoder \
#           trainer.max_epochs=${max_epochs} \
#           trainer.check_val_every_n_epoch=${check_val_every_n_epoch} \
#           datamodule.train_batch_size=${batch_size} \
#           datamodule.val_batch_size=${batch_size} \
#           datamodule.test_batch_size=${batch_size} \
#           task_definition.seq_len=$seq \
#           trainer.devices=[${device}] \
#           cache.use_cache_after_loading=True \
#           cache.use_cache_after_transfroms=True \
#           model.blocks_config="${final_arch_string}"
#       done
#     done
#   done
# done
# # Loop through each dataset in the list
# for dataset in "${datasets[@]}"; do
# for batch_size in "${BATCH_SIZES[@]}"; do
# for i in "${!ARCHITECTURES[@]}"; do
#   echo "--- Running experiment for dataset: ${dataset} ---"
#   # Get the full configuration string for the current iteration
#   arch_config="${ARCHITECTURES[$i]}"
# # DONT FORGET LATER: paths=rt_local \
#   python picid/run.py \
#     experiment=${dataset}/reconstruction/residual_autoencoder \
#     trainer.max_epochs=${max_epochs} \
#     trainer.check_val_every_n_epoch=${check_val_every_n_epoch} \
#     datamodule.train_batch_size=${batch_size} \
#     datamodule.val_batch_size=${batch_size} \
#     datamodule.test_batch_size=${batch_size} \
#     trainer.devices=[1] \
#     model.blocks_config=${arch_config_val}
# done
# done
# done
echo "--- All experiments finished. ---"





ARCH_SHALLOW_VAL='[{"in": ${model.input_channels}, "out": 32, "kernel": 7, "stride": 2}, {"in": 32, "out": 64, "kernel": 5, "stride": 2}]'
ARCH_DEEP_VAL='[{"in": ${model.input_channels}, "out": 16, "kernel": 7, "stride": 2}, {"in": 16, "out": 32, "kernel": 5, "stride": 2}, {"in": 32, "out": 64, "kernel": 3, "stride": 2}, {"in": 64, "out": 128, "kernel": 3, "stride": 2}]'
ARCH_WIDE_VAL='[{"in": ${model.input_channels}, "out": 64, "kernel": 5, "stride": 2}, {"in": 64, "out": 256, "kernel": 3, "stride": 2}]'

# Create an array of these new value-only strings
ARCHITECTURES=(
  "$ARCH_SHALLOW_VAL"
  "$ARCH_DEEP_VAL"
  "$ARCH_WIDE_VAL"
)




# python picid/run.py paths=rt_local  experiment=nb14/reconstruction/residual_autoencoder trainer.max_epochs=500 trainer.check_val_every_n_epoch=5
# python picid/run.py paths=rt_local  experiment=unibo/reconstruction/residual_autoencoder trainer.max_epochs=500 trainer.check_val_every_n_epoch=5
# python picid/run.py paths=rt_local  experiment=xjtu_sy/reconstruction/residual_autoencoder trainer.max_epochs=500 trainer.check_val_every_n_epoch=5
# python picid/run.py paths=rt_local  experiment=pronostia/reconstruction/residual_autoencoder trainer.max_epochs=500 trainer.check_val_every_n_epoch=5
