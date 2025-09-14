import h5py

file_path = "language_model/parameter_logs/parameters_20250912_210535.h5"
with h5py.File(file_path, 'r') as f:
    grad = f['/gradients/step_0_epoch_0.00_training_step_4a8a2abd/transformer_h_0_attn_c_attn_weight'][:]
    print(grad.shape)   # (384, 1152)
    print(grad[:5, :5]) # preview first 5x5 entries
