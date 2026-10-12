# options: ["torch", "triton"]
layer_norm_implementation = "torch"

# options: ["torch", "triton", "cpp"]
dot_product_attention_implementation = "torch"

# options: ["torch", "triton", "cpp"] (cpp: optional Linux x86_64 CPU extension)
gated_linear_unit_implementation = "torch"

# options: ["torch", "cpp"]
triangle_multiplication_implementation = "torch"

# Stage 7 opt-ins; off preserves the earlier contiguous-weight cache behavior.
input_cache_enabled = False
packed_weight_cache_enabled = False
input_cache_max_bytes = 512 * 1024 * 1024
