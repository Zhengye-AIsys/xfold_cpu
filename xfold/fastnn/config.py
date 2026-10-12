# options: ["torch", "triton"]
layer_norm_implementation = "torch"

# options: ["torch", "triton"]
dot_product_attention_implementation = "torch"

# options: ["torch", "triton", "cpp"] (cpp: optional Linux x86_64 CPU extension)
gated_linear_unit_implementation = "torch"
