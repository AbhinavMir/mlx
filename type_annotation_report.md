# Type Annotation Report

## ./python/mlx/extension.py

### Missing function type annotations

- run (return)

## ./python/mlx/utils.py

### Missing function type annotations

- tree_map (return)
- tree_map (parameter: fn)
- tree_map (parameter: tree)
- tree_map_with_path (return)
- tree_map_with_path (parameter: fn)
- tree_map_with_path (parameter: tree)
- tree_flatten (return)
- tree_flatten (parameter: tree)
- tree_flatten (parameter: prefix)
- tree_flatten (parameter: is_leaf)
- tree_unflatten (return)
- tree_unflatten (parameter: tree)
- tree_reduce (return)
- tree_reduce (parameter: fn)
- tree_reduce (parameter: tree)
- tree_reduce (parameter: initializer)
- tree_reduce (parameter: is_leaf)

## ./python/mlx/_reprlib_fix.py

### Missing function type annotations

- repr_array (return)
- repr_array (parameter: x)
- repr_array (parameter: maxlevel)

## ./python/mlx/nn/init.py

### Missing function type annotations

- _calculate_fan_in_fan_out (return)
- _calculate_fan_in_fan_out (parameter: x)

## ./python/mlx/nn/utils.py

### Missing function type annotations

- value_and_grad (return)
- checkpoint (return)
- inner_fn (return)
- inner_fn (parameter: params)
- wrapped_value_grad_fn (return)
- inner_fn (return)
- inner_fn (parameter: params)
- wrapped_checkpointed_fn (return)

## ./python/mlx/nn/losses.py

### Missing function type annotations

- _reduce (return)
- _drop_dim (return)
- _drop_dim (parameter: shape)
- _drop_dim (parameter: axis)

## ./python/mlx/nn/layers/normalization.py

### Missing function type annotations

- __init__ (return)
- _extra_repr (return)
- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- _extra_repr (return)
- _pytorch_compatible_group_norm (return)
- _pytorch_compatible_group_norm (parameter: x)
- _group_norm (return)
- _group_norm (parameter: x)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- unfreeze (return)
- _extra_repr (return)

## ./python/mlx/nn/layers/pooling.py

### Missing function type annotations

- _value_or_list (return)
- _value_or_list (parameter: x)
- _value_or_list (parameter: n)
- _value_or_list (parameter: msg)
- _non_overlapping_sliding_windows (return)
- _non_overlapping_sliding_windows (parameter: x)
- _non_overlapping_sliding_windows (parameter: shape)
- _non_overlapping_sliding_windows (parameter: window_shape)
- _sliding_windows (return)
- _sliding_windows (parameter: x)
- _sliding_windows (parameter: window_shape)
- _sliding_windows (parameter: window_strides)
- __init__ (return)
- __init__ (parameter: pooling_function)
- __init__ (parameter: kernel_size)
- __init__ (parameter: stride)
- __init__ (parameter: padding)
- __init__ (parameter: padding_value)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- __init__ (parameter: pooling_function)
- __init__ (parameter: padding_value)
- __init__ (return)
- __init__ (parameter: pooling_function)
- __init__ (parameter: padding_value)
- __init__ (return)
- __init__ (return)
- __init__ (return)
- __init__ (return)

## ./python/mlx/nn/layers/activations.py

### Missing function type annotations

- _make_activation_module (return)
- _make_activation_module (parameter: f)
- sigmoid (return)
- sigmoid (parameter: x)
- relu (return)
- relu (parameter: x)
- leaky_relu (return)
- leaky_relu (parameter: x)
- leaky_relu (parameter: negative_slope)
- log_softmax (return)
- log_softmax (parameter: x)
- log_softmax (parameter: axis)
- elu (return)
- elu (parameter: x)
- elu (parameter: alpha)
- relu6 (return)
- relu6 (parameter: x)
- softmax (return)
- softmax (parameter: x)
- softmax (parameter: axis)
- softplus (return)
- softplus (parameter: x)
- softsign (return)
- softsign (parameter: x)
- softshrink (return)
- softshrink (parameter: x)
- celu (return)
- celu (parameter: x)
- celu (parameter: alpha)
- silu (return)
- silu (parameter: x)
- log_sigmoid (return)
- log_sigmoid (parameter: x)
- gelu (parameter: x)
- gelu_approx (return)
- gelu_approx (parameter: x)
- gelu_fast_approx (return)
- gelu_fast_approx (parameter: x)
- step (return)
- selu (return)
- selu (parameter: x)
- hardswish (return)
- hardswish (parameter: x)
- hard_tanh (return)
- hard_tanh (parameter: x)
- hard_tanh (parameter: min_val)
- hard_tanh (parameter: max_val)
- hard_shrink (return)
- hard_shrink (parameter: x)
- hard_shrink (parameter: lambd)
- softmin (return)
- softmin (parameter: x)
- softmin (parameter: axis)
- tanh (return)
- tanh (parameter: x)
- decorator (return)
- decorator (parameter: klass)
- __init__ (return)
- __call__ (parameter: x)
- __init__ (return)
- __init__ (parameter: negative_slope)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- __init__ (parameter: alpha)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- __init__ (parameter: lambd)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- __init__ (parameter: alpha)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- __init__ (parameter: num_parameters)
- __init__ (parameter: init)
- __call__ (return)
- __init__ (return)
- __init__ (parameter: approx)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- __call__ (return)

## ./python/mlx/nn/layers/linear.py

### Missing function type annotations

- to_quantized (return)

## ./python/mlx/nn/layers/quantized.py

### Missing function type annotations

- quantize (return)
- _maybe_quantize (return)
- _maybe_quantize (parameter: path)
- _maybe_quantize (parameter: m)
- __init__ (return)
- __call__ (return)
- __call__ (parameter: x)
- as_linear (return)
- as_linear (parameter: x)
- _extra_repr (return)
- from_embedding (return)
- from_embedding (parameter: cls)
- __init__ (return)
- unfreeze (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- from_linear (return)
- from_linear (parameter: cls)

## ./python/mlx/nn/layers/embedding.py

### Missing function type annotations

- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- as_linear (return)
- as_linear (parameter: x)
- to_quantized (return)

## ./python/mlx/nn/layers/upsample.py

### Missing function type annotations

- _scaled_indices (return)
- _scaled_indices (parameter: N)
- _scaled_indices (parameter: scale)
- _scaled_indices (parameter: align_corners)
- _scaled_indices (parameter: dim)
- _scaled_indices (parameter: ndims)
- _nearest_indices (return)
- _nearest_indices (parameter: N)
- _nearest_indices (parameter: scale)
- _nearest_indices (parameter: dim)
- _nearest_indices (parameter: ndims)
- _linear_indices (return)
- _linear_indices (parameter: N)
- _linear_indices (parameter: scale)
- _linear_indices (parameter: align_corners)
- _linear_indices (parameter: dim)
- _linear_indices (parameter: ndims)
- _cubic_indices (return)
- _cubic_indices (parameter: N)
- _cubic_indices (parameter: scale)
- _cubic_indices (parameter: align_corners)
- _cubic_indices (parameter: dim)
- _cubic_indices (parameter: ndims)
- upsample_nearest (return)
- _interpolate (return)
- upsample_linear (return)
- upsample_cubic (return)
- _get_weight (return)
- _get_weight (parameter: ind)
- _get_weight (parameter: grid)
- _get_weight (parameter: dist)
- __init__ (return)

## ./python/mlx/nn/layers/convolution.py

### Missing function type annotations

- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)

## ./python/mlx/nn/layers/containers.py

### Missing function type annotations

- __init__ (return)
- __call__ (return)
- __call__ (parameter: x)

## ./python/mlx/nn/layers/positional_encoding.py

### Missing function type annotations

- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- __call__ (return)
- __call__ (parameter: x)
- create_alibi_matrix (return)
- create_alibi_matrix (parameter: cls)
- create_alibi_matrix (parameter: dtype)
- create_alibi_slope (return)
- create_alibi_slope (parameter: num_heads)
- __call__ (return)
- __call__ (parameter: attention_scores)
- __call__ (parameter: offset)
- __call__ (parameter: mask)

## ./python/mlx/nn/layers/base.py

### Missing function type annotations

- _unwrap (return)
- _unwrap (parameter: model)
- _unwrap (parameter: value_key)
- _unwrap (parameter: value)
- _unwrap (parameter: filter_fn)
- _unwrap (parameter: map_fn)
- _unwrap (parameter: is_leaf_fn)
- __init__ (return)
- training (return)
- state (return)
- _extra_repr (return)
- __repr__ (return)
- __getattr__ (return)
- __setattr__ (return)
- save_weights (return)
- is_module (return)
- is_module (parameter: value)
- valid_child_filter (return)
- valid_child_filter (parameter: module)
- valid_child_filter (parameter: key)
- valid_child_filter (parameter: value)
- valid_parameter_filter (return)
- valid_parameter_filter (parameter: module)
- valid_parameter_filter (parameter: key)
- valid_parameter_filter (parameter: value)
- trainable_parameter_filter (return)
- trainable_parameter_filter (parameter: module)
- trainable_parameter_filter (parameter: key)
- trainable_parameter_filter (parameter: value)
- filter_and_map (return)
- parameters (return)
- trainable_parameters (return)
- children (return)
- leaf_modules (return)
- modules (return)
- named_modules (return)
- _validate_keys (return)
- _validate_keys (parameter: keys)
- _validate_keys (parameter: strict)
- set_dtype (return)
- _is_leaf_module (return)
- _is_leaf_module (parameter: m)
- _is_leaf_module (parameter: k)
- _is_leaf_module (parameter: v)
- apply (return)
- apply (parameter: dst)
- apply (parameter: parameters)
- apply (return)
- apply (parameter: dst)
- apply (parameter: modules)
- _freeze_impl (return)
- _freeze_impl (parameter: _)
- _freeze_impl (parameter: m)
- _unfreeze_impl (return)
- _unfreeze_impl (parameter: _)
- _unfreeze_impl (parameter: m)
- _set_train (return)
- _set_train (parameter: _)
- _set_train (parameter: m)
- predicate (return)
- predicate (parameter: _)

## ./python/mlx/nn/layers/transformer.py

### Missing function type annotations

- __init__ (return)
- __call__ (return)
- __call__ (parameter: queries)
- __call__ (parameter: keys)
- __call__ (parameter: values)
- __call__ (parameter: mask)
- create_additive_causal_mask (return)
- __init__ (return)
- __call__ (return)
- __call__ (parameter: x)
- __call__ (parameter: mask)
- __init__ (return)
- __init__ (parameter: activation)
- __call__ (return)
- __call__ (parameter: x)
- __call__ (parameter: mask)
- __init__ (return)
- __call__ (return)
- __call__ (parameter: x)
- __call__ (parameter: memory)
- __call__ (parameter: x_mask)
- __call__ (parameter: memory_mask)
- __init__ (return)
- __init__ (parameter: activation)
- __call__ (return)
- __call__ (parameter: x)
- __call__ (parameter: memory)
- __call__ (parameter: x_mask)
- __call__ (parameter: memory_mask)
- __init__ (return)
- __call__ (return)
- __call__ (parameter: src)
- __call__ (parameter: tgt)
- __call__ (parameter: src_mask)
- __call__ (parameter: tgt_mask)
- __call__ (parameter: memory_mask)

## ./python/mlx/nn/layers/dropout.py

### Missing function type annotations

- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)

## ./python/mlx/nn/layers/recurrent.py

### Missing function type annotations

- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- __call__ (parameter: hidden)
- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- __call__ (parameter: hidden)
- __init__ (return)
- _extra_repr (return)
- __call__ (return)
- __call__ (parameter: x)
- __call__ (parameter: hidden)
- __call__ (parameter: cell)

## ./python/mlx/optimizers/optimizers.py

### Missing function type annotations

- clip_grad_norm (return)
- clip_grad_norm (parameter: grads)
- clip_grad_norm (parameter: max_norm)
- __init__ (return)
- __init__ (parameter: schedulers)
- update (return)
- init (return)
- init_single (return)
- apply_gradients (return)
- apply_single (return)
- state (return)
- state (return)
- step (return)
- learning_rate (return)
- learning_rate (return)
- _maybe_schedule (return)
- __init__ (return)
- init_single (return)
- apply_single (return)
- __init__ (return)
- init_single (return)
- apply_single (return)
- __init__ (return)
- init_single (return)
- apply_single (return)
- __init__ (return)
- init_single (return)
- apply_single (return)
- __init__ (return)
- init_single (return)
- apply_single (return)
- __init__ (return)
- apply_single (return)
- __init__ (return)
- init_single (return)
- apply_single (return)
- __init__ (return)
- init_single (return)
- apply_single (return)
- __init__ (return)
- init_single (return)
- _compute_rms (return)
- _compute_rms (parameter: inputs)
- _compute_learning_rate (return)
- _compute_learning_rate (parameter: step)
- _compute_learning_rate (parameter: parameter_rms)
- _approximate_exp_moving_avg (return)
- _approximate_exp_moving_avg (parameter: exp_avg_sq_row)
- _approximate_exp_moving_avg (parameter: exp_avg_sq_col)
- apply_single (return)
- clipper (return)
- clipper (parameter: g)

## ./python/mlx/optimizers/schedulers.py

### Missing function type annotations

- schedule (return)
- schedule (parameter: step)
- schedule (return)
- schedule (parameter: step)
- scheduler (return)
- scheduler (parameter: step)
- schedule (return)
- schedule (parameter: step)
- step_fn (return)
- step_fn (parameter: step)

