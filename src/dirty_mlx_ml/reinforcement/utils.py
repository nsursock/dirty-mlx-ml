import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def polyak_update(params, target_params, tau: float):
    import mlx.core as mx
    from mlx.utils import tree_map

    return tree_map(lambda p, tp: tau * p + (1.0 - tau) * tp, params, target_params)


def explained_variance(y_pred, y_true):
    import mlx.core as mx

    var_y = mx.var(y_true)
    return (1.0 - mx.var(y_true - y_pred) / (var_y + 1e-8)).item()


def to_float(x):
    import mlx.core as mx

    if isinstance(x, mx.array):
        mx.eval(x)
        return float(x.item()) if x.size == 1 else float(mx.mean(x).item())
    return float(x)
