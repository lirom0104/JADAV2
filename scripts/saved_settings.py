"""Read saved argparse settings without importing training or experiment code."""

import ast


def saved_namespace(path):
    tree = ast.parse(path.read_text(), mode="eval").body
    if (not isinstance(tree, ast.Call) or not isinstance(tree.func, ast.Name)
            or tree.func.id != "Namespace" or tree.args):
        raise ValueError(f"Invalid saved Namespace: {path}")
    return {item.arg: ast.literal_eval(item.value) for item in tree.keywords}


def options(root, key, filename="optimization_args"):
    path = root / key[0] / key[1] / filename
    return saved_namespace(path) if path.exists() else None
