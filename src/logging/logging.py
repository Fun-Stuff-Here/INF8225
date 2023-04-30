def print_logs(dataset_type: str, logs: dict):
    """Print the logs.

    Args
    ----
        dataset_type: Either "Train", "Eval", "Test" type.
        logs: Containing the metric's name and value.
    """
    desc = [f"{name}: {value:.2f}" for name, value in logs.items()]
    desc = "\t".join(desc)
    desc = f"{dataset_type} -\t" + desc
    desc = desc.expandtabs(5)
    print(desc)
