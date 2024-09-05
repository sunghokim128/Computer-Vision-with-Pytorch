def dataset_name(model_name):
        if "mnist" in model_name:
            return "mnist"
        # cifar100 if statement must come before cifar10 if statement!!!
        elif "cifar100" in model_name:
            return "cifar100"
        elif "cifar10" in model_name:
            return "cifar10"