""" util functions for model managment (save, load, deployment and etc.)
"""
import torch


def save_model(model, filepath):
    """
    """
    if isinstance(model, torch.nn.Module):
        torch.save(model, filepath)
        print("SUCCESS: saved model in {filepath}".format(filepath=filepath))
    else:
        raise TypeError("only `torch.nn.Module` is supported ({})".format(type(model)))


def load_model(filepath, turn_on_eval_mode=True):
    """
    """
    model = torch.load(filepath)
    # set dropout and batch normalization layers to evaluation
    # mode before running inference. Failed to do so, it will
    # yield inconsistant inference.
    if turn_on_eval_mode:
        model.eval()
    return model
