import logging

logger = logging.getLogger("base")


def create_model(opt, **kwargs):
    model = opt["model"]

    if model == "denoising":
        from .denoising_model import DenoisingModel as M
    elif model == "denoising_ss":
        from .denoising_model_ss import DenoisingModelSS as M
    elif model == "ff_model":
        from .ff_model import FeedForwardModel as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt, **kwargs)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
