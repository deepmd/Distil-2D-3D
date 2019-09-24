from libs.eval.retrieve_clips import RetrieveClipsEval


def get_evaluator(opt, model):
    if opt.eval.method == 'Retrieval':
        return RetrieveClipsEval(opt, model)
    else:
        raise ValueError("There's no evaluation method named '{}'!".format(opt.eval.method))
