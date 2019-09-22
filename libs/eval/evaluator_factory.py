from libs.eval.retrieve_clips import RetrieveClipsEval


def get_evaluator(opt, model):
    return RetrieveClipsEval(opt, model)
