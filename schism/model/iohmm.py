"""
IOHMMBase: subclass hmmlearn GaussianHMM. Override transition with softmax(Ut). Enforce Tr(Sigma) <= tau + epsilon*I floor post M-step.
"""
