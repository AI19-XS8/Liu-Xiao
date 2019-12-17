import numpy as np
import torch
import torch.nn as nn
from _YSZ_ import _Base_ as base


class FGSM:
    def __init__(self, model=None, epsilon=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilons=None):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_nat)

        X_var = base.to_var(torch.from_numpy(X), requires_grad=True)
        y_var = base.to_var(torch.LongTensor(y))

        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()
        X += self.epsilon * grad_sign
        X = np.clip(X, 0, 1)
        grad = torch.Tensor(self.epsilon * grad_sign)
        grad=X_var.grad.data.cpu()
        #print(grad)

        X = torch.from_numpy(X)
        return X
