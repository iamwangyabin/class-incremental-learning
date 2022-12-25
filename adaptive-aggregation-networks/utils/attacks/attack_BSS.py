from torch import autograd
# from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
from .helpers import *
from utils.process_fp import process_inputs_fp


class AttackBSS:

    def __init__(
            self,
            targeted=True, max_epsilon=16, norm=float('inf'),
            step_alpha=None, num_steps=None, debug=False, device=None):

        self.targeted = targeted
        self.eps = 5.0 * max_epsilon / 255.0
        self.num_steps = num_steps or 10
        self.norm = norm
        if not step_alpha:
            if norm == float('inf'):
                self.step_alpha = self.eps / self.num_steps
            else:
                if norm == 1:
                    self.step_alpha = 500.0
                else:
                    self.step_alpha = 1.0
        else:
            self.step_alpha = step_alpha
        self.loss_fn = torch.nn.CrossEntropyLoss(size_average=False).to(device)
        self.device = device
        self.debug = debug

    def run(self, model, input, target, the_args=None, ref_fusion_vars=None, ref_b2_model=None):
        # input: batch_size x channel x height x width
        # target: batch_size
        input = input.to(self.device)
        input_var = autograd.Variable(input, requires_grad=True).to(self.device)
        target_var = autograd.Variable(target).to(self.device)
        GT_var = autograd.Variable(target).to(self.device)
        step = 0
        model.eval()
        while step < self.num_steps:

            if ref_b2_model is None:
                output = model(input_var)
            else:
                output, _ = process_inputs_fp(the_args, ref_fusion_vars, model, ref_b2_model, input_var)


            model.zero_grad()
            input_var.retain_grad()
            if not step:
                GT_var.data = output.data.max(1)[1]

            score = output

            score_GT = score.gather(1, GT_var.unsqueeze(1))
            score_target = score.gather(1, target_var.unsqueeze(1))

            loss = (score_target - score_GT).sum()
            loss.backward()

            step_alpha = self.step_alpha * (GT_var.data == output.data.max(1)[1]).float()
            step_alpha = step_alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            if step_alpha.sum() == 0:
                break

            pert = ((score_GT.data - score_target.data).unsqueeze(1).unsqueeze(1))
            normed_grad = step_alpha * (pert+1e-4) * input_var.grad.data / (l2_norm(input_var.grad.data))

            # perturb current input image by normalized and scaled gradient
            overshoot = 0.0
            step_adv = input_var.data + (1+overshoot) * normed_grad

            total_adv = step_adv - input

            # apply total adversarial perturbation to original image and clip to valid pixel range
            input_adv = input + total_adv
            input_adv = torch.clamp(input_adv, -2.5, 2.5)
            input_var.data = input_adv
            step += 1

            if input_var.grad is not None:
                input_var.grad.zero_()

        return input_adv

