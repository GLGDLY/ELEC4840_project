import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class MeanTeacherFramework(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.student = base_model
        self.teacher = copy.deepcopy(base_model)
        self.ema = EMA(self.teacher)
        self.ema.register()

        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def update_teacher(self):
        self.ema.update()

    def forward(self, x):
        return self.student(x)
