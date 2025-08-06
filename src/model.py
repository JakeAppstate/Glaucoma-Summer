import torch

class ModelWithTemperature(torch.nn):
    def __init__(self, model, **kwargs):
        super().__init(**kwargs)

        self.model = model
        self.temp = torch.nn.Parameter([1.5])

    def forward(self, x, **kwargs):
        logits = self.model(x, **kwargs)
        return torch.sigmoid(logits / self.temp)