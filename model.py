import torch
from torch import nn as nn
from torchvision import models
from torchinfo import summary
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.nn import functional as F


class OrthoFace(nn.Module):
    """
    doc
    """
    def __init__(self):
        super().__init__()
        # weights = models.InceptionResnetV1_Weights.DEFAULT
        self.base = InceptionResnetV1(pretrained='vggface2', classify=False)
        

    def forward(self, x):
        
        return self.base(x)


def main():
    x = torch.randn(size=(10, 3, 224, 224))
    model = OrthoFace()

    summary(model=model, input_size=[1, 3, 224, 224], row_settings=['var_names'])
    out = model(x)
    
    print(out.shape)

if __name__ == '__main__':
    main()




# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# model_ft = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes = 105)

# list(model_ft.children())[-6:]

# layer_list = list(model_ft.children())[-5:] # all final layers

# model_ft = nn.Sequential(*list(model_ft.children())[:-5])

# for param in model_ft.parameters():
#     param.requires_grad = False
    
# class Flatten(nn.Module):
#     def __init__(self):
#         super(Flatten, self).__init__()
        
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         return x
      
# class normalize(nn.Module):
#     def __init__(self):
#         super(normalize, self).__init__()
        
#     def forward(self, x):
#         x = F.normalize(x, p=2, dim=1)
#         return x    
# model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)

# model_ft.last_linear = nn.Sequential(
#     Flatten(),
#     nn.Linear(in_features=1792, out_features=512, bias=False),
#     normalize()
# )


# summary(model=model_ft, input_size=[1, 3, 224, 224], row_settings=['var_names'])


# # model_ft.logits = nn.Linear(layer_list[2].out_features, 105)
# # model_ft.softmax = nn.Softmax(dim=1)
# # model_ft = model_ft.to(device)
# # criterion = nn.CrossEntropyLoss()

# # # Observe that all parameters are being optimized
# # optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-2, momentum=0.9)

# # # Decay LR by a factor of *gamma* every *step_size* epochs
# # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  

# # model_ft = model_ft.to(device)
# # criterion = nn.CrossEntropyLoss()
# # # Observe that all parameters are being optimized
# # optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-2, momentum=0.9)
# # # Decay LR by a factor of *gamma* every *step_size* epochs
# # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# # model_ft
