import torch 
import torch.nn.functional as F

class Optimization():
    def __init__(self, train_loader, device):
        self.train_loader = train_loader
        self.device = device

    def cdw_feature_distance(self, old_model, old_classifier, new_model):
        """cosine distance weight (cdw): calculate feature distance of 
           the features of a batch of data by cosine distance.
        """
        old_model=old_model.to(self.device)
        old_classifier=old_classifier.to(self.device)
        
        for data in self.train_loader:
            inputs, _ = data
            inputs=inputs.to(self.device)

            with torch.no_grad():
                old_out = old_classifier(old_model(inputs))
                new_out = new_model(inputs)
        
            distance = 1 - torch.cosine_similarity(old_out, new_out)
            return torch.mean(distance)

    def kd_generate_soft_label(self, model, data, regularization):
        """knowledge distillation (kd): generate soft labels.
        """
        result = model(data)
        if regularization:
            result = F.normalize(result, dim=1, p=2)
        return result
