from typing import List, Tuple
import torch
import torch.nn.functional as F

from moviad.models.audio.audio_vad_model import AudioVADModel
from moviad.models.training_args import TrainingArgs
from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor

class STFPM(AudioVADModel):

    def __init__ (
        self,
        teacher:AudioFeatureExtractor,
        student:AudioFeatureExtractor,
    ):
        super().__init__(feature_extractor=student, device=student.device)
        self.teacher = teacher
        self.student = student

    @staticmethod
    def _loss(teacher_features, student_features):
        loss = 0
        for i in range(len(student_features)):
            teacher_features[i] = F.normalize(teacher_features[i], dim=1)
            student_features[i] = F.normalize(student_features[i], dim=1)
            loss += torch.sum((teacher_features[i] - student_features[i]) ** 2, 1).mean()
        return loss

    def to(self, device: torch.device | str):
        device = torch.device(device)
        super().to(device)
        self.teacher.to(device)
        self.student.to(device)
        return self

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        if training_args.optimizer is None:
            training_args.optimizer = torch.optim.SGD(
                self.student.model.parameters(),
                lr=0.4,
                momentum=0.9,
                weight_decay=1e-4,
            )

        batch = self.batch_input(batch).to(self.device)
        teacher_features, student_features = self(batch)
        loss = self._loss(teacher_features, student_features)

        training_args.optimizer.zero_grad()
        loss.backward()
        training_args.optimizer.step()
        return loss.item()
        
    def forward(self, batch: torch.Tensor): 

        if self.training: 
            teacher_features, student_features = None, None
            with torch.no_grad():
                teacher_features = self.teacher(batch)
            student_features = self.student(batch)

            return teacher_features, student_features 
        
        else: 
            student_features = self.student(batch)
            teacher_features = self.teacher(batch)

            return self.post_process(teacher_features, student_features)
        
    def train(self, *args, **kwargs):
        self.teacher.model.eval()
        self.student.model.train()
        return super().train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.teacher.model.eval()
        self.student.model.eval()
        return super().eval(*args, **kwargs)

    
    def post_process(self, t_feat, s_feat) -> torch.Tensor:
        
        """
        This method actually produces the anomaly maps for evalution purposes

        Args:
            - t_feat: teacher features maps
            - s_feat: student features maps

        Returns: 
            - anomaly maps

        """

        device = t_feat[0].device
        score_maps = torch.tensor([1.0], device=device)
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(
                sm, size=(self.student.spec_shape[2], self.student.spec_shape[3]), mode="bilinear", align_corners=False
            )
            # aggregate score map by element-wise product
            score_maps = score_maps * sm

        tmp_scores = score_maps.squeeze(1).topk(5,dim=2).values.mean(dim=2)

        anomaly_scores = torch.max(score_maps.view(score_maps.size(0), -1), dim=1)[0]
        return score_maps, anomaly_scores, tmp_scores
