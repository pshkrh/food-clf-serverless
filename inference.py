import config as cfg
import torch
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
import io


class Inference:
    def __init__(self, model_path, device):
        """Initialize the paths and device, but don't load the model yet."""
        self.device = device
        self.model_path = model_path
        self.model = None  # Model will be loaded later

    def load_model(self):
        """Load the model into memory."""
        model = models.mobilenet_v3_large()

        # Modify last layer
        num_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(num_features, 101)

        # Load model weights
        model.load_state_dict(
            torch.load(self.model_path, map_location=torch.device(self.device))
        )
        model.to(self.device)
        model.eval()
        return model

    def predict_image(self, image_data):
        """Predict the class of an image and return top 5 predictions."""
        if self.model is None:
            self.model = self.load_model()  # Load the model upon first request

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        # Load and transform the image
        image = Image.open(io.BytesIO(image_data))
        image = transform(image).unsqueeze(0)
        image = image.to(self.device)

        # Perform the prediction
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)
            top5_prob, top5_class = torch.topk(probabilities, 5)

        top5_prob = top5_prob.cpu().numpy()[0] * 100  # Convert to percentages
        top5_prob = [float(prob) for prob in top5_prob]  # Convert to float list
        top5_class_indices = top5_class.cpu().numpy()[0]

        # Map class indices to class names
        top5_class_names = [cfg.CLASS_NAMES[str(index)] for index in top5_class_indices]

        predicted_class = cfg.CLASS_NAMES[str(top5_class_indices[0])]

        # Return both the predicted class and top 5 predictions
        return {
            "predicted_class": predicted_class,
            "top_5_predictions": [
                {"class": class_name, "probability": round(prob, 5)}
                for class_name, prob in zip(top5_class_names, top5_prob)
            ],
        }
