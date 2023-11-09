from typing import Tuple, Dict
import gradio as gr
from sources import models
from timeit import default_timer as timer
from PIL import Image

# Setup class names
class_names = ["pizza", "steak", "sushi"]

# create a pretrained EfficientNetB2 model
model, model_transforms = models.create_effnet(
    effnet_version=2,
    device=torch.device("cpu")
)

# load model state dict
model.load_state_dict(torch.load(
    f="models/EfficientNet_B2.pth",
    map_location=torch.device("cpu")
))


def predict(img: Image)->Tuple[Dict,float]:
    """predict the food name probability from a list of class names
    Args:
        img: PIL image
    Return:
        Tuple[Dict, float]
            Dict: dictionary of food name probability
            float: wall time of the food name prediction
    """

    # start the timer
    start_time = timer()
    
    # transform PIL image to torch tensor, add batch dimension
    img = model_transforms(img).unsqueeze(dim=0)

    # switch model to evaluation mode
    model.eval()
    with torch.inference_mode():
        # food name probability prediction
        pred_probs = torch.softmax(model(img), dim=1)

    # assign prediction probability to a result dictionary
    pred_probs_dict = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    # prediction wall time
    pred_wtime = timer() - start_time

    # return probability dictionary and wall time
    return pred_probs_dict, pred_wtime

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Predictions"),
        gr.Number(label="Prediction time (s)")
    ],
    examples=example_list
)

demo.launch()
