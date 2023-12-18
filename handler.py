import json
import logging
import base64
import config as cfg
from inference import Inference

inference = Inference(cfg.MODEL_PATH, cfg.DEVICE)


def predict(event, context):
    try:
        body = json.loads(event["body"])
        if "image" in body:
            # Decode the base64 string to bytes
            image_data = base64.b64decode(body["image"])

            predictions = inference.predict_image(image_data)
            logging.info(f"Predicted class: {predictions['predicted_class']}")
            logging.info(f"Top 5 Predictions: {predictions['top_5_predictions']}")

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Credentials": True,
                },
                "body": json.dumps(predictions),
            }
        else:
            raise ValueError("Image data not found in the request")

    except Exception as e:
        logging.error(e)
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.dumps({"error": repr(e)}),
        }
