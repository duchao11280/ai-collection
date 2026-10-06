import os
os.environ['HF_HOME'] = './cache'
import time
import re
import cv2
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class Qwen2VLObjectDetector:
    def __init__(self, weight_path=None, device="auto"):
        self.weight_path = weight_path
        self.model, self.processor = self.build_model(self.weight_path)
        self.system_prompt = "You are a helpfull assistant to detect objects in images. When asked to detect elements based on a description you return bounding boxes for all elements in the form of [xmin, ymin, xmax, ymax] whith the values beeing scaled to 1000 by 1000 pixels. When there are more than one result, answer with a list of bounding boxes in the form of [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]."
        self.device = device

    def build_model(self, weight_path=None):
        # default: Load the model on the available device(s)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            weight_path, torch_dtype="auto", device_map=self.device
        )
        # default processer
        processor = AutoProcessor.from_pretrained(weight_path)

        return model, processor

    def detect_objects(self, image_path, prompt):
        start_time = time.perf_counter()
        image_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": self.system_prompt},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        width, height = image_inputs[0].size
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        end_time = time.perf_counter()
        pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        matches = re.findall(pattern, str(output_text))
        parsed_boxes = [[int(num) for num in match] for match in matches]

        print("Cost:", end_time - start_time)
        print(output_text)
        print(parsed_boxes)
        return output_text, parsed_boxes
    
    def process_prediction(self, pred, raw_image_path, save_image_path):
        image = cv2.imread(raw_image_path)
        height, width, _ = image.shape
        bounding_boxes = []
        output_text, parsed_boxes = pred
        scaled_boxes = Qwen2VLObjectDetector.rescale_bounding_boxes(parsed_boxes, width, height)
        for box in scaled_boxes:
            xmin, ymin, xmax, ymax = box
            bounding_boxes.append([xmin, ymin, xmax, ymax])
        image_with_boxes = Qwen2VLObjectDetector.draw_bounding_boxes_cv2(image, bounding_boxes)
        cv2.imwrite(save_image_path, image_with_boxes)
        return save_image_path

    @staticmethod
    def rescale_bounding_boxes(bounding_boxes, original_width, original_height, scaled_width=1000, scaled_height=1000):
        x_scale = original_width / scaled_width
        y_scale = original_height / scaled_height
        rescaled_boxes = []
        for box in bounding_boxes:
            xmin, ymin, xmax, ymax = box
            rescaled_box = [
                xmin * x_scale,
                ymin * y_scale,
                xmax * x_scale,
                ymax * y_scale
            ]
            rescaled_boxes.append(rescaled_box)
        return rescaled_boxes

    @staticmethod
    def draw_bounding_boxes_cv2(image, bounding_boxes, outline_color=(0, 255, 0), line_width=2):
        for box in bounding_boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), outline_color, line_width)
        return image


if __name__ == "__main__":
    weight_path = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
    detector = Qwen2VLObjectDetector(weight_path=weight_path)
    image_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    output_path = "output.jpeg"
    prompt = "The dog"
    pred = detector.detect_objects(image_path, prompt)
    image_with_boxes = detector.process_prediction(pred, image_path, output_path)

