import argparse
import logging
import os
import PIL
import torch
import cv2
from torchvision import transforms
from torch.autograd import Variable
from typing import Union
from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes


class Slatedetection(ClamsApp):

    def __init__(self):
        self.device = torch.device("cpu")
        self.model = torch.load(
            os.path.join("data","slate_model.pth"), map_location=torch.device("cpu")
        )
        self.model.eval()
        super().__init__()

    def _appmetadata(self):
        #see metadata.py
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        
        logging.debug(f"loading documents with type: {DocumentTypes.VideoDocument}")
        video_filename = mmif.get_document_location(DocumentTypes.VideoDocument)
        logging.debug(f"video_filename: {video_filename}")
        new_view = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.metadata.get_parameter()
        # fill the params dict with the default values if not provided
        parameters = self.get_configuration(**parameters)
        unit = parameters["timeUnit"]
        new_view.new_contain(
            AnnotationTypes.TimeFrame,
            timeUnit=unit,
            document=mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0].id
        )
        slate_output = self.run_slatedetection(video_filename, **parameters)
        if unit == "milliseconds":
            slate_output = slate_output[1]
        elif unit == "frames":
            slate_output = slate_output[0]
        else:
            raise TypeError(
                "invalid"
            )
        for _id, frames in enumerate(slate_output):
            start_frame, end_frame = frames
            timeframe_annotation = new_view.new_annotation(AnnotationTypes.TimeFrame)
            timeframe_annotation.add_property("start", int(start_frame))
            timeframe_annotation.add_property("end", int(end_frame))
            timeframe_annotation.add_property("frameType","slate")
        return mmif

    def run_slatedetection(self, video_filename, **parameters):
        image_transforms = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor()]
        )
        sample_ratio = int(parameters.get("sampleRatio", 30))
        min_duration = int(parameters.get("minFrameCount", 10))
        stop_after_one = parameters.get("stopAfterOne", True)
        stop_at = int(parameters.get("stopAt", 30*60*60*5))

        threshold = 0.5 if "threshold" not in parameters else float(parameters["threshold"])

        def frame_is_slate(frame_, _threshold=threshold):
            image_tensor = image_transforms(PIL.Image.fromarray(frame_)).float()
            image_tensor = image_tensor.unsqueeze_(0)
            input = Variable(image_tensor)
            input = input.to(self.device)
            output = self.model(input)
            output = torch.nn.functional.softmax(output, dim=1)
            output = output.data.cpu().numpy()[0]
            return output.data[1] > _threshold
        
        cap = cv2.VideoCapture(video_filename)
        counter = 0
        frame_number_result = []
        seconds_result = []
        in_slate = False
        start_frame = None
        start_seconds = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if counter > stop_at:
                if in_slate:
                    if counter - start_frame > min_duration:
                        frame_number_result.append((start_frame, counter))
                        seconds_result.append((start_seconds, cap.get(cv2.CAP_PROP_POS_MSEC)))
                break 
            if counter % sample_ratio == 0:
                result = frame_is_slate(frame)
                if result:
                    if not in_slate:
                        in_slate = True
                        start_frame = counter
                        start_seconds = cap.get(cv2.CAP_PROP_POS_MSEC)
                else:
                    if in_slate:
                        in_slate = False
                        if counter - start_frame > min_duration:
                            frame_number_result.append((start_frame, counter))
                            seconds_result.append((start_seconds, cap.get(cv2.CAP_PROP_POS_MSEC)))
                        if stop_after_one:
                            return frame_number_result, seconds_result
            counter += 1
        return frame_number_result, seconds_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", action="store", default="5000", help="set port to listen"
    )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    app = Slatedetection()

    http_app = Restifier(app, port=int(parsed_args.port)
    )
    if parsed_args.production:
        http_app.serve_production()
    else:
        http_app.run()
