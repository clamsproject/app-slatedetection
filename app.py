import os
import logging
import PIL
import clams
import torch
import cv2
from torchvision import transforms
from torch.autograd import Variable
from clams.app import ClamsApp
from clams.restify import Restifier
from mmif.vocabulary import AnnotationTypes, DocumentTypes
from mmif import Document


APP_VERSION = 0.1


class SlateDetection(ClamsApp):
    def _appmetadata(self):
        metadata = {
            "name": "Slate Detection",
            "description": "This tool detects slates.",
            "app_version": str(APP_VERSION),
            "license": "MIT",
            "identifier": f"http://mmif.clams.ai/apps/slatedetect/{APP_VERSION}",
            "input": [{"@type": DocumentTypes.VideoDocument, "required": True}],
            "output": [{"@type": AnnotationTypes.TimeFrame, "properties": {"frameType": "string"}}],
            "parameters": [
                {
                    "name": "timeUnit",
                    "type": "string",
                    "choices": ["frames", "milliseconds"],
                    "default": "msec",
                    "description": "Unit for output typeframe.",
                },
                {
                    "name": "sampleRatio",
                    "type": "integer",
                    "default": "30",
                    "description": "Frequency to sample frames.",
                },
                {
                    "name": "stopAt",
                    "type": "integer",
                    "default": 30 * 60 * 60 * 5,
                    "description": "Frame number to stop processing",
                },
                {
                    "name": "stopAfterOne",
                    "type": "boolean",
                    "default": True,
                    "description": "When True, processing stops after first timeframe is found.",
                },
                {
                    "name": "minFrameCount",
                    "type": "integer",
                    "default": 10,  # minimum value = 1 todo how to include minimum
                    "description": "Minimum number of frames required for a timeframe to be included in the output",
                },
            ],
        }
        return clams.AppMetadata(**metadata)

    def __init__(self):
        self.device = torch.device("cpu")
        self.model = torch.load(
            os.path.join("data", "slate_model.pth"), map_location=torch.device("cpu")
        )
        self.model.eval()
        super().__init__()

    def _annotate(self, mmif, **kwargs):
        logging.debug(f"loading document with type: {DocumentTypes.VideoDocument}")
        video_filename = mmif.get_document_location(DocumentTypes.VideoDocument)
        logging.debug(f"video_filename: {video_filename}")
        config = self.get_configuration(**kwargs)

        new_view = mmif.new_view()
        self.sign_view(new_view, config)

        unit = "milliseconds" if "unit" not in kwargs else kwargs["unit"]
        new_view.new_contain(
            AnnotationTypes.TimeFrame,
            timeUnit=unit,
            document=mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0].id
        )
        slate_output = self.run_slatedetection(video_filename, mmif, new_view, **kwargs)
        if unit == "milliseconds":
            slate_output = slate_output[1]
        elif unit == "frames":
            slate_output = slate_output[0]
        else:
            raise TypeError(
                "invalid unit type"
            )  ##todo 6/29/21 kelleylynch is handling valid input types be moved to sdk?
        for _id, frames in enumerate(slate_output):
            start_frame, end_frame = frames
            timeframe_annotation = new_view.new_annotation(AnnotationTypes.TimeFrame)
            timeframe_annotation.add_property("start", int(start_frame))
            timeframe_annotation.add_property("end", int(end_frame))
            timeframe_annotation.add_property("frameType", "slate")
        return mmif


    def run_slatedetection(
        self, video_filename, mmif, view, **kwargs
    ):  # todo 6/1/21 kelleylynch this could be optimized by generating a batch of frames
        image_transforms = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor()]
        )
        sample_ratio = int(kwargs.get("sampleRatio", 30))
        min_duration = int(kwargs.get("minFrameCount", 10))
        stop_after_one = kwargs.get("stopAfterOne", True)
        stop_at = int(kwargs.get("stopAt", 30 * 60 * 60 * 5))  # default 5 hours

        def frame_is_slate(frame_):
            image_tensor = image_transforms(PIL.Image.fromarray(frame_)).float()
            image_tensor = image_tensor.unsqueeze_(0)
            input = Variable(image_tensor)
            input = input.to(self.device)
            output = self.model(input)
            index = output.data.cpu().numpy().argmax()
            return index == 1

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
                        seconds_result.append(
                            (start_seconds, cap.get(cv2.CAP_PROP_POS_MSEC))
                        )
                break
            if counter % sample_ratio == 0:
                result = frame_is_slate(frame)
                if result:  # in slate
                    if not in_slate:
                        in_slate = True
                        start_frame = counter
                        start_seconds = cap.get(cv2.CAP_PROP_POS_MSEC)
                else:
                    if in_slate:
                        in_slate = False
                        if counter - start_frame > min_duration:
                            frame_number_result.append((start_frame, counter))
                            seconds_result.append(
                                (start_seconds, cap.get(cv2.CAP_PROP_POS_MSEC))
                            )
                    if stop_after_one:
                        return frame_number_result, seconds_result
            counter += 1
        return frame_number_result, seconds_result


if __name__ == "__main__":
    slate_tool = SlateDetection()
    slate_service = Restifier(slate_tool)
    slate_service.run()
