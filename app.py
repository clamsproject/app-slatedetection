import os
import PIL
from imutils.video import FileVideoStream
import torch
import cv2
from torchvision import transforms
from torch.autograd import Variable

from clams.app import ClamsApp
from clams.restify import Restifier
from mmif.vocabulary import AnnotationTypes, DocumentTypes
from mmif import Mmif


APP_VERSION = 0.1
class SlateDetection(ClamsApp):
    def _appmetadata(self):
        metadata = {
            "name": "Slate Detection",
            "description": "This tool detects slates.",
            "vendor": "Team CLAMS",
            "iri": f"http://mmif.clams.ai/apps/slatedetect/{APP_VERSION}",
            "app": f"http://mmif.clams.ai/apps/slatedetect/{APP_VERSION}",
            "requires": [DocumentTypes.VideoDocument.value],
            "produces": [AnnotationTypes.TimeFrame.value]
        }
        return metadata

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(os.path.join("data", "slate_model.pth"), map_location=torch.device('cpu'))
        self.model.eval()
        super().__init__()

    def _annotate(self, mmif: Mmif, **kwargs):
        logging.debug(f"loading document with type: {DocumentTypes.VideoDocument.value}")
        video_filename = mmif.get_document_location(DocumentTypes.VideoDocument.value)
        logging.debug(f"video_filename: {video_filename}")
        slate_output = self.run_slatedetection(
            video_filename, mmif, **kwargs
        )
        new_view = mmif.new_view()
        new_view.metadata.set_additional_property("parameters", kwargs.copy())
        new_view.metadata['app'] = self.metadata["iri"]
        for _id, frames in enumerate(slate_output):
            start_frame, end_frame = frames
            timeframe_annotation = new_view.new_annotation(f"tf{_id}", AnnotationTypes.TimeFrame)
            timeframe_annotation.add_property("start", int(start_frame))
            timeframe_annotation.add_property("end", int(end_frame))
            timeframe_annotation.add_property("unit", "msec")
            timeframe_annotation.add_property("frameType", "slate")
        return mmif


    def run_slatedetection(self, video_filename, mmif=None, **kwargs):
        image_transforms = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor()]
        )
        sample_ratio = int(kwargs.get('sampleRatio', 30))
        min_duration = int(kwargs.get('minFrameCount', 10))
        stop_after_one = kwargs.get('stopAfterOne', False)
        stop_at = int(kwargs.get('stopAt', 30*60*60*5)) # default 5 hours

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
        slate_result = []
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
                        slate_result.append((start_frame, counter))
                break
            if counter % sample_ratio == 0:
                result = frame_is_slate(frame)
                if result:  # in slate
                    if not in_slate:
                        in_slate = True
                        start_frame = counter
                else:
                    if in_slate:
                        in_slate = False
                        if counter - start_frame > min_duration:
                            slate_result.append((start_frame, counter))
                        if stop_after_one:
                            return slate_result
            counter += 1
        return slate_result


if __name__ == "__main__":
    slate_tool = SlateDetection()
    slate_service = Restifier(slate_tool)
    slate_service.run()
