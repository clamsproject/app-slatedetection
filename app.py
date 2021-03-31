import os
import PIL
from imutils.video import FileVideoStream
import torch
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
            "requires": [DocumentTypes.VideoDocument],
            "produces": [AnnotationTypes.TimeFrame]
        }
        return metadata

    def setupmetadata(self):
        return None

    def _annotate(self, mmif: Mmif, **kwargs):
        video_filename = mmif.get_document_location(DocumentTypes.VideoDocument.value)
        slate_output = self.run_slatedetection(
            video_filename, mmif, **kwargs
        )
        new_view = mmif.new_view()
        new_view.metadata.set_additional_property("parameters", kwargs.copy())
        new_view.metadata['app'] = self.metadata["iri"]
        for _id, frames in enumerate(slate_output):
            start_frame, end_frame = frames
            timeframe_annotation = new_view.new_annotation(f"tf{_id}", AnnotationTypes.TimeFrame)
            timeframe_annotation.add_property("start", start_frame)
            timeframe_annotation.add_property("end", end_frame)
            timeframe_annotation.add_property("unit", "frame")
            timeframe_annotation.add_property("frameType", "slate")
        return mmif

    @staticmethod
    def run_slatedetection(video_filename, mmif=None, stop_after_one=True):
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
            input = input.to(device)
            output = model(input)
            index = output.data.cpu().numpy().argmax()
            return index == 1

        fvs = FileVideoStream(video_filename).start()
        counter = 0
        slate_result = []
        in_slate = False
        start_frame = None
        while fvs.running():
            frame = fvs.read()
            if frame is None:
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
