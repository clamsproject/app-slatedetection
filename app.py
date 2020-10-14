import os
import timecode
import PIL
from imutils.video import FileVideoStream
import torch
from torchvision import transforms
from torch.autograd import Variable

from clams.serve import ClamsApp
from clams.restify import Restifier
from mmif.vocabulary import AnnotationTypes, DocumentTypes
from mmif import Mmif, Annotation


class SlateDetection(ClamsApp):
    def appmetadata(self):
        metadata = {
            "name": "Slate Detection",
            "description": "This tool detects slates.",
            "vendor": "Team CLAMS",
            "requires": [DocumentTypes.VideoDocument],
            "produces": [AnnotationTypes.TimeFrame],
        }
        return metadata

    def setupmetadata(self):
        return None

    def sniff(self, mmif):
        # this mock-up method always returns true
        return True

    def annotate(self, mmif_json):
        mmif = Mmif(mmif_json, validate=False) ##todo 2020-10-14 kelleyl remove validate=False with new version of sdk
        video_filename = mmif.get_document_location(DocumentTypes.VideoDocument.value)
        slate_output = self.run_slatedetection(
            video_filename, mmif_json
        )

        new_view = mmif.new_view()
        contain = new_view.new_contain(AnnotationTypes.TimeFrame)
        contain.producer = str(self.__class__)
        for start_frame, end_frame in slate_output:
            annotation = Annotation()
            annotation.add_property("start", start_frame)
            annotation.add_property("end", end_frame)
            annotation.at_type = AnnotationTypes.TimeFrame
            new_view.add_annotation(annotation)
        return mmif

    @staticmethod
    def run_slatedetection(video_filename, mmif=None, stop_after_one=True):
        image_transforms = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor()]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(os.path.join("data", "slate_model.pth"))
        model.eval()
        sample_ratio = 1

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
            if counter > (30 * 60 * 5):  ## about 5 minutes
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
                        if counter - start_frame > 59:
                            start_timecode = timecode.Timecode(
                                framerate=30000 / 1001, frames=start_frame + 1
                            )
                            end_timecode = timecode.Timecode(
                                framerate=30000 / 1001, frames=counter + 1
                            )
                            slate_result.append((start_timecode, end_timecode))
                        if stop_after_one:
                            return slate_result
            counter += 1
        return slate_result


if __name__ == "__main__":
    slate_tool = SlateDetection()
    slate_service = Restifier(slate_tool)
    slate_service.run()
