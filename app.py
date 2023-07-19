import argparse
import logging
import os
from typing import Union

import PIL
import cv2
import torch
from clams import ClamsApp, Restifier
from mmif import Mmif, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh
from torch.autograd import Variable
from torchvision import transforms


class Slatedetection(ClamsApp):

    def __init__(self):
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)
        self.model = torch.load(
            os.path.join("data", "slate_model.pth"), map_location=torch.device(dev)
        )
        self.model.eval()
        super().__init__()

    def _appmetadata(self):
        # see metadata.py
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        
        logging.debug(f"loading documents with type: {DocumentTypes.VideoDocument}")
        new_view = mmif.new_view()
        self.sign_view(new_view, parameters)
        vds = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if vds:
            vd = vds[0]
        else:
            return mmif
        # fill the params dict with the default values if not provided
        conf = self.get_configuration(**parameters)
        unit = conf["timeUnit"]
        new_view.new_contain(
            AnnotationTypes.TimeFrame,
            timeUnit=unit,
            document=vd.id
        )
        for slate in self.run_slatedetection(vd, **conf):
            start_frame, end_frame = slate
            timeframe_annotation = new_view.new_annotation(AnnotationTypes.TimeFrame)
            timeframe_annotation.add_property("start", vdh.convert(start_frame, 'f', unit, vd.get_property("fps")))
            timeframe_annotation.add_property("end", vdh.convert(end_frame, 'f', unit, vd.get_property("fps")))
            timeframe_annotation.add_property("frameType","slate")
        return mmif

    def run_slatedetection(self, vd, **parameters):
        video_filename = vd.location_path()
        logging.debug(f"video_filename: {video_filename}")
        image_transforms = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor()]
        )

        def frame_is_slate(frame_, _threshold=parameters["threshold"]):
            image_tensor = image_transforms(PIL.Image.fromarray(frame_)).float()
            image_tensor = image_tensor.unsqueeze_(0)
            input = Variable(image_tensor)
            input = input.to(self.device)
            output = self.model(input)
            output = torch.nn.functional.softmax(output, dim=1)
            output = output.data.cpu().numpy()[0]
            return output.data[1] > _threshold

        cap = vdh.capture(vd)
        frames_to_test = vdh.sample_frames(0, parameters['stopAt'], parameters['sampleRatio'])
        logging.debug(f"frames_to_test: {frames_to_test}")
        found_slates = []
        in_slate = False
        start_frame = None
        cur_frame = frames_to_test[0]
        for cur_frame in frames_to_test:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame - 1)
            ret, frame = cap.read()
            if not ret:
                break
            logging.debug(f"cur_frame: {cur_frame}, slate? : {frame_is_slate(frame)}")
            if frame_is_slate(frame):
                if not in_slate:
                    in_slate = True
                    start_frame = cur_frame
            else:
                if in_slate:
                    in_slate = False
                    if cur_frame - start_frame > parameters['minFrameCount']:
                        found_slates.append((start_frame, cur_frame))
                    if parameters['stopAfterOne']:
                        return found_slates
        if in_slate:
            if cur_frame - start_frame > parameters['minFrameCount']:
                found_slates.append((start_frame, cur_frame))
        return found_slates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", action="store", default="5000", help="set port to listen"
    )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    # create the app instance
    app = Slatedetection()

    http_app = Restifier(app, port=int(parsed_args.port)
    )
    if parsed_args.production:
        http_app.serve_production()
    else:
        http_app.run()
