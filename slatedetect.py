import cv2
import pytesseract
import re

from clams.serve import ClamApp
from clams.serialize import *
from clams.vocab import AnnotationTypes
from clams.vocab import MediaTypes
from clams.restify import Restifier

class SlateDetection(ClamApp):

    def appmetadata(self):
        metadata = {"name": "Slate Detection",
                    "description": "This tool detects slates based on ... ",
                    "vendor": "Team CLAMS",
                    "requires": [MediaTypes.V],
                    "produces": [AnnotationTypes.OCR]}
        return metadata

    def sniff(self, mmif):
        # this mock-up method always returns true
        return True

    def annotate(self, mmif_json):
        mmif = Mmif(mmif_json)
        video_filename = mmif.get_medium_location(MediaTypes.V)
        slate_output = self.run_slatedetection(video_filename, mmif_json) #slate_output is a list of (start frame#, end frame#)

        new_view = mmif.new_view()
        contain = new_view.new_contain(AnnotationTypes.SD)
        contain.producer = str(self.__class__)

        for int_id, (start_frame, end_frame) in enumerate(slate_output):
            annotation = new_view.new_annotation(int_id)
            annotation.start = str(start_frame)
            annotation.end = str(end_frame)
            #annotation.feature = {'conf':confidence}
            annotation.attype = AnnotationTypes.SD

        for contain in new_view.contains.keys():
            mmif.contains.update({contain: new_view.id})
        return mmif

    @staticmethod
    def run_slatedetection(video_filename, mmif, stop_after_one=True):
        sample_ratio = 30

        def process_image(f):
            proc = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            proc = cv2.bitwise_not(proc)
            proc = cv2.threshold(proc, 0, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            return proc

        def frame_is_slate(frame):
            res = pytesseract.image_to_string(frame)
            lines = [l for l in res.split('\n') if len(l.strip()) > 0]
            if len(lines) < 3:
                return False
            elif len([l for l in lines if bool(re.search(r"\d", l))]) > 1:
                #print (res)
                return True
            else:
                return False

        cap = cv2.VideoCapture(video_filename)
        counter = 0
        slate_result = []
        in_slate = False
        start_frame = None
        while cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            if counter % sample_ratio == 0:
                processed_frame = process_image(f)
                result = frame_is_slate(processed_frame)
                if (result): #in slate
                    if not in_slate:
                        in_slate = True
                        start_frame = counter
                else:
                    if (in_slate):
                        in_slate = False
                        if (counter-start_frame > 59):
                            slate_result.append((start_frame, counter))
                        if stop_after_one:
                            return slate_result
            counter += 1
        return slate_result

if __name__ == "__main__":
    slate_tool = SlateDetection()
    slate_service = Restifier(slate_tool)
    slate_service.run()

