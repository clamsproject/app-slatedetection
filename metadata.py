"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes
from clams.appmetadata import AppMetadata
from clams.app import ClamsApp


# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    
    metadata = AppMetadata(
        name="Slate Detection",
        description="This tool detects slates.",
        app_license="MIT",
        identifier="slatedetection",
        url="https://github.com/clamsproject/app-slatedetection", 
    )
    metadata.add_input(DocumentTypes.VideoDocument)
    metadata.add_output(AnnotationTypes.TimeFrame, properties={"frameType": "slate"})
    
    metadata.add_parameter(name="timeUnit",
                           description="Unit of time to use in output.",
                           type="string",
                           choices=["frames","seconds", "milliseconds"],
                           default="frames")
    
    metadata.add_parameter(name="sampleRatio",
                           description="Frequency to sample frames.",
                           type="integer",
                           default=30)  # ~1 frame per second
    
    metadata.add_parameter(name="stopAt",
                           description="Frame number to stop processing",
                           type="integer",
                           default=5*60*30)  # ~5 minutes of video at 30fps
    
    metadata.add_parameter(name="stopAfterOne",
                           description="When True, processing stops after first timeframe is found",
                           type="boolean",
                           default=True)
    
    metadata.add_parameter(name="minFrameCount",
                           description="Minimum number of frames required for a timeframe to be included in the output",
                           type="integer",
                           default=10)
    
    metadata.add_parameter(name="threshold",
                           description="Threshold from 0-1, lower accepts more potential slates.",
                           type="number",
                           default=0.7)
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
