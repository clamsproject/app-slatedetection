"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes
from clams.appmetadata import AppMetadata
import re 

APP_VERSION = 0.2
# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    # first set up some basic information
    metadata = AppMetadata(
        name="Slate Detection",
        description="This tool detects slates.",  # briefly describe what the purpose and features of the app
        app_license="MIT",
        identifier="slatedetection",  # should be a single string without whitespaces. If you don't intent to publish this app to the CLAMS app-directory, please use a full IRI format. 
        url="https://github.com/clams-project/app-slatedetection", 
        
        # this trick can also be useful (replace ANALYZER_NAME with the pypi dist nam
        analyzer_version =[l.strip().rsplit('==')[-1] for l in open('requirements.txt').readlines() if re.match(r'^torchvision>=', l)][0],
        analyzer_license="BSD"  # short name for a software license
    )
    # and then add I/O specifications: an app must have at least one input and ont output
    metadata.add_input(DocumentTypes.VideoDocument)
    metadata.add_output(AnnotationTypes.TimeFrame, properties={"frameType":"string"})
    
    # (optional) and finally add runtime parameter specifications
    metadata.add_parameter(name="timeUnit",
                           description="Unit for output typeframe",
                           type="string",
                           choices= ["frames","milliseconds"],
                           default="frames")
    
    metadata.add_parameter(name="sampleRatio",
                          description="Frequency to sample frames.",
                          type="integer",
                          default=30)
    
    metadata.add_parameter(name="stopAt",
                           description="Frame number to stop processing",
                           type="integer",
                           default=30*60*60*5)
    
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
                           default=0.5)
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    sys.stdout.write(appmetadata().jsonify(pretty=True))
