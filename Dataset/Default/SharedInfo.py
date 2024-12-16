from Dataset.Dataset import SharedInfo
from Dataset.constants import *

# Any other default SharedInfo objects should be defined here. The superclass defined in Dataset.py should be pretty universal though
# Used to get true efficiency plots where we count all the gen muons! -- Dont try to use any TrainingVariables except GeneratorVariables with this
class AllEventSharedInfo(SharedInfo):
    def __init__(self):
        pass
    
    def calculate(self, event, track):
        pass