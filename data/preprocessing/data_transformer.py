#Data transformation is a critical part of the data integration process in which raw data is converted into a unified format or structure.
# Data transformation ensures compatibility with target systems and enhances data quality and usability.
from physical_tensor import PhysicalTensor


class FidelityAssigner:
    def __init__(self, config):
        self.config = config
    def assign(self, sources:list[PhysicalTensor]):
        pass
    def compute_score(self,pt:PhysicalTensor):
        pass
    def validate(self,pt:PhysicalTensor):
        return True


class DataTransformer(object):
    """docstring for DataTransformer"""