from ..modified_base_model import ModifiedBaseModel


class GemsPortConnection(ModifiedBaseModel):
    component1: str
    port1: str
    component2: str
    port2: str
