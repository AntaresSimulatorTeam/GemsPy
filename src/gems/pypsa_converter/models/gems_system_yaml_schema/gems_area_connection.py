from ..modified_base_model import ModifiedBaseModel


class GemsAreaConnection(ModifiedBaseModel):
    component: str
    port: str
    area: str
