from typing import Optional, List
from ..modified_base_model import ModifiedBaseModel
from .gems_component_parameter import GemsComponentParameter


class GemsComponent(ModifiedBaseModel):
    id: str
    model: str
    scenario_group: Optional[str] = None
    parameters: Optional[List[GemsComponentParameter]] = None
