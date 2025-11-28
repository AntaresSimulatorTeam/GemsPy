from typing import List, Optional
from pydantic import Field
from ..modified_base_model import ModifiedBaseModel
from .gems_component import GemsComponent
from .gems_port_connection import GemsPortConnection
from .gems_area_connection import GemsAreaConnection


class GemsSystem(ModifiedBaseModel):
    id: Optional[str] = None
    model_libraries: Optional[str] = None  # Parsed but unused for n
    components: List[GemsComponent] = Field(default_factory=list)
    connections: Optional[List[GemsPortConnection]] = None
    area_connections: Optional[List[GemsAreaConnection]] = None
    nodes: Optional[List[GemsComponent]] = []
