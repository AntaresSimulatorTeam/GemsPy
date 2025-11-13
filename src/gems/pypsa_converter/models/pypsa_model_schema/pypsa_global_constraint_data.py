from dataclasses import dataclass


@dataclass
class PyPSAGlobalConstraintData:
    pypsa_name: str
    # pypsa_investment_period
    pypsa_carrier_attribute: str
    pypsa_sense: str
    pypsa_constant: float
    gems_model_id: str  # gems model for this GlobalConstraint
    gems_port_id: str  # gems port for this GlobalConstraint
    gems_components_and_ports: list[tuple[str, str]]
