TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD = {
    "thermal": "get_thermals",
    "renewable": "get_renewables",
    "st_storage": "get_st_storages",
}
TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD = {
    "thermal": "delete_thermal_cluster",
    "renewable": "delete_renewable_cluster",
    "st_storage": "delete_st_storage",
}
STUDY_LEVEL_DELETION = {
    "area": "delete_area",
    "link": "delete_link",
    "binding_constraint": "delete_binding_constraint",
}
STUDY_LEVEL_GET = {
    "area": "get_areas",
    "link": "get_links",
    "binding_constraint": "get_binding_constraints",
}
MATRIX_TYPES_TO_SET_METHOD = {
    "load": "set_load",
    "solar": "set_solar",
    "wind": "set_wind",
}


MATRIX_TYPES_TO_GET_METHOD = {
    "load": "get_load_matrix",
    "solar": "get_solar_matrix",
    "wind": "get_wind_matrix",
}
TIMESERIES_NAME_TO_METHOD = {
    "capacity_direct": "get_capacity_direct",
    "capacity_indirect": "get_capacity_indirect",
    "links_parameters": "get_parameters",
    "thermal_series": "get_series_matrix",
    "thermal_co2": "get_co2_cost_matrix",
    "thermal_fuel": "get_fuel_cost_matrix",
    "renewable_series": "get_timeseries",
    "pmax_injection": "get_pmax_injection",
    "pmax_withdrawal": "get_pmax_withdrawal",
    "lower_rule_curve": "get_lower_rule_curve",
    "upper_rule_curve": "get_upper_rule_curve",
    "storage_inflows": "get_storage_inflows",
    "cost_injection": "get_cost_injection",
    "cost_withdrawal": "get_cost_withdrawal",
    "cost_level": "get_cost_level",
    "cost_variation_injection": "get_cost_variation_injection",
    "cost_variation_withdrawal": "get_cost_variation_withdrawal",
}
TEMPLATE_CLUSTER_TYPE_TO_CLUSTER_PATH = {
    "thermal": "thermal",
    "renewable": "renewables",
    "st_storage": "st-storage",
}
TEMPLATE_LINK_TO_TIMESERIES_FILE_TYPE = {
    "links_parameters": "LINKS_PARAMETERS",
    "capacity_direct": "LINKS_CAPACITIES_DIRECT",
    "capacity_indirect": "LINKS_CAPACITIES_INDIRECT",
}
TEMPLATE_TO_TIMESERIES_FILE_TYPE = {
    "pmax_injection": "ST_STORAGE_PMAX_INJECTION",
    "pmax_withdrawal": "ST_STORAGE_PMAX_WITHDRAWAL",
    "lower_rule_curve": "ST_STORAGE_LOWER_RULE_CURVE",
    "upper_rule_curve": "ST_STORAGE_UPPER_RULE_CURVE",
    "storage_inflows": "ST_STORAGE_INFLOWS",
    "cost_injection": "ST_STORAGE_COST_INJECTION",
    "cost_withdrawal": "ST_STORAGE_COST_WITHDRAWAL",
    "cost_level": "ST_STORAGE_COST_LEVEL",
    "cost_variation_injection": "ST_STORAGE_COST_VARIATION_INJECTION",
    "cost_variation_withdrawal": "ST_STORAGE_COST_VARIATION_WITHDRAWAL",
    "thermal_series": "THERMAL_SERIES",
    "series_co2_cost": "THERMAL_CO2",
    "series_fuel_cost": "THERMAL_FUEL",
    "renewable_series": "RENEWABLE_SERIES",
}
