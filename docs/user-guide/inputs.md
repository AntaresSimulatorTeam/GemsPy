# Reading input files with GemsPy


## Loading the libraries & the system 



~~~ python
with open("system_example.yml") as compo_file:
    input_system = parse_yaml_components(compo_file)

with open("simple_library.yml") as lib_file:
    input_libraries = [parse_yaml_library(lib_file)]

result_lib = resolve_library(input_libraries)
components_input = resolve_system(input_system, result_lib)
database = build_data_base(input_system, Path(series_dir))
~~~

## Loading timeseries data

~~~ python
database = build_data_base(input_system, Path(series_dir))
~~~