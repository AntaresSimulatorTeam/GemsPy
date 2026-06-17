# type:ignore


def full_workflow():

    for s in scenarios:
        # Initial data
        get_network()
        get_data()

        for component in get_hydro_component():
            # Annual part of hydro heuristic
            get_heuristic_model()
            get_heuristic_data()
            build_heuristic_problem()
            solve()
            get_output()
            set_data()

            # Monthly part of hydro heuristic
            for month in range(12):
                get_heuristic_model()
                get_heuristic_data()
                build_heuristic_problem()
                solve()
                get_output()
                set_data()

            # Weekly targets of hydro heuristic
            set_data()

        for w in weeks:
            # Iteration 1
            build_main_problem()
            solve()
            get_output()
            set_data()

            # Thermal heuristic
            for component in get_thermal_component():
                get_heuristic_model()
                build_heuristic_problem()
                solve()
                get_output()
                set_data()

            # Iteration 2
            build_main_problem()
            solve()


class ResolutionWorkflow:

    def get_network(component_file, model_file):
        """Initial network (with components and models) used for main resolution steps.
        @param component_file : description of all components, their model used during main resolution steps and their connections
        @param model_file : description of all models
        @output : network"""
        pass

    def get_data(component_file, data_dir):
        """Initial database that will be used for all resolutions steps.
        @param component_file : description of all components, their model used during main resolution steps and their connections
        @param data_dir : where to look for complex data
        @output : database"""
        pass

    def get_heuristic_model(component, model_file):
        """Get model used for heuristic steps, we supposed that we have one main model for each component and an undefined number of heuristic models (zero, one or more) for each component.
        @param component : component concerned by the heuristic
        @param model_file : description of all models
        @output : model"""
        pass

    def get_heuristic_data(component, data_transformation_functions):
        """Get data used only for heuristics. It could be heuristic parameters or combination of parameters of the database (agregation of residual load for example)
        @param component : component concerned by the heuristic
        @param data_transformation_functions : description of data transformations to perform
        """
        pass

    def build_heuristic_problem(component, heuristic_model, database):
        """An heuristic problem is usally composed of only one composant in the network.
        @param component : component concerned by the heuristic
        @param heuristic_model : model to use for the heuristic
        @param database : database to use for the heuristic
        @output optimization problem"""
        pass

    def build_main_problem(network, database):
        """Main problem built with the whole network and the current database (correspond to weekly problems solved in iteration 1 and 2)
        @param network : network with components and their models used during main resolution steps
        @param database : current database
        @output : optimization problem"""
        pass

    def solve(problem):
        """Solve optimization problems
        @param problem : could be an heuristic problem or a main problem"""
        pass

    def get_output(problem):
        """Get optimal solution of relevant variables
        @param problem : solved problem
        @output : output data"""
        pass

    def set_data(output, database, data_update_functions):
        """Set data for further optimization problems based on previous resolution
        @param output : from previous solved problem
        @param database : current database to update
        @param data_update_functions : description of operations to perform on output data to update the database
        """
        pass

    def get_thermal_component():
        pass

    def get_hydro_component():
        pass
