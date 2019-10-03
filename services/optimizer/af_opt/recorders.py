import h5py

from openmdao.recorders.case_recorder import CaseRecorder

from differential_evolution import DifferentialEvolutionDriver


class PopulationReporter(CaseRecorder):
    def record_metadata_system(self, recording_requester):
        pass

    def record_metadata_solver(self, recording_requester):
        pass

    def record_iteration_system(self, recording_requester, data, metadata):
        pass

    def record_iteration_solver(self, recording_requester, data, metadata):
        pass

    def record_iteration_problem(self, recording_requester, data, metadata):
        pass

    def record_derivatives_driver(self, recording_requester, data, metadata):
        pass

    def record_viewer_data(self, model_viewer_data):
        pass

    def record_iteration_driver(self, recording_requester, data, metadata):
        assert isinstance(recording_requester, DifferentialEvolutionDriver)
        de = recording_requester.get_de()
        with h5py.File(f"{de.generation}.hdf5", "w") as f:
            f.create_dataset("pop", data=de.pop)
            f.create_dataset("fit", data=de.fit)
