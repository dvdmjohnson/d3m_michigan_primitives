import abc
import json

class BasePipeline(object):
    def __init__(self):
        self._pipeline = self._gen_pipeline()

    def get_dataset_class(self):
        """Returns the dataset class, i.e., which dataset to run this pipeline on.

        :return: BasePipeline class
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_primitive_entry_point(self):
        """Returns the qualified name of the primitive that this pipeline is evaluating.

        Example: d3m.primitives.regression.owl_regression.Umich

        :return: str
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _gen_pipeline(self):
        """Create a D3M pipeline for this class."""
        raise NotImplementedError

    @abc.abstractmethod
    def assert_result(self, tester, results, dataset):
        """Make sure that the results from an invocation of this pipeline are valid."""
        raise NotImplementedError

    def get_id(self):
        return self._pipeline.id
        
    def get_json(self):
        # Make it pretty.
        return json.dumps(json.loads(self._pipeline.to_json()), indent = 4)
