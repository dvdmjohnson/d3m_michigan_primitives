import json
from abc import abstractmethod

class BasePipeline(object):

    @staticmethod
    @abstractmethod
    def get_dataset_class():
        """Returns the dataset class, i.e., which dataset to run this pipeline on.

        :return: BasePipeline class
        """
        raise NotImplementedError


    @staticmethod
    @abstractmethod
    def get_primitive_entry_point():
        """Returns the qualified name of the primitive that this pipeline is evaluating.

        Example: d3m.primitives.regression.owl_regression.Umich

        :return: str
        """
        raise NotImplementedError


    @staticmethod
    @abstractmethod
    def _gen_pipeline():
        """Create a D3M pipeline for this class."""
        raise NotImplementedError


    @classmethod
    def get_id(cls):
        return cls._gen_pipeline().id


    @classmethod
    def get_json(cls):
        # Make it pretty.
        return json.dumps(json.loads(cls._gen_pipeline().to_json()), indent=4)
