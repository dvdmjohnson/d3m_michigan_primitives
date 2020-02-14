import json
from abc import abstractmethod

class BasePipeline(object):

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
