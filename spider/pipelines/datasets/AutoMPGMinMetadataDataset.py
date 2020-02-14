from .BaseDataset import BaseDataset


class AutoMPGMinMetadataDataset(BaseDataset):

    @staticmethod
    def get_dataset_name():
        """Returns the proper name of this dataset as it appears the dataset repository.

        :return: str
        """
        return '196_autoMpg_MIN_METADATA'
