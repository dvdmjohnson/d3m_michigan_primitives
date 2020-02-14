from .BaseDataset import BaseDataset


class AutoPriceMinMetadataDataset(BaseDataset):

    @staticmethod
    def get_dataset_name():
        """Returns the proper name of this dataset as it appears the dataset repository.

        :return: str
        """
        return 'LL0_207_autoPrice_MIN_METADATA'
