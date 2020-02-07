from abc import abstractmethod

class BaseDataset(object):

    @staticmethod
    @abstractmethod
    def get_dataset_name():
        """Returns the proper name of this dataset as it appears the dataset repository.

        :return: str
        """
        raise NotImplementedError

    @staticmethod
    def get_fit_score_command_template():
        """Returns the template for the command used to run a pipeline on this dataset.

        This method gets used with str.format() to generate the pipeline command. The returned string should contain
        these fields:
        - {version} (the d3m package version)
        - {primitive} (the fully-qualified path of the primitive being evaluated by the pipeline)
        - {pipeline} (the name of the BasePipeline object)
        - {instanceid} (the hash of the BasePipeline object, generated automatically by d3m)
        - {dataset} (the proper name of this dataset as it appears the dataset repository)

        For most datasets, this method does not need to be overridden. However, there are some datasets that require a
        slightly different template, and thus require this method to be overridden.

        :return: str
        """
        template = 'python3 -m d3m runtime -v /volumes fit-score' \
                   ' -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json' \
                   ' -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json' \
                   ' -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json' \
                   ' -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json' \
                   ' -a /datasets/seed_datasets_current/{dataset}/SCORE/dataset_SCORE/datasetDoc.json' \
                   ' -o /dev/null' \
                   ' -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml' \
                   ' > pipeline_results/{pipeline}.txt'

        return template
