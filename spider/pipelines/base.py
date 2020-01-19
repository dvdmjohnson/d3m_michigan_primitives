import abc
import json

class BasePipeline(object):
    def __init__(self):
        self._pipeline = self._gen_pipeline()

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
        
    def genmeta(self, dataset):
        meta_info = {
            'problem': dataset + "_problem",
            'full_inputs': [ dataset + "_dataset" ],
            'train_inputs': [ dataset + "_dataset_TRAIN" ],
            'test_inputs': [ dataset + "_dataset_TEST" ],
            'score_inputs': [ dataset + "_dataset_SCORE" ]}
        return meta_info

    def get_json(self):
        # Make it pretty.
        return json.dumps(json.loads(self._pipeline.to_json()), indent = 4)

    def get_fit_score_command_template(self):
        """Returns the template for the command used to run this pipeline.

        This method gets used with str.format() to generate the pipeline command. The returned string should contain
        these fields:
        - {version} (the d3m package version)
        - {primitive} (the fully-qualified path of the primitive being evaluated by this pipeline)
        - {pipeline} (the name of this BasePipeline object)
        - {instanceid} (the hash of this BasePipeline object, generated automatically by d3m)
        - {dataset} (the name of the dataset that this pipeline is using for evaluation)

        For most pipelines, this method does not need to be overridden. However, there are some datasets that require a
        slightly different template, and thus require this method to be overridden.

        :return: str
        """
        template = 'python3 -m d3m runtime -v /volumes fit-score' \
                   ' -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json' \
                   ' -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json' \
                   ' -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json' \
                   ' -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json' \
                   ' -a /datasets/seed_datasets_current/{dataset}/SCORE/dataset_TEST/datasetDoc.json' \
                   ' -o /dev/null' \
                   ' -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml' \
                   ' > pipeline_results/{pipeline}.txt'

        return template
