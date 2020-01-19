from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import ArgumentType, Context

from spider.pipelines.base import BasePipeline
from spider.featurization.audio_featurization import AudioFeaturization
from bbn_primitives.time_series import AudioReader
from d3m.primitives.data_transformation.extract_columns_by_semantic_types import Common as ExtractColumnsBySemanticTypesPrimitive
from d3m.primitives.data_transformation.denormalize import Common as DenormalizePrimitive
from d3m.primitives.data_transformation.dataframe_to_ndarray import Common as DataFrameToNDArrayPrimitive
from d3m.primitives.data_transformation.ndarray_to_dataframe import Common as NDArrayToDataFramePrimitive
from d3m.primitives.data_transformation.construct_predictions import Common as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
from sklearn_wrap.SKRandomForestClassifier import SKRandomForestClassifier as RandomForestClassifierPrimitive

class AudioFeaturizationUrbansoundPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        
        #specify one seed dataset on which this pipeline can operate
        self.dataset = '31_urbansound'
        self.meta_info = self.genmeta(self.dataset)

    def get_primitive_entry_point(self):
        return 'd3m.primitives.feature_extraction.i3d.Umich'

    def get_primitive_entry_point(self):
        return 'd3m.primitives.feature_extraction.audio_featurization.Umich'

    def get_fit_score_command_template(self):
        """Override the default template to use a different scoring problem JSON for this particular dataset."""
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

    #define pipeline object
    def _gen_pipeline(self):
        #pipeline context is just metadata, ignore for now
        pipeline = meta_pipeline.Pipeline()
        #define inputs.  This will be read in automatically as a Dataset object.
        pipeline.add_input(name = 'inputs')
        
        #denormalize to bring audio data collection into primary dataframe
        step_0 = meta_pipeline.PrimitiveStep(primitive_description = DenormalizePrimitive.metadata.query())
        step_0.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'inputs.0' #i.e. the original inputs
        )
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        #extract raw audio sequence for each training element from the Dataset into a list of ndarrays
        step_1 = meta_pipeline.PrimitiveStep(primitive_description = AudioReader.metadata.query())
        step_1.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'inputs.0' #same as step 1
        )
        step_1.add_output('produce')
        pipeline.add_step(step_1)

        #featurize each audio sequence (list of raw ndarrays to single feature matrix ndarray)
        step_2 = meta_pipeline.PrimitiveStep(primitive_description = AudioFeaturization.metadata.query())
        step_2.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.1.produce' #inputs here are the outputs from step 1
        )
        step_2.add_output('produce')
        pipeline.add_step(step_2)

        #convert audio features to dataframe
        step_3 = meta_pipeline.PrimitiveStep(primitive_description = NDArrayToDataFramePrimitive.metadata.query())
        step_3.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.2.produce'
        )
        step_3.add_output('produce')
        pipeline.add_step(step_3)

        #denormalize to bring audio data collection into primary dataframe
        step_4 = meta_pipeline.PrimitiveStep(primitive_description = DenormalizePrimitive.metadata.query())
        step_4.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'inputs.0' #i.e. the original inputs
        )
        step_4.add_output('produce')
        pipeline.add_step(step_4)

        #transform denormalized dataset into a dataframe
        step_5 = meta_pipeline.PrimitiveStep(primitive_description = DatasetToDataFramePrimitive.metadata.query())
        step_5.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.0.produce' #inputs here are the original dataset
        )
        step_5.add_output('produce')
        pipeline.add_step(step_5)

        #Extract target
        step_6 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_6.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.5.produce')
        step_6.add_output('produce')
        step_6.add_hyperparameter(
                name='semantic_types',
                argument_type=ArgumentType.VALUE,
                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
        pipeline.add_step(step_6)

        #classify the featurized outputs (feature matrix dataframe to class prediction vector dataframe)
        step_7 = meta_pipeline.PrimitiveStep(primitive_description = RandomForestClassifierPrimitive.metadata.query())
        step_7.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.3.produce'
        )
        step_7.add_argument(
                name = 'outputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.6.produce'
        )
        step_7.add_output('produce')
        pipeline.add_step(step_7)
        
        #finally generate a properly-formatted output dataframe from the prediction outputs using the input dataframe as a reference
        step_8 = meta_pipeline.PrimitiveStep(primitive_description = ConstructPredictionsPrimitive.metadata.query())
        step_8.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.7.produce' #inputs here are the prediction column
        )
        step_8.add_argument(
                name = 'reference',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.5.produce' #inputs here are the dataframed input dataset
        )
        step_8.add_output('produce')
        pipeline.add_step(step_8)

        # Adding output step to the pipeline
        pipeline.add_output(name = 'Links', data_reference = 'steps.8.produce')

        return pipeline