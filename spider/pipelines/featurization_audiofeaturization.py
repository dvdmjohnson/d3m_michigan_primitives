from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import ArgumentType, Context

import spider.pipelines.datasets
from spider.pipelines.base import BasePipeline
from spider.featurization.audio_featurization import AudioFeaturization
from common_primitives.audio_reader import AudioReaderPrimitive
from d3m.primitives.data_transformation.dataframe_to_ndarray import Common as DataFrameToNDArrayPrimitive
from d3m.primitives.data_transformation.ndarray_to_dataframe import Common as NDArrayToDataFramePrimitive
from common_primitives.horizontal_concat import HorizontalConcatPrimitive
from d3m.primitives.classification.random_forest import DataFrameCommon as RandomForestClassifierPrimitive
from d3m.primitives.data_transformation.construct_predictions import DataFrameCommon as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
from d3m.primitives.data_transformation.denormalize import Common as DenormalizePrimitive
from d3m.primitives.data_transformation.extract_columns_by_semantic_types import DataFrameCommon as ExtractColumnsBySemanticTypesPrimitive



class FeaturizationAudioFeaturizationPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        
        #specify one seed dataset on which this pipeline can operate
        self.dataset = '31_urbansound'
        self.meta_info = {
                'problem': spider.pipelines.datasets.get_problem_id(self.dataset),
                'train_inputs': [ spider.pipelines.datasets.get_train_id(self.dataset) ],
                'test_inputs': [ spider.pipelines.datasets.get_problem_id(self.dataset) ],
            }

    #define pipeline object
    def _gen_pipeline(self):
        pipeline = meta_pipeline.Pipeline(context=Context.TESTING)
        pipeline.add_input(name = 'inputs')

        step_0 = meta_pipeline.PrimitiveStep(primitive_description = DatasetToDataFramePrimitive.metadata.query())
        step_0.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'inputs.0'
        )
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        #step 1: Read the audio into dataframe
        step_1 = meta_pipeline.PrimitiveStep(primitive_description = AudioReaderPrimitive.metadata.query())
        step_1.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.0.produce')
        step_1.add_output('produce')
        pipeline.add_step(step_1)

        #step 2: Extract image
        step_2 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_2.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.1.produce')
        step_2.add_output('produce')
        step_2.add_hyperparameter(
                name='semantic_types',
                argument_type=ArgumentType.VALUE,
                data=['https://metadata.datadrivendiscovery.org/types/AudioObject'])
        pipeline.add_step(step_2)

        #step 3: Extract target
        step_3 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_3.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.1.produce')
        step_3.add_output('produce')
        step_3.add_hyperparameter(
                name='semantic_types',
                argument_type=ArgumentType.VALUE,
                data=['https://metadata.datadrivendiscovery.org/types/SuggestedTarget'])
        pipeline.add_step(step_3)

        #step 4: Dataframe to ndarray on step 2 output as VGG16 needs ndarray input
        step_4 = meta_pipeline.PrimitiveStep(primitive_description = DataFrameToNDArrayPrimitive.metadata.query())
        step_4.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.2.produce'
        )
        step_4.add_output('produce')
        pipeline.add_step(step_4)

        #featurize each audio sequence (list of raw ndarrays to single feature matrix ndarray)
        step_5 = meta_pipeline.PrimitiveStep(primitive_description = AudioFeaturization.metadata.query())
        step_5.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.1.produce' #inputs here are the outputs from step 4
        )

        step_5.add_output('produce')
        pipeline.add_step(step_5)

        #classify the featurized outputs (feature matrix dataframe to class prediction vector dataframe)
        step_6 = meta_pipeline.PrimitiveStep(primitive_description = RandomForestClassifierPrimitive.metadata.query())
        step_6.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.5.produce' #inputs here are the outputs from step 5
        )
        step_6.add_argument(
                name = 'outputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.3.produce' #inputs here are the outputs from step 3
        )
        step_6.add_output('produce')
        pipeline.add_step(step_6)

        #finally generate a properly-formatted output dataframe from the prediction outputs using the input dataframe as a reference
        step_7 = meta_pipeline.PrimitiveStep(primitive_description = ConstructPredictionsPrimitive.metadata.query())
        step_7.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.6.produce' #inputs here are the prediction column
        )
        step_7.add_argument(
                name = 'reference',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.0.produce' #inputs here are the dataframed input dataset
        )
        step_7.add_output('produce')
        pipeline.add_step(step_7)

        # Adding output step to the pipeline
        pipeline.add_output(name = 'Links', data_reference = 'steps.7.produce')

        return pipeline
