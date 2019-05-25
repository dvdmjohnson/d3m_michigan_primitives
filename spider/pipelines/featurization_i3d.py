from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import ArgumentType, Context

import spider.pipelines.datasets
from spider.pipelines.base import BasePipeline
from spider.featurization.i3d import I3D

from common_primitives.video_reader import VideoReaderPrimitive


from common_primitives.random_forest import RandomForestClassifierPrimitive
from d3m.primitives.data_transformation.ndarray_to_dataframe import Common as NDArrayToDataFramePrimitive
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
from d3m.primitives.data_transformation.extract_columns_by_semantic_types import DataFrameCommon as ExtractColumnsBySemanticTypesPrimitive
from d3m.primitives.classification.random_forest import DataFrameCommon as RandomForestClassifierPrimitive
from d3m.primitives.data_transformation.construct_predictions import DataFrameCommon as ConstructPredictionsPrimitive

class FeaturizationI3DPipeline(BasePipeline):
    def __init__(self):
        super().__init__()

        self.dataset = 'LL1_3476_HMDB_actio_recognition'
        self.meta_info = self.genmeta(self.dataset)

    def _gen_pipeline(self):
        #Creating pipeline
        pipeline = meta_pipeline.Pipeline()
        pipeline.add_input(name = 'inputs')

        #Step 0: Convert Dataset to DataFrame object
        step_0 = meta_pipeline.PrimitiveStep(primitive_description = DatasetToDataFramePrimitive.metadata.query())
        step_0.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'inputs.0'
        )
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        #Step 1: Extract raw video sequence into dataframe and resize to appropriate size
        step_1 = meta_pipeline.PrimitiveStep(primitive_description = VideoReaderPrimitive.metadata.query())
        step_1.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.0.produce' #inputs are from step 0
        )
        step_1.add_output('produce')
        pipeline.add_step(step_1)

        #Step 2: Extract features from videos using I3D
        step_2 = meta_pipeline.PrimitiveStep(primitive_description = I3D.metadata.query())
        step_2.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.1.produce' #inputs are from step 1
        )
        step_2.add_output('produce')
        pipeline.add_step(step_2)

        #Step 3: Transform ndarray output to dataframe
        step_3 = meta_pipeline.PrimitiveStep(primitive_description = NDArrayToDataFramePrimitive.metadata.query())
        step_3.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.2.produce'
        )
        step_3.add_output('produce')
        pipeline.add_step(step_3)

        #Step 4: Extract suggested targets from input dataframe
        step_4 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_4.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.0.produce'
        )
        step_4.add_hyperparameter(
                name='semantic_types',
                argument_type=ArgumentType.VALUE,
                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget']
        )
        step_4.add_output('produce')
        pipeline.add_step(step_4)

        #Step 5: Classify the featurized outputs using Random Forest Classifier
        step_5 = meta_pipeline.PrimitiveStep(primitive_description = RandomForestClassifierPrimitive.metadata.query())
        step_5.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.3.produce'
        )
        step_5.add_argument(
                name = 'outputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.4.produce'
        )
        step_5.add_output('produce')
        pipeline.add_step(step_5)

        #Step 6: Generate a properly-formatted output dataframe from the dataframed prediction outputs using the input dataframe as a reference
        step_6 = meta_pipeline.PrimitiveStep(primitive_description = ConstructPredictionsPrimitive.metadata.query())
        step_6.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.5.produce'
        )
        step_6.add_argument(
                name = 'reference',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.0.produce'
        )
        step_6.add_output('produce')
        pipeline.add_step(step_6)

        #Add output step to the pipeline
        pipeline.add_output(name = 'Links', data_reference = 'steps.6.produce')
        return pipeline 

    def assert_result(self, tester, results, dataset):
        tester.assertEquals(len(results), 1)
        tester.assertEquals(len(results[0]), 1)
