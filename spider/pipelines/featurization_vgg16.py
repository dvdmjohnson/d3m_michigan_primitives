from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import ArgumentType, Context

import spider.pipelines.datasets
from spider.pipelines.base import BasePipeline
from spider.featurization.vgg16 import VGG16
from d3m.primitives.data_transformation.dataframe_to_ndarray import Common as DataFrameToNDArrayPrimitive
from d3m.primitives.data_transformation.ndarray_to_dataframe import Common as NDArrayToDataFramePrimitive
from common_primitives.dataframe_image_reader import DataFrameImageReaderPrimitive
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
from d3m.primitives.data_transformation.construct_predictions import DataFrameCommon as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.extract_columns_by_semantic_types import DataFrameCommon as ExtractColumnsBySemanticTypesPrimitive
from sklearn_wrap.SKLinearSVR import SKLinearSVR


class FeaturizationVGG16Pipeline(BasePipeline):
    def __init__(self):
        super().__init__()

        self.dataset = '22_handgeometry'
        self.meta_info = {
                'problem': spider.pipelines.datasets.get_problem_id(self.dataset),
                'full_inputs': [ spider.pipelines.datasets.get_full_id(self.dataset) ],
                'train_inputs': [ spider.pipelines.datasets.get_train_id(self.dataset) ],
                'test_inputs': [ spider.pipelines.datasets.get_problem_id(self.dataset) ],
                'score_inputs': [ spider.pipelines.datasets.get_problem_id(self.dataset) ],
            }

    def _gen_pipeline(self):
        pipeline = meta_pipeline.Pipeline()
        pipeline.add_input(name = 'inputs')

        step_0 = meta_pipeline.PrimitiveStep(primitive_description = DatasetToDataFramePrimitive.metadata.query())
        step_0.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'inputs.0'
        )
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        #step 1: Read the images into dataframe
        step_1 = meta_pipeline.PrimitiveStep(primitive_description = DataFrameImageReaderPrimitive.metadata.query())
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
                data=['http://schema.org/ImageObject'])
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
                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
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
        
        #step 5: use VGG16 to featurize raw image data
        step_5 = meta_pipeline.PrimitiveStep(primitive_description=VGG16.metadata.query())
        step_5.add_argument(
                name='inputs',
                argument_type=ArgumentType.CONTAINER,
                data_reference='steps.4.produce')
        step_5.add_output('produce')
        pipeline.add_step(step_5)
        
        #step 6: convert featurized image data to a dataframe
        step_6 = meta_pipeline.PrimitiveStep(primitive_description=NDArrayToDataFramePrimitive.metadata.query())
        step_6.add_argument(
            name='inputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.5.produce'
        )
        step_6.add_output('produce')
        pipeline.add_step(step_6)

        #step 7: Linear Regression on featurized data
        step_7 = meta_pipeline.PrimitiveStep(primitive_description = SKLinearSVR.metadata.query())
        step_7.add_argument(
        	name = 'inputs',
        	argument_type = ArgumentType.CONTAINER,
        	data_reference = 'steps.6.produce'
        )
        step_7.add_argument(
            name = 'outputs',
            argument_type = ArgumentType.CONTAINER,
            data_reference = 'steps.3.produce'
        )
        step_7.add_output('produce')
        pipeline.add_step(step_7)
        
        #step 8: generate a properly-formatted output dataframe from the prediction outputs using the input dataframe as a reference
        step_8 = meta_pipeline.PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
        step_8.add_argument(
            name='inputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.7.produce'  # inputs here are the prediction column
        )
        step_8.add_argument(
            name='reference',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.0.produce'  # inputs here are the dataframed input dataset
        )
        step_8.add_output('produce')
        pipeline.add_step(step_8)

        # Adding output step to the pipeline
        pipeline.add_output(
            name='output', 
            data_reference='steps.8.produce')

        return pipeline


