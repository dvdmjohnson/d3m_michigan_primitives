from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import ArgumentType, Context

import spider.pipelines.datasets
from spider.pipelines.base import BasePipeline
from spider.supervised_learning.owl import OWLRegression
from d3m.primitives.data_transformation.dataframe_to_ndarray import Common as DataFrameToNDArrayPrimitive
from d3m.primitives.data_transformation.ndarray_to_dataframe import Common as NDArrayToDataFramePrimitive
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
from d3m.primitives.data_transformation.column_parser import DataFrameCommon as ColumnParserPrimitive
from d3m.primitives.data_transformation.construct_predictions import DataFrameCommon as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.extract_columns_by_semantic_types import DataFrameCommon as ExtractColumnsBySemanticTypesPrimitive
from d3m.primitives.data_preprocessing.one_hot_encoder import MakerCommon as OneHotMaker


class OWLRegressionPipeline(BasePipeline):
    def __init__(self):
        super().__init__()

        #specify one seed dataset on which this pipeline can operate
        self.dataset = 'LL0_207_autoPrice'
        self.meta_info = {
                'problem': spider.pipelines.datasets.get_problem_id(self.dataset),
                'full_inputs': [ spider.pipelines.datasets.get_full_id(self.dataset) ],
                'train_inputs': [ spider.pipelines.datasets.get_train_id(self.dataset) ],
                'test_inputs': [ spider.pipelines.datasets.get_problem_id(self.dataset) ],
                'score_inputs': [ spider.pipelines.datasets.get_problem_id(self.dataset) ],
            }

    #define pipeline object
    def _gen_pipeline(self):
        #pipeline context is just metadata, ignore for now
        pipeline = meta_pipeline.Pipeline()
        #define inputs.  This will be read in automatically as a Dataset object.
        pipeline.add_input(name = 'inputs')

        #step 0: Dataset -> Dataframe
        step_0 = meta_pipeline.PrimitiveStep(primitive_description = DatasetToDataFramePrimitive.metadata.query())
        step_0.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'inputs.0')
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        #step 1: ColumnParser
        step_1 = meta_pipeline.PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
        step_1.add_argument(
                name='inputs',
                argument_type=ArgumentType.CONTAINER,
                data_reference='steps.0.produce')
        step_1.add_output('produce')
        pipeline.add_step(step_1)

        #step 2: one-hot encoding for discrete features.
        step_2 = meta_pipeline.PrimitiveStep(primitive_description = OneHotMaker.metadata.query())
        step_2.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.1.produce'
        )
        step_2.add_output('produce')
        pipeline.add_step(step_2)
        
        #step 3: Extract attributes from dataset into a dedicated dataframe
        step_3 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_3.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.2.produce')
        step_3.add_output('produce')
        step_3.add_hyperparameter(
                name='semantic_types',
                argument_type=ArgumentType.VALUE,
                data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
        pipeline.add_step(step_3)

        #step 4: Extract Targets
        step_4 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
        step_4.add_output('produce')
        step_4.add_hyperparameter(
                name='semantic_types',
                argument_type=ArgumentType.VALUE,
                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
        pipeline.add_step(step_4)
        
        #step 5: transform attributes dataframe into an ndarray
        step_5 = meta_pipeline.PrimitiveStep(primitive_description = DataFrameToNDArrayPrimitive.metadata.query())
        step_5.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.3.produce'
        )
        step_5.add_output('produce')
        pipeline.add_step(step_5)

        #step 6: transform targets dataframe into an ndarray
        step_6 = meta_pipeline.PrimitiveStep(primitive_description = DataFrameToNDArrayPrimitive.metadata.query())
        step_6.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.4.produce'
        )
        step_6.add_output('produce')
        pipeline.add_step(step_6)
        
        attributes = 'steps.5.produce'
        targets    = 'steps.6.produce'

        #step 7: OWLRegression
        step_7 = meta_pipeline.PrimitiveStep(primitive_description=OWLRegression.metadata.query())
        step_7.add_hyperparameter(
                name='normalize',
                argument_type=ArgumentType.VALUE,
                data=True)
        step_7.add_hyperparameter(
                name='learning_rate',
                argument_type=ArgumentType.VALUE,
                data=1e-1)
        step_7.add_hyperparameter(
                name='tol',
                argument_type=ArgumentType.VALUE,
                data=1e-5)
        step_7.add_argument(
                name='inputs',
                argument_type=ArgumentType.CONTAINER,
                data_reference=attributes)
        step_7.add_argument(
                name='outputs',
                argument_type=ArgumentType.CONTAINER,
                data_reference=targets)
        step_7.add_output('produce')
        pipeline.add_step(step_7)

        #step 8: convert numpy-formatted prediction outputs to a dataframe
        step_8 = meta_pipeline.PrimitiveStep(primitive_description = NDArrayToDataFramePrimitive.metadata.query())
        step_8.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.7.produce'
        )
        step_8.add_output('produce')
        pipeline.add_step(step_8)

        #step 9: generate a properly-formatted output dataframe from the dataframed prediction outputs using the input dataframe as a reference
        step_9 = meta_pipeline.PrimitiveStep(primitive_description = ConstructPredictionsPrimitive.metadata.query())
        step_9.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.8.produce' #inputs here are the prediction column
        )
        step_9.add_argument(
                name = 'reference',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.0.produce' #inputs here are the dataframed input dataset
        )
        step_9.add_output('produce')
        pipeline.add_step(step_9)

        # Final Output
        pipeline.add_output(
                name='output',
                data_reference='steps.9.produce')

        return pipeline
