from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import ArgumentType, Context

from spider.pipelines.base import BasePipeline
from spider.unsupervised_learning.grasta import GRASTA
from d3m.primitives.data_transformation.dataframe_to_ndarray import Common as DataFrameToNDArrayPrimitive
from d3m.primitives.data_transformation.ndarray_to_dataframe import Common as NDArrayToDataFramePrimitive
from d3m.primitives.data_transformation.construct_predictions import Common as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
from d3m.primitives.data_transformation.column_parser import Common as ColumnParserPrimitive
from d3m.primitives.data_transformation.construct_predictions import Common as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.extract_columns_by_semantic_types import Common as ExtractColumnsBySemanticTypesPrimitive
from sklearn_wrap.SKLinearSVR import SKLinearSVR
from sklearn_wrap.SKImputer import SKImputer


class GRASTAAutoPricePipeline(BasePipeline):

    #specify one seed dataset on which this pipeline can operate

    def __init__(self):
        super().__init__()
        
        #specify one seed dataset on which this pipeline can operate
        self.dataset = 'LL0_207_autoPrice'
        self.meta_info = self.genmeta(self.dataset)

    def get_primitive_entry_point(self):
        return 'd3m.primitives.data_compression.grasta.Umich'

    #define pipeline object
    def _gen_pipeline(self):
        #pipeline context is just metadata, ignore for now
        pipeline = meta_pipeline.Pipeline()
        #define inputs.  This will be read in automatically as a Dataset object.
        pipeline.add_input(name = 'inputs')

        # Step 0: DatasetToDataFrame
        step_0 = meta_pipeline.PrimitiveStep(primitive_description = DatasetToDataFramePrimitive.metadata.query())
        step_0.add_argument(name='inputs', argument_type = ArgumentType.CONTAINER, data_reference='inputs.0')
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        # Step 1: ColumnParser
        step_1 = meta_pipeline.PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
        step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
        step_1.add_output('produce')
        pipeline.add_step(step_1)

        # Step 2: Extract Attributes
        step_2 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
        step_2.add_output('produce')
        step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE, data=['https://metadata.datadrivendiscovery.org/types/Attribute'] )
        #step_2.add_hyperparameter(name='exclude_columns', argument_type=ArgumentType.VALUE, data=[0, 1, 6, 7] )
        pipeline.add_step(step_2)

        # Step 3 impute missing data and nans
        step_3 = meta_pipeline.PrimitiveStep(primitive_description = SKImputer.metadata.query())
        step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
        step_3.add_output('produce')
        step_3.add_hyperparameter(name='use_semantic_types', argument_type=ArgumentType.VALUE, data=True)
        step_3.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE, data='replace')
        pipeline.add_step(step_3)

        # Step 4: Extract Targets
        step_4 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
        step_4.add_output('produce')
        step_4.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE, data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'] )
        pipeline.add_step(step_4)

        #Transform attributes dataframe into an ndarray
        step_5 = meta_pipeline.PrimitiveStep(primitive_description = DataFrameToNDArrayPrimitive.metadata.query())
        step_5.add_argument(
            name = 'inputs',
            argument_type = ArgumentType.CONTAINER,
            data_reference = 'steps.3.produce' #inputs here are the outputs from step 3
        )
        step_5.add_output('produce')
        pipeline.add_step(step_5)

        #Run GRASTA
        step_6 = meta_pipeline.PrimitiveStep(primitive_description = GRASTA.metadata.query())
        step_6.add_argument(
            name = 'inputs',
            argument_type = ArgumentType.CONTAINER,
            data_reference = 'steps.5.produce' #inputs here are the outputs from step 5
        )
        step_6.add_hyperparameter(name='constant_step', argument_type=ArgumentType.VALUE, data=0.1)
        step_6.add_output('produce')
        pipeline.add_step(step_6)
        
        # convert numpy-formatted attribute data to a dataframe
        step_7 = meta_pipeline.PrimitiveStep(primitive_description=NDArrayToDataFramePrimitive.metadata.query())
        step_7.add_argument(
            name='inputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.6.produce'  # inputs here are the outputs from step 6
        )
        step_7.add_output('produce')
        pipeline.add_step(step_7)

        #Linear Regression on low-rank data (inputs and outputs for sklearns are both dataframes)
        step_8 = meta_pipeline.PrimitiveStep(primitive_description = SKLinearSVR.metadata.query())
        step_8.add_argument(
        	name = 'inputs',
        	argument_type = ArgumentType.CONTAINER,
        	data_reference = 'steps.7.produce'
        )
        step_8.add_argument(
            name = 'outputs',
            argument_type = ArgumentType.CONTAINER,
            data_reference = 'steps.4.produce'
        )
        step_8.add_output('produce')
        pipeline.add_step(step_8)

        #finally generate a properly-formatted output dataframe from the prediction outputs using the input dataframe as a reference
        step_9 = meta_pipeline.PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
        step_9.add_argument(
            name='inputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.8.produce'  # inputs here are the prediction column
        )
        step_9.add_argument(
            name='reference',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.0.produce'  # inputs here are the dataframed input dataset
        )
        step_9.add_output('produce')
        pipeline.add_step(step_9)

        # Adding output step to the pipeline
        pipeline.add_output(
            name='output', 
            data_reference='steps.9.produce')

        return pipeline
