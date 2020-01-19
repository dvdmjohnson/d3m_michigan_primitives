# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 20:06:16 2018

@author: shanthakumar
"""
from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import ArgumentType, Context

from spider.pipelines.base import BasePipeline
from spider.cluster.ssc_omp import SSC_OMP
from d3m.primitives.data_transformation.dataframe_to_ndarray import Common as DataFrameToNDArrayPrimitive
from d3m.primitives.data_transformation.ndarray_to_dataframe import Common as NDArrayToDataFramePrimitive
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
from d3m.primitives.data_transformation.column_parser import Common as ColumnParserPrimitive
from d3m.primitives.data_transformation.extract_columns_by_semantic_types import Common as ExtractColumnsBySemanticTypesPrimitive
from d3m.primitives.data_transformation.construct_predictions import Common as ConstructPredictionsPrimitive


class SSCOMPOneHundredPlantsMarginPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        
        #choose one or more seed datasets on which this pipeline can operate
        self.dataset = '1491_one_hundred_plants_margin'
        self.meta_info = self.genmeta(self.dataset)
        
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
        
        #Dataframe -> Column Parsing
        step_1 = meta_pipeline.PrimitiveStep(primitive_description = ColumnParserPrimitive.metadata.query())
        step_1.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.0.produce'
        )
        step_1.add_output('produce')
        pipeline.add_step(step_1)
        
        # Column -> Column Extract attributes
        step_2 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_2.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.1.produce')
        step_2.add_output('produce')
        step_2.add_hyperparameter(
                name='semantic_types',
                argument_type=ArgumentType.VALUE,
                data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
        pipeline.add_step(step_2)
        
        # Attribute Dataframe -> NDArray
        step_3 = meta_pipeline.PrimitiveStep(primitive_description = DataFrameToNDArrayPrimitive.metadata.query())
        step_3.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.2.produce' #inputs here are the outputs from step 0
        )
        step_3.add_output('produce')
        pipeline.add_step(step_3)
        
        # NDARRAY -> SSC_OMP
        step_4 = meta_pipeline.PrimitiveStep(primitive_description = SSC_OMP.metadata.query())
        step_4.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.3.produce' #same as step 0
        )
        step_4.add_hyperparameter(
                name='n_clusters',
                argument_type=ArgumentType.VALUE,
                data=100
        )
        step_4.add_hyperparameter(
                name='sparsity_level',
                argument_type=ArgumentType.VALUE,
                data=5
        )
        step_4.add_output('produce')
        pipeline.add_step(step_4)
        
        # SSC_OMP -> Dataframe
        step_5 = meta_pipeline.PrimitiveStep(primitive_description = NDArrayToDataFramePrimitive.metadata.query())
        step_5.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.4.produce' #inputs here are the outputs from step 4
        )
        step_5.add_output('produce')
        pipeline.add_step(step_5)
        
        # Dataframe -> combine with original
        step_6 = meta_pipeline.PrimitiveStep(primitive_description = ConstructPredictionsPrimitive.metadata.query())
        step_6.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.5.produce' #inputs here are the prediction column
        )
        step_6.add_argument(
                name = 'reference',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.0.produce' #inputs here are the dataframed input dataset
        )
        step_6.add_output('produce')
        pipeline.add_step(step_6)
        
        # Final Output
        pipeline.add_output(
                name='output',
                data_reference='steps.6.produce')
        
        return pipeline
        
