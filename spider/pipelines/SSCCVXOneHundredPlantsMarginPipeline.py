# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 20:06:16 2018

@author: shanthakumar
"""
from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import ArgumentType

from spider.pipelines.base import BasePipeline
from spider.cluster.ssc_cvx import SSC_CVX
from d3m.primitives.data_transformation.dataframe_to_ndarray import Common as DataFrameToNDArrayPrimitive
from d3m.primitives.data_transformation.ndarray_to_dataframe import Common as NDArrayToDataFramePrimitive
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
from d3m.primitives.data_transformation.column_parser import Common as ColumnParserPrimitive
from d3m.primitives.data_transformation.extract_columns_by_semantic_types import Common as ExtractColumnsBySemanticTypesPrimitive
from d3m.primitives.data_transformation.construct_predictions import Common as ConstructPredictionsPrimitive
from d3m.primitives.schema_discovery.profiler import Common as SimpleProfilerPrimitive

from .datasets import OneHundredPlantsMarginClustDataset


class SSCCVXOneHundredPlantsMarginPipeline(BasePipeline):

    dataset_class = OneHundredPlantsMarginClustDataset
    primitive_entry_point = 'd3m.primitives.clustering.ssc_cvx.Umich'

    @staticmethod
    def _gen_pipeline():
        #pipeline context is just metadata, ignore for now
        pipeline = meta_pipeline.Pipeline()
        # define inputs.  This will be read in automatically as a Dataset object.
        pipeline.add_input(name='inputs')

        # step 0: Dataset -> Dataframe
        step_0 = meta_pipeline.PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
        step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        # Profiler to infer semantic types
        step_1 = meta_pipeline.PrimitiveStep(primitive_description=SimpleProfilerPrimitive.metadata.query())
        step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
        step_1.add_output('produce')
        pipeline.add_step(step_1)

        # Dataframe -> Column Parsing
        step_2 = meta_pipeline.PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
        step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
        step_2.add_output('produce')
        pipeline.add_step(step_2)

        # Column -> Column Extract attributes
        step_3 = meta_pipeline.PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
        step_3.add_output('produce')
        step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE, data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
        pipeline.add_step(step_3)

        # Attribute Dataframe -> NDArray
        step_4 = meta_pipeline.PrimitiveStep(primitive_description=DataFrameToNDArrayPrimitive.metadata.query())
        step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
        step_4.add_output('produce')
        pipeline.add_step(step_4)

        # NDARRAY -> Cluster
        step_5 = meta_pipeline.PrimitiveStep(primitive_description=SSC_CVX.metadata.query())
        step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
        step_5.add_hyperparameter(name='n_clusters', argument_type=ArgumentType.VALUE, data=100)
        step_5.add_output('produce')
        pipeline.add_step(step_5)

        # Cluster -> Dataframe
        step_6 = meta_pipeline.PrimitiveStep(primitive_description=NDArrayToDataFramePrimitive.metadata.query())
        step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
        step_6.add_output('produce')
        pipeline.add_step(step_6)

        # Dataframe -> combine with original
        step_7 = meta_pipeline.PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
        step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
        step_7.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
        step_7.add_output('produce')
        pipeline.add_step(step_7)

        # Final Output
        pipeline.add_output(name='output', data_reference='steps.7.produce')

        return pipeline
