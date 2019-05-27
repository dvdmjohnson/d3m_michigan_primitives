from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import ArgumentType, Context

from spider.pipelines.base import BasePipeline
from spider.featurization.audio_featurization import AudioFeaturization
from bbn_primitives.time_series import AudioReader
from bbn_primitives.time_series import TargetsReader
from d3m.primitives.data_transformation.denormalize import Common as DenormalizePrimitive
from d3m.primitives.data_transformation.dataframe_to_ndarray import Common as DataFrameToNDArrayPrimitive
from d3m.primitives.data_transformation.ndarray_to_dataframe import Common as NDArrayToDataFramePrimitive
from d3m.primitives.classification.random_forest import DataFrameCommon as RandomForestClassifierPrimitive
from d3m.primitives.data_transformation.construct_predictions import DataFrameCommon as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive

class FeaturizationAudioFeaturizationPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        
        #specify one seed dataset on which this pipeline can operate
        self.dataset = '31_urbansound'
        self.meta_info = self.genmeta(self.dataset)

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

        #extract targets (i.e. classification labels) from dataset into a dedicated dataframe
        step_1 = meta_pipeline.PrimitiveStep(primitive_description = TargetsReader.metadata.query())
        step_1.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.0.produce'' 
        )
        step_1.add_output('produce')
        pipeline.add_step(step_1)

        #extract raw audio sequence for each training element from the Dataset into a list of ndarrays
        step_2 = meta_pipeline.PrimitiveStep(primitive_description = AudioReader.metadata.query())
        step_2.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'inputs.1' #same as step 1
        )
        step_2.add_output('produce')
        pipeline.add_step(step_2)

        #featurize each audio sequence (list of raw ndarrays to single feature matrix ndarray)
        step_3 = meta_pipeline.PrimitiveStep(primitive_description = AudioFeaturization.metadata.query())
        step_3.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.2.produce' #inputs here are the outputs from step 2
        )

        step_3.add_output('produce')
        pipeline.add_step(step_3)

        #classify the featurized outputs (feature matrix dataframe to class prediction vector dataframe)
        step_4 = meta_pipeline.PrimitiveStep(primitive_description = RandomForestClassifierPrimitive.metadata.query())
        step_4.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.3.produce' #inputs here are the outputs from step 4
        )
        step_4.add_argument(
                name = 'outputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.1.produce' #inputs here are the outputs from step 1
        )
        step_4.add_output('produce')
        pipeline.add_step(step_4)

        #transform original dataset into a dataframe
        step_5 = meta_pipeline.PrimitiveStep(primitive_description = DatasetToDataFramePrimitive.metadata.query())
        step_5.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'inputs.1' #inputs here are the original dataset
        )
        step_5.add_hyperparameter(
                name='dataframe_resource',
                argument_type=ArgumentType.VALUE,
                data='2' #select the resource within urbansound that contains the d4mIndex values
        )
        step_5.add_output('produce')
        pipeline.add_step(step_5)
        
        #finally generate a properly-formatted output dataframe from the prediction outputs using the input dataframe as a reference
        step_6 = meta_pipeline.PrimitiveStep(primitive_description = ConstructPredictionsPrimitive.metadata.query())
        step_6.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.4.produce' #inputs here are the prediction column
        )
        step_6.add_argument(
                name = 'reference',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.5.produce' #inputs here are the dataframed input dataset
        )
        step_6.add_output('produce')
        pipeline.add_step(step_6)

        # Adding output step to the pipeline
        pipeline.add_output(name = 'Links', data_reference = 'steps.6.produce')

        return pipeline
