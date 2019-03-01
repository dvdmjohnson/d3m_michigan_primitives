# Information on all pipelines and which primitives are used.

import json

from spider.pipelines.base import BasePipeline
from spider.pipelines.featurization_vgg16 import FeaturizationVGG16Pipeline
from spider.pipelines.featurization_audiofeaturization import FeaturizationAudioFeaturizationPipeline
from spider.pipelines.featurization_i3d import FeaturizationI3DPipeline
from spider.pipelines.supervised_learning_owl import OWLRegressionPipeline

from spider.pipelines.cluster_ssc_cvxcluster import SSCCVXPipeline
from spider.pipelines.cluster_ssc_ompcluster import SSCOMPPipeline
from spider.pipelines.cluster_ssc_admmcluster import SSCADMMPipeline
from spider.pipelines.cluster_ksscluster import KSSPipeline
from spider.pipelines.cluster_eksscluster import EKSSPipeline
from spider.pipelines.unsupervised_learning_grasta import GRASTAPipeline

from spider.featurization.vgg16 import VGG16

PIPELINES_BY_PRIMITIVE = {
    'd3m.primitives.spider.featurization.VGG16': [
        FeaturizationVGG16Pipeline,
    ],
    'd3m.primitives.feature_extraction.audio_featurization.umich': [
        FeaturizationAudioFeaturizationPipeline,
    ],
    'd3m.primitives.feature_extraction.i3d.umich': [
        FeaturizationI3DPipeline,
    ],
    'd3m.primitives.regression.owl_regression.umich': [
        OWLRegressionPipeline,
    ],
    'd3m.primitives.clustering.ssc_cvx.umich': [
        SSCCVXPipeline,
    ],
    'd3m.primitives.clustering.ssc_omp.umich': [
        SSCOMPPipeline,
    ],
    'd3m.primitives.clustering.ssc_admm.umich': [
        SSCADMMPipeline,
    ],
    'd3m.primitives.clustering.kss.umich': [
        KSSPipeline,
    ],
    'd3m.primitives.clustering.ekss.umich': [
        EKSSPipeline,
    ],
    'd3m.primitives.data_compression.grasta.umich': [
        GRASTAPipeline,
    ],
    'd3m.primitives.data_compression.grasta.umich': [
        GROUSEPipeline,
    ],
}

def get_primitives():
    return PIPELINES_BY_PRIMITIVE.keys()

def get_pipelines(primitive = None):
    if (primitive is not None):
        if (primitive not in PIPELINES_BY_PRIMITIVE):
            return []
        return PIPELINES_BY_PRIMITIVE[primitive]

    pipelines = set()
    for primitive_pipelines in PIPELINES_BY_PRIMITIVE.values():
        pipelines = pipelines | set(primitive_pipelines)
    return pipelines

if __name__ == '__main__':
    print(json.dumps(PIPELINES_BY_PRIMITIVE))

