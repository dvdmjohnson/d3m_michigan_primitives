# Output pipelines in JSON.

import argparse
import torch
import os

import spider.pipelines.all

def load_args():
    parser = argparse.ArgumentParser(description = "Output a pipeline's JSON")

    parser.add_argument(
        'pipeline', action = 'store', metavar = 'PIPELINE',
        help = "the name of the pipeline to generate",
    )

    arguments = parser.parse_args()

    return arguments.pipeline

def main():
    pipeline_name = load_args()

    pipeline = None
    for pipeline_class in spider.pipelines.all.get_pipelines():
        if (pipeline_class.__name__ == pipeline_name):
            pipeline = pipeline_class()
            break

    if (pipeline is None):
        raise ValueError("Could not find pipeline with name: %s." % (pipeline_name))

    print(pipeline.get_json())

    # Output pipeline json
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output', 'pipeline.json'))
    with open(output_path, 'w', encoding='utf8') as fd:
        str_ = pipeline.get_json()
        fd.write("{}".format(str_))

if __name__ == '__main__':
    main()
