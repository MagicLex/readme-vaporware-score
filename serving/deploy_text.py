"""Deploy the vaporware_text model as a KServe online inference endpoint.

The heavy word+char TF-IDF model serves on its own predictor pod; clients (the
Streamlit app) send raw README text and get a 0-100 score back.
"""
import hopsworks
from hsml.resources import PredictorResources, Resources
from hsml.scaling_config import PredictorScalingConfig, ScaleMetric

DEPLOYMENT = "vaporwaretext"
ENV = "vaporware-serve-env"
MODEL = "vaporware_text"
VERSION = 1


def main():
    project = hopsworks.login()
    mr = project.get_model_registry()
    model = mr.get_model(MODEL, version=VERSION)
    script = f"/Projects/{project.name}/Models/{MODEL}/{VERSION}/Files/predict.py"

    ms = project.get_model_serving()
    existing = ms.get_deployment(DEPLOYMENT)
    if existing is not None:
        print("deleting existing deployment", flush=True)
        existing.delete()

    dep = model.deploy(
        name=DEPLOYMENT,
        script_file=script,
        environment=ENV,
        resources=PredictorResources(
            requests=Resources(cores=1, memory=2048, gpus=0),
            limits=Resources(cores=2, memory=4096, gpus=0),
        ),
        scaling_configuration=PredictorScalingConfig(
            min_instances=1, max_instances=1,
            scale_metric=ScaleMetric.CONCURRENCY, target=16,
        ),
    )
    dep.start(await_running=600)
    print("running:", dep.is_running(), flush=True)
    print("inference_url:", dep.get_inference_url(), flush=True)


if __name__ == "__main__":
    main()
