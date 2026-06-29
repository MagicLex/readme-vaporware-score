"""Deploy the scorer as a Hopsworks Streamlit app.

Run from a Hopsworks terminal where this repo lives under the project FUSE mount
(/hopsfs/...). The app's project-relative path is derived from this file's own
location, so it is not hardcoded to one user.
"""
from pathlib import Path

import hopsworks

APP_NAME = "vaporwarescore"
ENV_NAME = "vaporware-app-env"

# /hopsfs/<...project-relative...>/app/deploy.py -> <project-relative> paths
rel = str(Path(__file__).resolve()).split("/hopsfs/", 1)[1]
APP_PATH = str(Path(rel).parent / "app.py")
REQS_PATH = str(Path(rel).parent.parent / "app-requirements.txt")


def ensure_env(project):
    """Create the app env (cloned from python-app-pipeline, with the model's
    pinned libs) if it does not exist yet. Build takes a few minutes."""
    env_api = project.get_environment_api()
    env = env_api.get_environment(ENV_NAME)
    if env is None:
        print(f"creating {ENV_NAME} from python-app-pipeline (few min)...", flush=True)
        env = env_api.create_environment(ENV_NAME, base_environment_name="python-app-pipeline")
        env.install_requirements(REQS_PATH, await_installation=True)
        print("env ready", flush=True)
    return env


def main():
    project = hopsworks.login()
    ensure_env(project)
    apps = project.get_app_api()
    print(f"app_path={APP_PATH} env={ENV_NAME}", flush=True)

    app = apps.get_app(APP_NAME)
    if app is None:
        app = apps.create_app(
            name=APP_NAME,
            app_path=APP_PATH,
            environment=ENV_NAME,
            memory=2048,
            cores=1.0,
        )
    app.run(await_serving=True)
    print("serving:", app.serving)
    print("URL:", app.get_url())


if __name__ == "__main__":
    main()
