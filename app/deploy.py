"""Deploy the scorer as a Hopsworks Streamlit app.

Run from a Hopsworks terminal where this repo lives under the project FUSE mount
(/hopsfs/...). The app's project-relative path is derived from this file's own
location, so it is not hardcoded to one user.
"""
from pathlib import Path

import hopsworks

APP_NAME = "vaporwarescore"
ENV_NAME = "vaporware-app-env"

# /hopsfs/<...project-relative...>/app/deploy.py -> <project-relative>/app/app.py
rel = str(Path(__file__).resolve()).split("/hopsfs/", 1)[1]
APP_PATH = str(Path(rel).parent / "app.py")


def main():
    project = hopsworks.login()
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
