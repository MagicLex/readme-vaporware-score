# Hopsworks terminals have python on PATH. Override with `make PY=/path/to/python <target>`.
PY := python

.PHONY: collect text features eda train serve-model sync-app app deploy-app clean

collect:          ## Pull labelled README dataset from GitHub (needs gh auth)
	$(PY) collect/collect.py

text:             ## Re-fetch raw README text for each repo (needs gh auth)
	$(PY) collect/add_text.py

serve-model:      ## Build + register the text model, deploy to a KServe endpoint
	$(PY) serving/register_text.py
	$(PY) serving/deploy_text.py

features:         ## Load dataset into the feature group (Hopsworks job)
	-hops files mkdir Resources/vaporware
	hops files upload data/repos.jsonl Resources/vaporware/ --overwrite
	hops job deploy vaporware-features pipelines/feature_pipeline.py \
		--env python-feature-pipeline --run --wait --overwrite

eda:              ## Profile features vs label, write report + plots (local)
	$(PY) pipelines/eda.py

train:            ## Feature view -> train -> register model (Hopsworks job)
	hops job deploy vaporware-train pipelines/train.py \
		--env pandas-training-pipeline --run --wait --overwrite

sync-app:         ## Copy the canonical feature extractor next to the app
	cp readme_features.py app/readme_features.py

app: sync-app     ## Run the scorer app locally
	$(PY) -m streamlit run app/app.py

deploy-app: sync-app  ## Deploy the scorer as a Hopsworks app
	$(PY) app/deploy.py

clean:
	rm -rf models/artifact models/eda models/registry_dl __pycache__ */__pycache__
