PY ?= .venv/bin/python
STACK ?= gaugeless
REGION ?= us-east-1
BUCKET = gaugeless-$(shell aws sts get-caller-identity --query Account --output text)-$(REGION)
DEMO_SITES = LaMorgeRiver RiverArrow/Stabilised/UAS_stable_trans_seeded RiverArrow/Stabilised/UAS_stable_trans_unseeded RiverArrow/Reference RiverLaVence RiverNoce

.PHONY: test baseline agent degrade eval package deploy upload-code upload-demo-data submit-demo web-up

test:            ## unit tests (synthetic, no dataset needed)
	$(PY) -m pytest -q tests

baseline:        ## every method x every case, fixed parameters
	$(PY) -m gaugeless.evaluate --max-frames 150 --out results/baseline.jsonl

agent:           ## rules planner on every case
	$(PY) -m gaugeless.agent.run --all --quiet --out results/agent_rules.jsonl

degrade:         ## agent vs fixed pipeline on controlled degradations
	$(PY) -m gaugeless.agent.degrade_eval --out results/degrade.jsonl

eval: test baseline agent degrade
	$(PY) -m gaugeless.discharge

package:
	mkdir -p dist && COPYFILE_DISABLE=1 tar --no-xattrs --exclude __pycache__ -czf dist/gaugeless.tar.gz gaugeless requirements.txt

upload-code: package
	aws s3 cp dist/gaugeless.tar.gz s3://$(BUCKET)/code/gaugeless.tar.gz

upload-demo-data:
	for s in $(DEMO_SITES); do aws s3 sync data/perks/$$s s3://$(BUCKET)/data/$$s --only-show-errors; done

deploy:          ## make deploy EMAIL=you@example.com
	aws cloudformation deploy --region $(REGION) --stack-name $(STACK) --template-file infra/gaugeless.yaml \
	  --capabilities CAPABILITY_IAM --parameter-overrides AlertEmail=$(EMAIL)

submit-demo:     ## queue a dataset site as a job
	echo '{"job_id":"demo-$(shell date +%s)","site":"$(SITE)","flood_q":$(or $(FLOOD),null)}' > /tmp/job.json && \
	aws s3 cp /tmp/job.json s3://$(BUCKET)/uploads/demo-$(shell date +%s).json

web-up: upload-code   ## public demo web app on the Graviton worker, port 80 (URL: stack output WebUrl)
	STACK=$(STACK) REGION=$(REGION) BUCKET=$(BUCKET) sh infra/web-up.sh
