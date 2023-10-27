PROJECT_NAME := meta-timeseries
IMAGE_NAME := $(PROJECT_NAME):$(USER)
PROJECT_PATH := $(shell dirname $(dir $(realpath $(firstword $(MAKEFILE_LIST)))))
DOCKER_PARAMETERS := --user $(shell id -u) -v $(PROJECT_PATH):/project -w /project/source -e PYTHONPATH=/project/source

ifdef gpu
	DOCKER_PARAMETERS += --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$(gpu)
endif

ifdef filter
	filter_param = --summary_filter=$(filter)
endif

ifdef validation
	validation_param = --validation_mode=$(validation)
endif

.PHONY: test

build:
	@docker build . -t $(IMAGE_NAME)

push:
	@docker push $(IMAGE_NAME)

init-resources:
	@docker run -it --rm $(DOCKER_PARAMETERS) $(IMAGE_NAME) python resources/m3/dataset.py
	@docker run -it --rm $(DOCKER_PARAMETERS) $(IMAGE_NAME) python resources/m4/dataset.py
	@docker run -it --rm $(DOCKER_PARAMETERS) $(IMAGE_NAME) python resources/tourism/dataset.py
	@docker run -it --rm $(DOCKER_PARAMETERS) $(IMAGE_NAME) python resources/electricity/dataset.py
	@docker run -it --rm $(DOCKER_PARAMETERS) $(IMAGE_NAME) python resources/traffic/dataset.py
	@docker run -it --rm $(DOCKER_PARAMETERS) $(IMAGE_NAME) python resources/fred/dataset.py

init-stats:
	@docker run -it --rm $(DOCKER_PARAMETERS) $(IMAGE_NAME) python experiments/stat_models/datasets.py

create-experiments: .require_experiment .require_name
	@docker run --rm $(DOCKER_PARAMETERS) $(IMAGE_NAME) python $(experiment)/main.py init --name=$(name)

run: .require_experiment .require_name
	@./cluster_submit.sh $(experiment)/$(name) $(IMAGE_NAME)

summary: .require_experiment .require_name
	@docker run --rm $(DOCKER_PARAMETERS) $(IMAGE_NAME) python $(experiment)/main.py summary --name=$(name) $(filter_param) $(validation_param)

exec:
	@docker run -it --rm $(DOCKER_PARAMETERS) $(IMAGE_NAME) $(cmd)

jupyterlab: .require_port
	docker run -d --rm --name=$(PROJECT_NAME)-jupyterlab -p $(port):8888 -e HOME=/home/jupyter $(DOCKER_PARAMETERS) $(IMAGE_NAME) \
		bash -c "jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token='$(PROJECT_NAME)'"
	@echo "Jupyterlab token: $(PROJECT_NAME)"

tensorboard: .require_port .require_experiment
	docker run -d --rm --name=$(PROJECT_NAME)-tensorboard -p $(port):2121 $(DOCKER_PARAMETERS) $(IMAGE_NAME) \
		bash -c "/start_tensorboard.sh 2121 /project/$(experiment)"
test:
	@docker run -it --rm $(DOCKER_PARAMETERS) $(IMAGE_NAME) python -m unittest

.require_port:
ifndef port
	$(error port parameter is required)
endif

.require_experiment:
ifndef experiment
	$(error experiment parameter is required)
endif

.require_name:
ifndef name
	$(error name parameter is required)
endif

