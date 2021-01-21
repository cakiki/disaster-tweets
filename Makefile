.PHONY: clean data lint requirements

docker-build-image:
	docker build -t disaster-tweets:0.0.4 ./dockerfiles/vanilla/

docker-build-image-gpu:
	docker build -t disaster-tweets-gpu:0.0.1 ./dockerfiles/gpu/
docker-run-jupyter:
	docker run --rm -it -p 8889:8888 -p 9999:9999 -p 6006:6006 \
	--env PYTHONPATH=/home/jovyan/work/src \
	--mount type=bind,source=${PWD},target=/home/jovyan/work --name disaster-tweets \
	--workdir=/home/jovyan/work \
	 disaster-tweets:0.0.4
docker-run-jupyter-gpu:
	docker run --rm -it -p 8888:8888 --gpus all --env PYTHONPATH=/tf/src \
			--mount type=bind,source=${PWD},target=/tf \
			disaster-tweets-gpu:0.0.1
fix-permissions:
	sudo chown -R 1000:1000 .

docker-run-rapids:
	docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
		--mount type=bind,source=${PWD},target=/rapids/project \
		--workdir=/rapids --env PYTHONPATH=/rapids/project/src \
    	rapidsai/rapidsai-core:cuda10.2-runtime-ubuntu18.04