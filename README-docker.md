# CS6240-Demo
Example code for CS6240, updated 2023


Docker commands:
- Build the image with `docker build . -t cs6240` (from the root directory of the project).
- Start the container with `docker run -itd --privileged cs6240`. This command will also return the id of the container.
- To start a bash terminal on the container: `docker exec -ti <container_id> bash` where you have to replace `<container_id>` with the id you get from `docker ps`).
- `docker cp <container_id>:/app/ ./docker_results` will copy the results back to the host in a `docker_results` folder.