Video-Llava-7B-hf-Runpod-local_weights_access

These files are to be used for deploying the Video-Llava-7B-hf model onto runpod.
It's specifically is ensuring that the weights are downloaded within the docker container when built.
This should help improve cold boot time significantly. 
Repo also used for testing workflows with docker and github, specifically the creating a deployment from github repo feature within runpod.
