import yaml

DOCKER_COMPOSE_PATH = "docker-compose.yml"
NODES_YAML_PATH = "nodes/nodes.yaml"

# 1. Generate nodes.yaml
honest_nodes = [{"node_id": f"node-{i}", "domain": "general"} for i in range(1, 10)]
malicious_node = [{"node_id": "node-10", "domain": "medical"}]

nodes_yaml_data = {
    "defaults": {
        "server_address": "fedrag-coordinator:9091",
        "embedding_model_name": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "top_k_default": 5,
        "persist_directory_prefix": "/workspace/chroma",
        "data_directory_prefix": "/workspace/data",
        "collection_name_prefix": "fedrag"
    },
    "nodes": honest_nodes + malicious_node
}

with open(NODES_YAML_PATH, "w") as f:
    yaml.dump(nodes_yaml_data, f, sort_keys=False)

# Thread limits increased to 3
THREAD_LIMITS = [
    "OMP_NUM_THREADS=3",
    "OPENBLAS_NUM_THREADS=3",
    "MKL_NUM_THREADS=3",
    "VECLIB_MAXIMUM_THREADS=3",
    "NUMEXPR_NUM_THREADS=3"
]

# 2. Generate docker-compose.yml
services = {
    "coordinator": {
        "build": {"context": ".", "dockerfile": "coordinator/Dockerfile"},
        "container_name": "fedrag-coordinator",
        "ports": ["${COORDINATOR_PORT:-8000}:8000", "${FLOWER_PORT:-9091}:9091"],
        "extra_hosts": ["host.docker.internal:host-gateway"],
        "volumes": [
            "./workspace/coordinator:/workspace/coordinator",
            "./coordinator:/app/coordinator",
            "./coordinator/config.yaml:/app/coordinator/config.yaml:ro",
            "./shared:/app/shared"
        ],
        "environment": [
            "COORDINATOR_CONFIG=/app/coordinator/config.yaml",
            "API_KEY=${API_KEY:-fedrag-dev-key}",
            "OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://host.docker.internal:11434}",
            "OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2:3b}"
        ] + THREAD_LIMITS,
        "deploy": {"resources": {"limits": {"cpus": "2.0"}}},
        "healthcheck": {
            "test": ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"],
            "interval": "10s", "timeout": "5s", "retries": 5, "start_period": "120s"
        },
        "networks": {"fedrag-net": {"aliases": ["coordinator", "fedrag-coordinator"]}}
    },
    "frontend": {
        "build": {"context": "./frontend", "dockerfile": "Dockerfile"},
        "container_name": "fedrag-frontend",
        "ports": ["8501:8501"],
        "volumes": ["./frontend:/app"],
        "environment": [
            "COORDINATOR_API_URL=${COORDINATOR_API_URL:-http://fedrag-coordinator:8000/api/v1/query}",
            "API_KEY=${API_KEY:-fedrag-dev-key}"
        ],
        "deploy": {"resources": {"limits": {"cpus": "1.0"}}},
        "depends_on": {"coordinator": {"condition": "service_healthy"}},
        "networks": ["fedrag-net"]
    }
}

# Generate Honest Nodes (1 to 9)
for i in range(1, 10):
    services[f"node-{i}"] = {
        "build": {"context": ".", "dockerfile": "nodes/Dockerfile"},
        "image": "fedrag-node:latest",
        "container_name": f"fedrag-node-{i}",
        "environment": [
            f"NODE_ID=node-{i}", 
            "NODES_CONFIG=/app/nodes/nodes.yaml", 
            "API_KEY=${API_KEY:-fedrag-dev-key}"
        ] + THREAD_LIMITS,
        "deploy": {"resources": {"limits": {"cpus": "3.0"}}},
        "volumes": [
            f"./workspace/chroma/node-{i}:/workspace/chroma/node-{i}",
            f"./workspace/data/node-{i}:/workspace/data/node-{i}:ro",
            "./nodes:/app/nodes", "./shared:/app/shared", "./nodes/nodes.yaml:/app/nodes/nodes.yaml:ro",
            "/etc/machine-id:/host/etc/machine-id:ro"
        ],
        "depends_on": {"coordinator": {"condition": "service_healthy"}},
        "networks": ["fedrag-net"]
    }

# Generate Malicious Node (10)
services["node-10"] = {
    "build": {"context": ".", "dockerfile": "nodes/Dockerfile"},
    "image": "fedrag-node:latest",
    "container_name": "fedrag-node-10",
    "command": ["python", "-m", "tests.mocks.malicious_node"],
    "environment": [
        "NODE_ID=node-10", 
        "COORDINATOR_ADDRESS=fedrag-coordinator:9091", 
        "API_KEY=${API_KEY:-fedrag-dev-key}"
    ] + THREAD_LIMITS,
    "deploy": {"resources": {"limits": {"cpus": "3.0"}}},
    "volumes": [
        "./nodes:/app/nodes", "./shared:/app/shared", "./tests:/app/tests",
        "/etc/machine-id:/host/etc/machine-id:ro"
    ],
    "depends_on": {"coordinator": {"condition": "service_healthy"}},
    "networks": ["fedrag-net"]
}

compose_data = {
    "services": services,
    "networks": {"fedrag-net": {"driver": "bridge"}}
}

with open(DOCKER_COMPOSE_PATH, "w") as f:
    yaml.dump(compose_data, f, sort_keys=False)
