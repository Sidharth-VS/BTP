import os
import sys
import math
import yaml

DOCKER_COMPOSE_PATH = "docker-compose.yml"
NODES_YAML_PATH = "nodes/nodes.yaml"

def get_user_inputs():
    print("=" * 65)
    print(" FedRAG Dynamic Cluster Scaler & Resource Allocator")
    print("=" * 65)
    try:
        total_containers = int(input("Enter TOTAL containers (Coordinator + Frontend + Clients + Malicious): ").strip())
        if total_containers < 3:
            print("❌ Error: Total containers must be at least 3 (1 Coordinator, 1 Frontend, and at least 1 Node).")
            sys.exit(1)

        max_workers = total_containers - 2
        num_malicious = int(input(f"Enter number of MALICIOUS nodes (0 to {max_workers}): ").strip())
        if num_malicious < 0 or num_malicious > max_workers:
            print(f"❌ Error: Malicious nodes must be between 0 and {max_workers}.")
            sys.exit(1)

    except ValueError:
        print("❌ Error: Please enter valid integer numbers.")
        sys.exit(1)

    num_honest = total_containers - 2 - num_malicious
    return total_containers, num_honest, num_malicious

def calculate_resources(num_workers):
    total_cpus = os.cpu_count() or 8
    
    # Reserve 10 cores for host OS / WSL overhead, 2.0 for Coordinator, 1.0 for Frontend
    os_overhead = 10.0
    coord_cpus = 2.0
    front_cpus = 1.0
    
    available_for_workers = max(1.0, total_cpus - os_overhead - coord_cpus - front_cpus)
    
    if num_workers > 0:
        # Calculate fair share per worker node, capped at 4.0 CPUs max and 0.5 CPUs min
        raw_cpu_per_node = available_for_workers / num_workers
        cpu_per_node = round(max(0.5, min(4.0, raw_cpu_per_node)), 2)
    else:
        cpu_per_node = 1.0

    # Set thread limit to integer ceiling of allocated CPUs (capped between 1 and 4)
    threads_per_node = max(1, min(4, int(math.ceil(cpu_per_node))))
    
    return total_cpus, coord_cpus, front_cpus, cpu_per_node, threads_per_node

def main():
    total_containers, num_honest, num_malicious = get_user_inputs()
    num_workers = num_honest + num_malicious

    total_cpus, coord_cpus, front_cpus, cpu_per_node, threads_per_node = calculate_resources(num_workers)

    print("\n" + "-" * 65)
    print(" 📊 Cluster Topology & Resource Allocation Summary")
    print("-" * 65)
    print(f"  • Host CPU Cores Detected : {total_cpus} cores")
    print(f"  • Total Containers        : {total_containers}")
    print(f"    ├── Coordinator         : 1 (Allocated: {coord_cpus} CPUs)")
    print(f"    ├── Frontend            : 1 (Allocated: {front_cpus} CPUs)")
    print(f"    ├── Honest Client Nodes : {num_honest} (node-1 to node-{num_honest})" if num_honest > 0 else "    ├── Honest Client Nodes : 0")
    print(f"    └── Malicious Nodes     : {num_malicious} (node-{num_honest + 1} to node-{num_workers})" if num_malicious > 0 else "    └── Malicious Nodes     : 0")
    print(f"  • Per-Worker Allocation   : {cpu_per_node} CPUs | OMP_NUM_THREADS={threads_per_node}")
    print("-" * 65)

    # 1. Generate nodes/nodes.yaml
    honest_nodes = [{"node_id": f"node-{i}", "domain": "general"} for i in range(1, num_honest + 1)]
    malicious_nodes = [{"node_id": f"node-{i}", "domain": "medical"} for i in range(num_honest + 1, num_workers + 1)]

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
        "nodes": honest_nodes + malicious_nodes
    }

    os.makedirs(os.path.dirname(NODES_YAML_PATH), exist_ok=True)
    with open(NODES_YAML_PATH, "w") as f:
        yaml.dump(nodes_yaml_data, f, sort_keys=False)

    # Dynamic thread limits based on CPU allocation
    THREAD_LIMITS = [
        f"OMP_NUM_THREADS={threads_per_node}",
        f"OPENBLAS_NUM_THREADS={threads_per_node}",
        f"MKL_NUM_THREADS={threads_per_node}",
        f"VECLIB_MAXIMUM_THREADS={threads_per_node}",
        f"NUMEXPR_NUM_THREADS={threads_per_node}"
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
            "deploy": {"resources": {"limits": {"cpus": str(coord_cpus)}}},
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
            "deploy": {"resources": {"limits": {"cpus": str(front_cpus)}}},
            "depends_on": {"coordinator": {"condition": "service_healthy"}},
            "networks": ["fedrag-net"]
        }
    }

    # Generate Honest Client Nodes (node-1 to node-{num_honest})
    for i in range(1, num_honest + 1):
        # Ensure host directories exist for volumes
        os.makedirs(f"./workspace/chroma/node-{i}", exist_ok=True)
        os.makedirs(f"./workspace/data/node-{i}", exist_ok=True)
        
        services[f"node-{i}"] = {
            "build": {"context": ".", "dockerfile": "nodes/Dockerfile"},
            "image": "fedrag-node:latest",
            "container_name": f"fedrag-node-{i}",
            "environment": [
                f"NODE_ID=node-{i}",
                "NODES_CONFIG=/app/nodes/nodes.yaml",
                "API_KEY=${API_KEY:-fedrag-dev-key}"
            ] + THREAD_LIMITS,
            "deploy": {"resources": {"limits": {"cpus": str(cpu_per_node)}}},
            "volumes": [
                f"./workspace/chroma/node-{i}:/workspace/chroma/node-{i}",
                f"./workspace/data/node-{i}:/workspace/data/node-{i}:ro",
                "./nodes:/app/nodes",
                "./shared:/app/shared",
                "./nodes/nodes.yaml:/app/nodes/nodes.yaml:ro",
                "/etc/machine-id:/host/etc/machine-id:ro"
            ],
            "depends_on": {"coordinator": {"condition": "service_healthy"}},
            "networks": ["fedrag-net"]
        }

    # Generate Malicious Nodes (node-{num_honest + 1} to node-{num_workers})
    for i in range(num_honest + 1, num_workers + 1):
        services[f"node-{i}"] = {
            "build": {"context": ".", "dockerfile": "nodes/Dockerfile"},
            "image": "fedrag-node:latest",
            "container_name": f"fedrag-node-{i}",
            "command": ["python", "-m", "tests.mocks.malicious_node"],
            "environment": [
                f"NODE_ID=node-{i}",
                "COORDINATOR_ADDRESS=fedrag-coordinator:9091",
                "API_KEY=${API_KEY:-fedrag-dev-key}"
            ] + THREAD_LIMITS,
            "deploy": {"resources": {"limits": {"cpus": str(cpu_per_node)}}},
            "volumes": [
                "./nodes:/app/nodes",
                "./shared:/app/shared",
                "./tests:/app/tests",
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

    print(f"✅ Successfully generated '{NODES_YAML_PATH}' and '{DOCKER_COMPOSE_PATH}'.")
    print("🚀 Run 'docker compose up -d --remove-orphans' to launch the scaled cluster.")

if __name__ == "__main__":
    main()
