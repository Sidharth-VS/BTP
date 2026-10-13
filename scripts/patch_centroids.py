import re

# 1. Patch endpoints.py
ep_path = "coordinator/app/api/v1/endpoints.py"
with open(ep_path, "r") as f:
    ep_data = f.read()

ep_data = re.sub(
    r'tasr_router\.register_node\(nid,\s*\[\[0\.0\].*?\]\)',
    'centroids = info.get("profile_centroids")\n            if not centroids or not isinstance(centroids, list) or len(centroids) == 0:\n                centroids = [[0.0] * 384] * 5\n            tasr_router.register_node(nid, centroids)',
    ep_data
)
with open(ep_path, "w") as f:
    f.write(ep_data)

# 2. Patch strategy.py
st_path = "coordinator/app/flwr_server/strategy.py"
with open(st_path, "r") as f:
    st_data = f.read()

if '"profile_centroids"' not in st_data:
    st_data = st_data.replace(
        'self._node_registry[client_proxy.cid] = {',
        'import json\n            try:\n                p_cents = json.loads(str(metrics.get("profile_centroids_json", "[]")))\n            except:\n                p_cents = []\n\n            self._node_registry[client_proxy.cid] = {\n                "profile_centroids": p_cents,'
    )
    with open(st_path, "w") as f:
        f.write(st_data)

print("TASR Centroid logic successfully injected!")
