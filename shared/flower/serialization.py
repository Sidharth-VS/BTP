import io
import json
import numpy as np
from flwr.common import Array, ConfigRecord, ArrayRecord, RecordDict
from shared.flower.messages import QueryIns, QueryRes, RegisterIns, RegisterRes
from shared.schemas.document import SearchResult


class FlowerMessageSerializer:
    @staticmethod
    def query_ins_to_recordset(ins: QueryIns) -> RecordDict:
        record_dict = RecordDict()
        config = ConfigRecord({
            "action": "query",
            "query": ins.query,
            "top_k": ins.top_k,
            "filters": json.dumps(ins.filters),
        })
        record_dict.configs_records["query_ins"] = config
        return record_dict

    @staticmethod
    def recordset_to_query_ins(record_dict: RecordDict) -> QueryIns:
        config = record_dict.configs_records["query_ins"]
        return QueryIns(
            query=str(config["query"]),
            top_k=int(config["top_k"]),
            filters=json.loads(str(config["filters"])),
        )

    @staticmethod
    def query_res_to_recordset(res: QueryRes) -> RecordDict:
        record_dict = RecordDict()
        config = ConfigRecord({
            "node_id": res.node_id,
            "processing_time": float(res.processing_time),
            "trust_score": float(res.trust_score),
            "results_json": json.dumps([r.model_dump() for r in res.results]),
        })
        record_dict.configs_records["query_meta"] = config

        if res.doc_embeddings:
            arr = np.array(res.doc_embeddings, dtype=np.float32)
            # Flower 1.30+ natively serializes ndarrays into .npy format
            record_dict.array_records["doc_embeddings"] = ArrayRecord({
                "embeddings": Array(arr)
            })
        return record_dict

    @staticmethod
    def recordset_to_query_res(record_dict: RecordDict) -> QueryRes:
        config = record_dict.configs_records["query_meta"]
        results_data = json.loads(str(config["results_json"]))
        results = [SearchResult(**item) for item in results_data]

        doc_embeddings = []
        if "doc_embeddings" in record_dict.array_records:
            flwr_arr = record_dict.array_records["doc_embeddings"]["embeddings"]
            # Use np.load with BytesIO to parse the .npy header correctly
            np_arr = np.load(io.BytesIO(flwr_arr.data))
            doc_embeddings = np_arr.tolist()

        return QueryRes(
            results=results,
            node_id=str(config["node_id"]),
            processing_time=float(config["processing_time"]),
            trust_score=float(config["trust_score"]),
            doc_embeddings=doc_embeddings,
        )

    @staticmethod
    def register_ins_to_recordset(ins: RegisterIns) -> RecordDict:
        record_dict = RecordDict()
        config = ConfigRecord({
            "action": "register",
            "node_id": ins.node_id,
            "address": ins.address,
            "domain": ins.domain,
            "capabilities": json.dumps(ins.capabilities),
        })
        record_dict.configs_records["register_info"] = config

        centroid_arr = np.array(ins.centroid, dtype=np.float32)
        arrays_dict = {
            "centroid": Array(centroid_arr)
        }
        
        if ins.profile_centroids:
            p_arr = np.array(ins.profile_centroids, dtype=np.float32)
            arrays_dict["profile_centroids"] = Array(p_arr)

        record_dict.array_records["profile_params"] = ArrayRecord(arrays_dict)
        return record_dict

    @staticmethod
    def recordset_to_register_ins(record_dict: RecordDict) -> RegisterIns:
        config = record_dict.configs_records["register_info"]
        arrays = record_dict.array_records["profile_params"]

        centroid_raw = arrays["centroid"]
        centroid = np.load(io.BytesIO(centroid_raw.data)).tolist()

        profile_centroids = []
        if "profile_centroids" in arrays:
            p_raw = arrays["profile_centroids"]
            profile_centroids = np.load(io.BytesIO(p_raw.data)).tolist()

        return RegisterIns(
            node_id=str(config["node_id"]),
            address=str(config["address"]),
            domain=str(config["domain"]),
            capabilities=json.loads(str(config["capabilities"])),
            centroid=centroid,
            profile_centroids=profile_centroids,
        )