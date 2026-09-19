import time
from flwr.client import ClientApp
from flwr.common import Context, Message
from nodes.app.rag.chroma_db import ChromaStore
from nodes.app.rag.profiler import NodeProfiler
from shared.flower.messages import QueryRes, RegisterIns
from shared.flower.serialization import FlowerMessageSerializer


def create_node_client_app(
    node_id: str,
    domain: str,
    chroma_store: ChromaStore,
    profiler: NodeProfiler,
) -> ClientApp:
    app = ClientApp()

    @app.query()
    def handle_message(msg: Message, ctx: Context) -> Message:
        record_dict = msg.content
        action = record_dict.configs_records.get("query_ins", {}).get("action")

        # 1. Registration Handler: Sends C_i and P_i vectors to Coordinator
        if action == "register":
            c_i, p_i = profiler.compute_profile()
            reg_ins = RegisterIns(
                node_id=node_id,
                address=f"{node_id}:9091",
                domain=domain,
                capabilities=["chromadb", "retrieval"],
                centroid=c_i,
                profile_centroids=p_i,
            )
            out_record = FlowerMessageSerializer.register_ins_to_recordset(reg_ins)
            return msg.create_reply(content=out_record)

        # 2. Query Handler: Runs similarity search & returns embeddings for TASR
        query_ins = FlowerMessageSerializer.recordset_to_query_ins(record_dict)

        start_time = time.time()
        results, doc_embeddings = chroma_store.query(
            query_text=query_ins.query,
            top_k=query_ins.top_k,
            filters=query_ins.filters or None,
        )
        latency = time.time() - start_time

        query_res = QueryRes(
            results=results,
            node_id=node_id,
            processing_time=latency,
            trust_score=1.0,
            doc_embeddings=doc_embeddings,
        )

        out_record = FlowerMessageSerializer.query_res_to_recordset(query_res)
        return msg.create_reply(content=out_record)

    return app