
import requests

URL = "http://127.0.0.1:8000/query"

probes = [
    ("Raft partitions", "How does Raft handle network partitions?", True),
    ("Raft quorum", "How many nodes are required for a Raft quorum?", True),
    ("Precision recall", "What is the difference between precision and recall?", False),
    ("B-tree", "How do B-tree indexes improve database query performance?", True),
    ("OSPF", "What is the role of OSPF in computer networks?", True),
    ("AdamW", "How does AdamW stabilize adaptive parameter updates?", True),
    ("Transformers", "How do transformer encoders differ from autoregressive transformer decoders?", True),
    ("FIFA", "Who won the FIFA World Cup in 1998?", False),
]

print("\n========== RAG TEST REPORT ==========\n")

passed = 0

for name, question, should_answer in probes:
    try:
        response = requests.post(
            URL,
            json={"question": question, "top_k": 3},
            timeout=15
        )

        data = response.json()
        has_answer = data.get("answer") is not None

        ok = response.status_code == 200 and has_answer == should_answer

        if ok:
            passed += 1

        print(f"{'PASS' if ok else 'FAIL'} | {name}")
        print("Status:", response.status_code)
        print("Sources:", data.get("sources"))
        print("Confidence:", data.get("confidence"))
        print("-" * 40)

    except Exception as error:
        print(f"FAIL | {name} | {error}")

print(f"\nRESULT: {passed}/{len(probes)} tests passed")
