# DataMate solution image.
# BASE: replace with the base image named in the organizers' Docker guideline (CUDA >= 12.8, Blackwell / sm_120).
ARG BASE=vllm/vllm-openai:v0.10.1
FROM ${BASE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY datamate /app/datamate
COPY predict.py predict.sh run_eval.py eval_sealed.py tools_perturb.py show.py validate_submission.py README.md /app/
COPY splits /app/splits
COPY tools /app/tools
COPY tests /app/tests
RUN chmod +x /app/predict.sh /app/tools/*.sh
# Model weights baked in (no network at run time). Declared in README; total must stay <= 15B (tools/count_params.py).
#   models/gen : code writer  (default Qwen2.5-Coder-7B-Instruct, 7.6B; FP8/AWQ allowed)
#   models/ver : verifier     (default Qwen3-4B, 4.0B)
COPY models/gen /opt/models/gen
COPY models/ver /opt/models/ver
ENTRYPOINT ["/bin/bash", "/app/predict.sh"]
CMD ["/data", "/result/submission.jsonl"]
