FROM tensorflow/serving:2.20.0

COPY serving_model/naufal_falah_a-pipeline /models/naufal_falah_a-pipeline

ENV MODEL_NAME=naufal_falah_a-pipeline
ENV MODEL_BASE_PATH=/models

EXPOSE 8501

# Shell form (bukan exec-form JSON) supaya $PORT di-expand saat container start —
# dibutuhkan Railway/Heroku yang meng-inject PORT secara dinamis per deploy.
# Lokal (docker run tanpa -e PORT) tetap fallback ke 8501.
CMD tensorflow_model_server \
    --rest_api_port=${PORT:-8501} \
    --model_name=${MODEL_NAME} \
    --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}
