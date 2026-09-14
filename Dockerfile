FROM tensorflow/serving:2.20.0

COPY serving_model/naufal_falah_a-pipeline /models/naufal_falah_a-pipeline
COPY tf_serving_monitoring.config /models/tf_serving_monitoring.config

ENV MODEL_NAME=naufal_falah_a-pipeline
ENV MODEL_BASE_PATH=/models

EXPOSE 8501

# Image dasar tensorflow/serving punya ENTRYPOINT bawaan (tf_serving_entrypoint.sh)
# yang menjalankan tensorflow_model_server versinya sendiri dengan flag default lalu
# menambahkan CMD kita sebagai argumen tambahan ke situ — bukan menggantinya. Ini bikin
# CMD custom kita (termasuk --monitoring_config_file) tidak pernah benar-benar dipakai.
# ENTRYPOINT [] mengosongkan itu supaya CMD di bawah jadi satu-satunya command yang jalan.
ENTRYPOINT []

# Shell form (bukan exec-form JSON) supaya $PORT di-expand saat container start —
# dibutuhkan Railway/Heroku yang meng-inject PORT secara dinamis per deploy.
# Lokal (docker run tanpa -e PORT) tetap fallback ke 8501.
# --monitoring_config_file mengaktifkan endpoint /monitoring/prometheus/metrics
# (nonaktif secara default di TF Serving) supaya bisa di-scrape Prometheus.
CMD tensorflow_model_server \
    --rest_api_port=${PORT:-8501} \
    --model_name=${MODEL_NAME} \
    --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
    --monitoring_config_file=/models/tf_serving_monitoring.config
