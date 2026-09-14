FROM prom/prometheus:v2.53.0

COPY prometheus.yml /etc/prometheus/prometheus.yml
COPY prometheus.config /etc/prometheus/prometheus.config

EXPOSE 9090

# ENTRYPOINT image dasar sudah fix ke binary prometheus dengan CMD sebagai args;
# di sini di-override total ke shell form supaya $PORT (di-inject Railway secara
# dinamis) bisa dipakai untuk --web.listen-address, sama seperti Dockerfile utama.
ENTRYPOINT []
CMD prometheus \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.console.libraries=/usr/share/prometheus/console_libraries \
    --web.console.templates=/usr/share/prometheus/consoles \
    --web.listen-address=:${PORT:-9090}
