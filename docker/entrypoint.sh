#!/bin/sh
# Container entrypoint.
#
#   serve      (default) build the index if needed, then run the web interface
#   reminders  print the deadline digest and exit
#   ingest     run the ingestion pipeline and exit
#   shell      drop into a shell for debugging
set -e

echo "Personal Assistant AI - starting (mode: ${1:-serve})"

case "${1:-serve}" in
  serve)
    # The index lives on a mounted volume, so it survives restarts and only
    # has to be built the first time.
    if [ ! -d "data/vector_store_cp3" ]; then
      echo "No index found - running initial ingestion..."
      python src/checkpoint3_demo.py >/dev/null 2>&1 || python -c "
import sys; sys.path.insert(0, 'src')
from rag_app import RAGApplication
print(RAGApplication().ingest().describe())
"
    fi
    exec streamlit run src/interface/app.py \
      --server.address=0.0.0.0 \
      --server.port=8501 \
      --server.headless=true \
      --browser.gatherUsageStats=false
    ;;
  reminders) exec python src/reminders.py ;;
  ingest)
    exec python -c "
import sys; sys.path.insert(0, 'src')
from rag_app import RAGApplication
print(RAGApplication().ingest().describe())
"
    ;;
  shell) exec /bin/sh ;;
  *) exec "$@" ;;
esac
