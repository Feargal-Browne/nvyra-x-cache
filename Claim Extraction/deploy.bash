cd sota_project && modal deploy pipeline.py

# A: Wake up the H200 (It will poll for work)
cd sota_project && modal run pipeline.py

# B: Fire the Rust Cannon (Replace URL with yours)
cd sota_project && cargo run --release -- \
  --url "https://YOUR_USERNAME--sota-h200-final-ingest.modal.run" \
  --input "data.csv"