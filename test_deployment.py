import modal

app = modal.App("test-billing-check")

@app.function()
def test_function():
    print("If you see this, basic deployment works!")
    return "Success"

@app.local_entrypoint()
def main():
    print("Attempting to run test function on Modal...")
    try:
        res = test_function.remote()
        print(f"Result: {res}")
    except Exception as e:
        print(f"Deployment failed: {e}")
