from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-lf-9e848996-e403-49a1-bf4a-8d0ef024f54a",
    secret_key="sk-lf-18427cd8-9e8f-4146-8686-90f9614d66b9",
    host="http://localhost:3000",   # your local server, not cloud
)

# verify the SDK can reach your local Langfuse server
if langfuse.auth_check():
    print("Langfuse connection OK")
else:
    print("Langfuse auth failed - check keys/host")
