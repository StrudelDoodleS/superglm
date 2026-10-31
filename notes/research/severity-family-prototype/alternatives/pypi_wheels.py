import json,sys
for name in ("jax","jaxlib"):
    d=json.load(open(f"{name}.json"))
    print(name,d["info"]["version"],d["info"]["license"],d["info"].get("license_expression"),d["info"]["requires_python"])
    for f in d["urls"]:
        print("  %-80s %7.1f MB"%(f["filename"],f["size"]/1e6))
