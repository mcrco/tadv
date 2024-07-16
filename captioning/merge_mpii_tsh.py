import json

with open('../captions/mpii_captions.json', 'r') as f:
    mpii = json.load(f)
with open('../captions/tsh_captions.json', 'r') as f:
    tsh = json.load(f)
merged = {**mpii, **tsh}
with open('../captions/mpii_tsh_captions.json', 'w') as f:
    json.dump(merged, f)
