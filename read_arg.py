import os
import json

with open('args.json') as f:
    args = json.load(f)
    assert type(args) == dict
    print(args)

    print(args['lstm']['hidden_dim'])
