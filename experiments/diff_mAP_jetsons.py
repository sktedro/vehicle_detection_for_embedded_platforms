import json

with open("test_results.json") as f:
    data = json.load(f)

model_shape = "dynamic"
backend = "tensorrt"

total = 0
broken = 0
deviations = []

for model in data["nano"]:
    for input_resolution in data["nano"][model]:
        for quantization in data["nano"][model][input_resolution][backend]:
            for batch_size in data["nano"][model][input_resolution][backend][quantization][model_shape]:
                vals = []
                for device in ["nano", "nx", "agx"]:
                    vals.append(data[device][model][input_resolution][backend][quantization][model_shape][batch_size])

                vals_all = []
                for key in vals[0]:
                    metrics = []
                    if key != "fps":
                        for i in range(len(vals)):
                            metrics.append(vals[i][key])
                        vals_all.append(metrics)

                for vals in vals_all:
                    try:
                        #  print("A")
                        total += 1
                        assert vals[0] == vals[1] and vals[1] == vals[2]
                    except:
                        vals = [100 * float(v) for v in vals]
                        broken += 1
                        dev = max(vals) - min(vals)
                        deviations.append(dev)
                        #  print(vals, max(vals) - min(vals))
                        #  print(max(vals) - min(vals))

#  print(len(total))

from pprint import pprint
import numpy as np
print(total)
print(broken)
print(len(deviations))
print(max(deviations))
print(np.mean(deviations))
print(np.median(deviations))
#  pprint(deviations)

l = [v for v in deviations if v >= 0.4]
print(len(l))
