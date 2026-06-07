# Data Samples

This directory contains small synthetic examples for running the repository demos.

SFT records use:

```json
{"instruction": "...", "input": "[1.0, 2.0]", "output": "[3.0]"}
```

DPO records use:

```json
{"instruction": "...", "input": "[1.0, 2.0]", "chosen": "[3.0]", "rejected": "[2.5]"}
```

The real paper-scale datasets are not included. Replace these files with your own authorized data using the same schema.

