import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd

sample = pd.DataFrame(
    [
        {
            "Age": 30,
            "Sex": "F",
            "BP": "NORMAL",
            "Cholesterol": "NORMAL",
            "Na_to_K": 15.0,
        }
    ]
)

# Original scikit-learn pipeline
model = joblib.load("model/drug_model.joblib")
skl_pred = model.predict(sample)[0]
skl_proba = model.predict_proba(sample)[0]
print(f"sklearn prediction: {skl_pred}")
print(f"sklearn probabilities: {skl_proba}")

# ONNX Runtime, same inputs/types the browser will send
sess = rt.InferenceSession("model/drug_model.onnx")
feeds = {
    "Age": np.array([[30]], dtype=np.int64),
    "Sex": np.array([["F"]], dtype=object),
    "BP": np.array([["NORMAL"]], dtype=object),
    "Cholesterol": np.array([["NORMAL"]], dtype=object),
    "Na_to_K": np.array([[15.0]], dtype=np.float64),
}
output = sess.run(None, feeds)
onnx_label = output[0][0]
onnx_probs = output[1][0]
print(f"onnx prediction: {onnx_label}")
print(f"onnx probabilities: {onnx_probs}")

assert skl_pred == onnx_label, f"Mismatch: {skl_pred} vs {onnx_label}"
assert np.allclose(skl_proba, onnx_probs, atol=1e-5), "Probability mismatch"
print("Verification passed! Sklearn and ONNX Runtime predictions match perfectly.")
