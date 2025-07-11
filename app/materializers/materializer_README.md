## 📦 Custom Materializers

This folder contains **custom ZenML materializers** used to properly store and retrieve artifacts produced by our pipelines.

---

### 🧩 Why custom materializers?

ZenML pipelines may produce artifacts of **custom Python classes** that ZenML doesn't know how to handle by default.

For example:
- `ConvAE_model_subclass`: a custom TensorFlow model class we built.

By default, ZenML uses a **Pickle materializer**:
- Only intended for quick prototyping.
- Not robust across Python versions or changes to the class.
- Not production-ready.

---

### ✅ What this materializer does

We created a custom materializer to:
- Define **how to serialize** (`save()`) and **deserialize** (`load()`) our `ConvAE_model_subclass`.
- Use TensorFlow’s native model saving (`model.save()`) or a domain-specific format.
- Make artifacts:
  - Portable and robust across environments.
  - Loadable in downstream steps, serving APIs, or deployment.

---

### 📄 Files in this folder

| File | Purpose |
|-----|---------|
| `conv_ae_model_materializer.py` | Contains the `ConvAEModelMaterializer` that implements `save` and `load` for our custom model class. |
| `README.md` | This explanation of why we need custom materializers and what this folder contains. |

---

### 📚 Reference

For more details:
- ZenML docs: [Materializers](https://docs.zenml.io/user-guide/storing-artifacts/materializers)
- [TensorFlow model saving](https://www.tensorflow.org/guide/keras/save_and_serialize)