# ModelSelect

This project improves the search and usability experience for open-source AI models on HuggingFace.  
It allows users to easily find a model and convert it to GGUF format for local use with `llama.cpp`.

---

## 📦 Required Files (EN)

This project depends on the following:

- `llama.cpp` directory (compiled locally)
- Model files in `.gguf` format
- Binary tools from `llama-b4714-bin-win-cuda-cu12.4-x64.zip`

> **Note:** Due to GitHub’s 100MB limit, large binaries (such as `.dll`, `.exe`, `.zip`) are not included in this repository. You need to download or build them manually.

### 🔧 Setup Instructions

1. Clone this repository.
2. Download and build [llama.cpp](https://github.com/ggerganov/llama.cpp) on your machine.
3. Extract and place `llama-b4714-bin-win-cuda-cu12.4-x64` contents into the project folder.
4. Obtain your desired AI model in `.gguf` format from HuggingFace.
5. Run the project:

```bash
python app.py
