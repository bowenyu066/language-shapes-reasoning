from src.model_interface import GeminiModel

model = GeminiModel("Gemini-3-Pro-Preview", "gemini-3-pro-preview")

print(model.generate("Hello, how are you?"))
