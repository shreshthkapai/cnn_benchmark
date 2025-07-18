import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from models.custom_cnn import create_custom_cnn
from models.resnet18 import load_resnet18
from utils.data_loader import get_cifar10_info

# CIFAR-10 preprocessing
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def preprocess_image(image):
    """Preprocess uploaded image for CIFAR-10 model."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

def predict_image(image, model_choice="CustomCNN"):
    """
    Predict class label for uploaded image.
    
    Args:
        image: PIL Image uploaded by user
        model_choice: Which model to use for prediction
        
    Returns:
        str: Formatted prediction result
    """
    try:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_choice == "CustomCNN":
            model = create_custom_cnn()
            model_path = "best_model_custom.pth"
        else:
            model = load_resnet18()
            model_path = "best_model_resnet18.pth"
        
        # Load trained weights
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except FileNotFoundError:
            return f"❌ Model weights not found: {model_path}\nTrain the model first!"
        
        model.to(device)
        model.eval()
        
        # Preprocess image
        input_tensor = preprocess_image(image).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get class info
        cifar10_info = get_cifar10_info()
        class_names = cifar10_info['class_names']
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Format result with top-3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        result = f"🎯 **Prediction: {predicted_class}** ({confidence_score:.1f}% confidence)\n\n"
        result += f"📊 **Top 3 Predictions:**\n"
        
        for i in range(3):
            class_name = class_names[top3_indices[0][i].item()]
            prob = top3_prob[0][i].item() * 100
            result += f"{i+1}. {class_name}: {prob:.1f}%\n"
        
        result += f"\n🔧 **Model:** {model_choice}\n"
        result += f"📱 **Device:** {device}"
        
        return result
        
    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nPlease ensure the image is valid and model is trained."

def create_gradio_interface():
    """Create and launch Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .output-text {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    """
    
    # Create interface
    interface = gr.Interface(
        fn=predict_image,
        inputs=[
            gr.Image(type="pil", label="Upload Image", height=300),
            gr.Dropdown(
                choices=["CustomCNN", "ResNet18"],
                value="CustomCNN",
                label="Select Model"
            )
        ],
        outputs=gr.Textbox(
            label="Prediction Result",
            lines=10,
            elem_classes=["output-text"]
        ),
        title="🧠 CIFAR-10 CNN Benchmark",
        description="""
        Upload an image to test our trained models! 
        
        **Models:**
        - **CustomCNN**: Lightweight 3M parameter model
        - **ResNet18**: Standard 11M parameter baseline
        
        **Note:** Images will be resized to 32x32 pixels (CIFAR-10 format)
        
        **CIFAR-10 Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
        """,
        css=css,
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    return interface

if __name__ == "__main__":
    # Create and launch interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )